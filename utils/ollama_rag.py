import os
import logging
import torch
import asyncio
from datetime import datetime, timedelta
from typing import TypedDict, Dict, Any, Literal

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from PyPDF2 import PdfReader

# LangGraph 관련 임포트
from langgraph.graph import StateGraph, END

from utils.config import Config

# DB 메타데이터 검색 모듈 (없으면 더미 함수)
try:
    from utils.db_full_schema import get_full_db_schema, search_db_metadata, get_all_table_names
except ImportError:
    def get_full_db_schema(): return []
    def search_db_metadata(k): return ""
    def get_all_table_names(): return ""

# 로거 설정
logger = logging.getLogger(__name__)
# 필요 시 로그 레벨 강제 조정 (메인에서 설정되어 있다면 생략 가능)
logger.setLevel(logging.INFO)


# ==========================================================================
# 0. Prompts
# ==========================================================================
SQL_SYSTEM_PROMPT = """
    You are an expert Oracle SQL architect.

    You MUST generate SQL through the following strict 7-step reasoning process.

    ### STEP 1: 규정 분석
    - 사용자의 질문에서 어떤 계산식, 조건, 정의가 필요한지 규정 문서로부터 추출한다.

    ### STEP 2: 필요 데이터 요소 도출
    - 규정에서 필요한 데이터 요소를 나열한다. 
    (예: 거래량, SMP, 단가, 정산금액, 발전기 ID 등)

    ### STEP 3: DB 스키마 매핑
    - 제공된 DB 스키마에서 각 데이터 요소가 어느 테이블/컬럼에 있는지 명확히 매핑한다.
    - 없는 경우 "존재하지 않음"이라고 표시 (지어내기 금지)

    ### STEP 4: JOIN 계획 생성
    - 어떤 테이블을 어떤 조건으로 JOIN해야 하는지 단계별로 설명한다.

    ### STEP 5: SQL 초안 조립
    - Oracle SQL 문법으로 SELECT, JOIN, WHERE, GROUP BY를 작성한다.

    ### STEP 6: 정합성 검증
    - 테이블/컬럼이 스키마에 실제로 존재하는지 확인한다.
    - 문제 있으면 수정한다.

    ### STEP 7: 최종 SQL 출력
    - 최종 SQL만 코드 블록으로 출력한다.

    You MUST follow all 7 steps exactly.
"""

VALIDATOR_SYSTEM_PROMPT = """
    당신은 AI 답변 평가자입니다.
    사용자의 [질문]과 AI가 생성한 [답변]을 보고, 답변이 질문 의도에 부합하고 정확한지 평가하세요.

    다음 기준을 따르세요:
    1. 답변이 질문에 직접적으로 대답하고 있는가?
    2. '모르겠습니다' 혹은 무의미한 내용이 아닌가?
    3. (SQL 생성 요청인 경우) SQL 코드가 포함되어 있는가?

    평가 결과는 반드시 다음 형식으로만 출력하세요:
    STATUS: [PASS 또는 FAIL]
    REASON: [평가 이유 및 개선 피드백 한 문장]
"""


# ==========================================================================
# 1. 전역 변수 & 설정
# ==========================================================================
embeddings = None
db_schema_vectorstore = None
doc_vectorstore = None

# 세션 저장소
store = {}
SESSION_TIMEOUT_MINUTES = 60

# LLM 초기화
llm = ChatOllama(
    model=Config.OLLAMA_MODEL,
    temperature=0.1,
    base_url=Config.OLLAMA_BASE_URL
)


# ==========================================================================
# 2. 세션 및 유틸리티
# ==========================================================================
def get_session_history(session_id: str):
    now = datetime.now()
    if session_id not in store:
        store[session_id] = { "history": ChatMessageHistory(), "last_access": now }
    store[session_id]["last_access"] = now

    history = store[session_id]["history"]

    MAX_HISTORY = 20
    if len(history.messages) > MAX_HISTORY:
        overflow = len(history.messages) - MAX_HISTORY
        history.messages = history.messages[overflow:]

    return history


async def cleanup_expired_sessions():
    while True:
        try:
            await asyncio.sleep(600)
            now = datetime.now()
            expired = [sid for sid, data in store.items()
                       if now - data["last_access"] > timedelta(minutes=SESSION_TIMEOUT_MINUTES)]
            for sid in expired:
                del store[sid]
            if expired:
                logger.info(f"🧹 만료된 세션 {len(expired)}개 삭제됨")
        except Exception as e:
            logger.error(f"세션 청소 오류: {e}")


def invoke_chain_with_history(system_prompt: str, user_question: str, context: str, session_id: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("system", "[Context]\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    chain_with_hist = RunnableWithMessageHistory(
        chain, get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history"
    )
    return chain_with_hist.invoke(
        {"question": user_question, "context": context},
        config={"configurable": {"session_id": session_id}}
    )


# ==========================================================================
# 3. 벡터스토어 초기화
# ==========================================================================
def initialize_all_vectorstores():
    global embeddings, db_schema_vectorstore, doc_vectorstore
    logger.info("🚀 [Init] 벡터 스토어 초기화 시작…")

    # Embeddings
    if embeddings is None:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            embeddings = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL_PATH,
                model_kwargs={"device": device}
            )
        except Exception as e:
            logger.error(f"임베딩 로딩 실패: {e}")
            return

    # DB Schema VectorStore
    if not os.path.exists(Config.DB_SCHEMA_VECTORSTORE_PATH):
        os.makedirs(Config.DB_SCHEMA_VECTORSTORE_PATH, exist_ok=True)

    idx_path = os.path.join(Config.DB_SCHEMA_VECTORSTORE_PATH, "index.faiss")
    if os.path.exists(idx_path):
        try:
            db_schema_vectorstore = FAISS.load_local(
                Config.DB_SCHEMA_VECTORSTORE_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("✅ [Init] DB Schema VectorStore 로드 완료")
        except Exception as e:
            logger.error(f"❌ DB Schema 로드 실패: {e}")
    else:
        docs = get_full_db_schema()
        if docs:
            lc_docs = [Document(page_content=d["content"], metadata={"name": d["name"]}) for d in docs]
            splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            db_schema_vectorstore = FAISS.from_documents(splitter.split_documents(lc_docs), embeddings)
            db_schema_vectorstore.save_local(Config.DB_SCHEMA_VECTORSTORE_PATH)
            logger.info("✨ [Init] DB Schema VectorStore 생성 완료")


    # Rule Doc VectorStore
    if not os.path.exists(Config.DOC_VECTORSTORE_PATH):
        os.makedirs(Config.DOC_VECTORSTORE_PATH, exist_ok=True)

    doc_index = os.path.join(Config.DOC_VECTORSTORE_PATH, "index.faiss")
    if os.path.exists(doc_index):
        try:
            doc_vectorstore = FAISS.load_local(
                Config.DOC_VECTORSTORE_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("✅ [Init] Rule Doc 로드 완료")
        except Exception as e:
            logger.error(f"Rule Doc 로드 실패: {e}")
    else:
        if os.path.exists(Config.PDF_FILE_PATH):
            raw = extract_text_from_pdf(Config.PDF_FILE_PATH)
            splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
            docs = [d for d in splitter.create_documents([raw]) if len(d.page_content.strip()) > 80]
            if docs:
                doc_vectorstore = FAISS.from_documents(docs, embeddings)
                doc_vectorstore.save_local(Config.DOC_VECTORSTORE_PATH)
                logger.info("✨ [Init] Rule Doc VectorStore 생성 완료")


def extract_text_from_pdf(path: str) -> str:
    text = ""
    try:
        with open(path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                pt = page.extract_text()
                if pt:
                    text += pt.replace("-\n", "").replace("\n", " ").strip() + "\n"
    except:
        pass
    return text


# ==========================================================================
# 4. Intent Classifier & Helpers
# ==========================================================================
def classify_intent_logic(question: str, has_file=False, file_snippet=None) -> str:
    file_info = "No File"
    if has_file:
        snippet = file_snippet[:300] if file_snippet else ""
        file_info = f"File Uploaded. Snippet: '{snippet}...'"

    router_prompt = f"""
    You are an AI Intent Classifier.
    [Context] Query: "{question}", File: {file_info}

    Classify into EXACTLY ONE category:

    1. FILE_ONLY: User asks about the uploaded file. (Valid ONLY if File exists)
    2. VERSION_COMPARE: User compares uploaded file vs existing rules. (Valid ONLY if File exists)
    
    3. CROSS_CHECK: 
       - User wants to COMPARE 'Market Rules' vs 'DB Schema'.
       - OR mentions BOTH "Document/Rule" AND "DB/Schema".

    4. DB_DESIGN:
       - User asks to DESIGN, CREATE, or MODEL a new table/procedure based on Rules.
       - Keywords: "Create table", "Design schema", "Write DDL", "DB Modeling".

    5. CODE_ANALYSIS: User pasted raw code/text (No file).
    
    6. DB_SCHEMA: 
       - General DB questions (Finding tables, columns).
       - Searching for data in a specific table (e.g. "Find in ete100t").
       
    7. RULE_DOC: General Rule questions.
    8. GENERAL: General chat.

    Output ONLY category name.
    """
    try:
        intent = llm.invoke(router_prompt).content.strip()
        valid = ["FILE_ONLY", "VERSION_COMPARE", "CROSS_CHECK", "DB_DESIGN", "CODE_ANALYSIS", "DB_SCHEMA", "RULE_DOC", "GENERAL"]
        for v in valid:
            if v in intent: return v
        return "FILE_ONLY" if has_file else "GENERAL"
    except Exception:
        return "FILE_ONLY" if has_file else "GENERAL"


def extract_keyword(question: str):
    return llm.invoke(f"질문: '{question}' 핵심 키워드 하나만 추출. 없으면 FALSE").content.strip()


def generate_sql_step_by_step(question: str, rule_context: str, db_context: str, session_id: str):
    prompt = f"""
[사용자 질문]
{question}

[규정]
{rule_context}

[DB 스키마]
{db_context}
    """
    return invoke_chain_with_history(SQL_SYSTEM_PROMPT, question, prompt, session_id)


# ==========================================================================
# 5. Handler Functions
# ==========================================================================
def log_task_start(name: str, attempts: int):
    """작업 시작 공통 로그"""
    prefix = "▶️ [First]" if attempts == 0 else f"🔄 [Retry {attempts}]"
    logger.info(f"{prefix} Node 실행: {name}")

def rag_for_db_design(question: str, session_id="default"):
    rule_ctx = ""
    if doc_vectorstore:
        rule_docs = doc_vectorstore.similarity_search(question, k=5)
        rule_ctx = "\n".join([d.page_content for d in rule_docs])

    db_ctx = ""
    if db_schema_vectorstore:
        db_docs = db_schema_vectorstore.similarity_search(question, k=5)
        db_ctx = "\n".join([d.page_content for d in db_docs])

    sql_result = generate_sql_step_by_step(question, rule_ctx, db_ctx, session_id)

    system = """
    당신은 수석 DB 아키텍트입니다.
    규정 기반으로 신규 Oracle 테이블 생성 DDL과 설계 근거를 설명하세요.
    """
    modeling_result = invoke_chain_with_history(system, question, rule_ctx + "\n\n" + db_ctx, session_id)

    return f"""
=========================
📌 Step-by-Step SQL 생성 결과
=========================
{sql_result}

=========================
📌 규정 기반 데이터 모델 제안
=========================
{modeling_result}
"""


def rag_for_uploaded_files(question, file_context, session_id):
    if len(file_context) < 30000:
        return invoke_chain_with_history("파일 내용을 분석하세요.", question, file_context, session_id)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([file_context])
    temp_vs = FAISS.from_documents(docs, embeddings)
    res = temp_vs.similarity_search(question or "요약", k=8)
    context = "\n".join([d.page_content for d in res])
    return invoke_chain_with_history("파일 발췌본 기반", question, context, session_id)


def rag_for_version_comparison(question, file_context, session_id):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([file_context])
    temp_vs = FAISS.from_documents(docs, embeddings)

    search_q = question if len(question) > 5 else "변경"
    old_ctx = ""
    if doc_vectorstore:
        old_docs = doc_vectorstore.similarity_search(search_q, k=5)
        old_ctx = "\n".join([d.page_content for d in old_docs])

    new_docs = temp_vs.similarity_search(search_q, k=5)
    new_ctx = "\n".join([d.page_content for d in new_docs])

    return invoke_chain_with_history(
        "기존 규정과 신규 파일을 비교하세요.",
        question,
        f"[OLD]\n{old_ctx}\n\n[NEW]\n{new_ctx}",
        session_id,
    )


def rag_for_cross_check(question, session_id, file_context=None):
    rule_ctx = ""
    if doc_vectorstore:
        docs = doc_vectorstore.similarity_search(question, k=5)
        rule_ctx = "\n".join([d.page_content for d in docs])

    db_ctx = ""
    kw = extract_keyword(question)
    if kw != "FALSE":
        db_ctx += search_db_metadata(kw)

    if db_schema_vectorstore:
        docs = db_schema_vectorstore.similarity_search(question, k=5)
        db_ctx += "\n" + "\n".join([d.page_content for d in docs])

    file_info = f"[FILE]\n{file_context[:2000]}" if file_context else ""

    return invoke_chain_with_history(
        "규정 vs DB 정합성 점검",
        question,
        f"{file_info}\n\n[규정]\n{rule_ctx}\n\n[DB]\n{db_ctx}",
        session_id,
    )


def analyze_code_context(question, full_context, session_id):
    return invoke_chain_with_history("코드 분석 전문가", question, full_context, session_id)


def rag_for_db_schema(question, session_id="default"):
    if any(kw in question.lower() for kw in ["sql", "쿼리", "select", "join", "ddl", "dml"]):
        rule_ctx = ""
        if doc_vectorstore:
            docs = doc_vectorstore.similarity_search(question, k=5)
            rule_ctx = "\n".join([d.page_content for d in docs])

        db_ctx = ""
        if db_schema_vectorstore:
            docs = db_schema_vectorstore.similarity_search(question, k=5)
            db_ctx = "\n".join([d.page_content for d in docs])

        return generate_sql_step_by_step(question, rule_ctx, db_ctx, session_id)

    keyword = extract_keyword(question)
    context = ""

    if keyword != "FALSE" and len(keyword) > 1:
        context += f"[DB 메타데이터]\n{search_db_metadata(keyword)}\n"
        if db_schema_vectorstore:
            docs = db_schema_vectorstore.similarity_search(question, k=8)
            context += "[유사도]\n" + "\n".join([d.page_content for d in docs])
    else:
        docs = db_schema_vectorstore.similarity_search(question, k=10)
        context = "\n".join([d.page_content for d in docs])

    return invoke_chain_with_history("Oracle DB 전문가", question, context, session_id)


def rag_for_rules(question, session_id):
    context = ""
    if doc_vectorstore:
        docs = doc_vectorstore.similarity_search(question, k=10)
        context = "\n".join([d.page_content for d in docs])
    return invoke_chain_with_history("규정 전문가", question, context, session_id)


def ask_llm_general(question, session_id):
    return invoke_chain_with_history("도움이 되는 AI", question, "", session_id)


# ==========================================================================
# 6. LangGraph Definition (Verified Loop with Logging & Path Map)
# ==========================================================================

# 6.1 State 정의
class AgentState(TypedDict):
    question: str
    session_id: str
    file_context: str
    has_file: bool
    intent: str
    answer: str
    attempts: int       # 재시도 횟수
    feedback: str       # Validator 피드백

# 6.2 Helper: 질문 강화 (Feedback Injection)
def enhance_query_with_feedback(state: AgentState) -> str:
    """재시도인 경우, 피드백을 반영해 질문을 수정"""
    query = state["question"]
    if state["attempts"] > 0 and state.get("feedback"):
        # 📝 로그: 피드백 반영 확인
        logger.info(f"🔄 [Loop] 피드백 반영 중: '{state['feedback']}'")
        
        enhanced_query = f"""
        {query}
        
        [이전 시도에 대한 피드백]
        이전 답변이 다음 이유로 부족했습니다: "{state['feedback']}"
        이 피드백을 반영하여 답변을 개선하거나 수정해 주세요.
        """
        return enhanced_query
    return query

# 6.3 Router Node
def router_node(state: AgentState):
    query = state["question"]
    file_snippet = state["file_context"][:500] if state["file_context"] else None
    
    # 🚦 로그: 라우터 시작
    logger.info(f"🚦 [Router] 질문 분석 중: '{query[:50]}...'")
    
    intent = classify_intent_logic(query, state["has_file"], file_snippet)
    
    # 🔀 로그: 분류된 의도
    logger.info(f"🔀 [Router] 분류 결과: {intent}")
    
    return {
        "intent": intent,
        "attempts": 0,
        "feedback": ""
    }

# 6.4 Task Wrapper Nodes (피드백 반영 & 로그 추가)
def file_only_node(state: AgentState):
    log_task_start("FILE_ONLY", state["attempts"])
    q = enhance_query_with_feedback(state)
    return {"answer": rag_for_uploaded_files(q, state["file_context"], state["session_id"]), "attempts": state["attempts"] + 1}

def version_compare_node(state: AgentState):
    log_task_start("VERSION_COMPARE", state["attempts"])
    q = enhance_query_with_feedback(state)
    return {"answer": rag_for_version_comparison(q, state["file_context"], state["session_id"]), "attempts": state["attempts"] + 1}

def cross_check_node(state: AgentState):
    log_task_start("CROSS_CHECK", state["attempts"])
    q = enhance_query_with_feedback(state)
    return {"answer": rag_for_cross_check(q, state["session_id"], state["file_context"]), "attempts": state["attempts"] + 1}

def db_design_node(state: AgentState):
    log_task_start("DB_DESIGN", state["attempts"])
    q = enhance_query_with_feedback(state)
    return {"answer": rag_for_db_design(q, state["session_id"]), "attempts": state["attempts"] + 1}

def code_analysis_node(state: AgentState):
    log_task_start("CODE_ANALYSIS", state["attempts"])
    q = enhance_query_with_feedback(state)
    return {"answer": analyze_code_context(q, state["file_context"], state["session_id"]), "attempts": state["attempts"] + 1}

def db_schema_node(state: AgentState):
    log_task_start("DB_SCHEMA", state["attempts"])
    q = enhance_query_with_feedback(state)
    return {"answer": rag_for_db_schema(q, state["session_id"]), "attempts": state["attempts"] + 1}

def rule_doc_node(state: AgentState):
    log_task_start("RULE_DOC", state["attempts"])
    q = enhance_query_with_feedback(state)
    return {"answer": rag_for_rules(q, state["session_id"]), "attempts": state["attempts"] + 1}

def general_node(state: AgentState):
    log_task_start("GENERAL", state["attempts"])
    return {"answer": ask_llm_general(state["question"], state["session_id"]), "attempts": state["attempts"] + 1}

# 6.5 Validator Node
def validator_node(state: AgentState):
    """답변 평가 및 재시도 결정"""
    current_answer = state["answer"]
    intent = state["intent"]
    
    # 🧐 로그: 검증 시작
    logger.info(f"🧐 [Validator] 답변 평가 중... (Intent: {intent})")
    
    # 간단한 검증 (일반 대화나 너무 짧은 답변은 패스)
    if intent == "GENERAL" or len(current_answer) < 10:
        logger.info("⏩ [Validator] 일반 대화/단답형이므로 즉시 통과")
        return {"feedback": "PASS"}

    validation_prompt = f"""
    [질문]: {state["question"]}
    [의도]: {intent}
    [AI 답변]: {current_answer}
    """
    
    try:
        # 평가 수행
        result = invoke_chain_with_history(VALIDATOR_SYSTEM_PROMPT, "평가하라", validation_prompt, "validator_session")
        
        if "FAIL" in result:
            reason = result.split("REASON:")[-1].strip() if "REASON:" in result else "정확도 부족"
            # ⚠️ 로그: 검증 실패
            logger.warning(f"⚠️ [Validator] REJECTED: {reason}")
            return {"feedback": reason}
        else:
            # ✅ 로그: 검증 통과
            logger.info("✅ [Validator] PASSED")
            return {"feedback": "PASS"}
            
    except Exception as e:
        logger.error(f"Validator Error: {e}")
        return {"feedback": "PASS"}

# 6.6 Conditional Logic Helpers
def should_retry(state: AgentState) -> Literal["retry", "end"]:
    feedback = state.get("feedback", "PASS")
    attempts = state["attempts"]
    MAX_RETRIES = 2 

    if feedback == "PASS":
        logger.info("🏁 [Edge] 검증 통과 -> 종료(End)")
        return "end"
    
    if attempts >= MAX_RETRIES:
        logger.info(f"🛑 [Edge] 최대 재시도({MAX_RETRIES}) 도달 -> 종료(End)")
        return "end"
    
    logger.info(f"🔙 [Edge] 재시도 필요 -> 다시 {state['intent']} 노드로 이동")
    return "retry"

def route_back_to_intent(state: AgentState):
    intent_map = {
        "FILE_ONLY": "file_only",
        "VERSION_COMPARE": "version_compare",
        "CROSS_CHECK": "cross_check",
        "DB_DESIGN": "db_design",
        "CODE_ANALYSIS": "code_analysis",
        "DB_SCHEMA": "db_schema",
        "RULE_DOC": "rule_doc",
        "GENERAL": "general"
    }
    return intent_map.get(state["intent"], "general")

# 6.7 Graph Build
def build_rag_graph():
    workflow = StateGraph(AgentState)

    # 노드 등록
    workflow.add_node("router", router_node)
    workflow.add_node("file_only", file_only_node)
    workflow.add_node("version_compare", version_compare_node)
    workflow.add_node("cross_check", cross_check_node)
    workflow.add_node("db_design", db_design_node)
    workflow.add_node("code_analysis", code_analysis_node)
    workflow.add_node("db_schema", db_schema_node)
    workflow.add_node("rule_doc", rule_doc_node)
    workflow.add_node("general", general_node)
    workflow.add_node("validator", validator_node)

    # Entry Point
    workflow.set_entry_point("router")

    # Router -> Tasks
    intent_map = {
        "FILE_ONLY": "file_only",
        "VERSION_COMPARE": "version_compare",
        "CROSS_CHECK": "cross_check",
        "DB_DESIGN": "db_design",
        "CODE_ANALYSIS": "code_analysis",
        "DB_SCHEMA": "db_schema",
        "RULE_DOC": "rule_doc",
        "GENERAL": "general"
    }
    
    workflow.add_conditional_edges("router", lambda x: x["intent"], intent_map)

    # Tasks -> Validator
    for node_name in intent_map.values():
        workflow.add_edge(node_name, "validator")

    # Validator -> Loop or End (여기에 Explicit Mapping 추가)
    def validator_router(state):
        decision = should_retry(state)
        if decision == "end":
            return "end"
        else:
            return route_back_to_intent(state)

    path_map = {
        "end": END,
        "file_only": "file_only",
        "version_compare": "version_compare",
        "cross_check": "cross_check",
        "db_design": "db_design",
        "code_analysis": "code_analysis",
        "db_schema": "db_schema",
        "rule_doc": "rule_doc",
        "general": "general"
    }

    workflow.add_conditional_edges("validator", validator_router, path_map)

    return workflow.compile()

rag_graph = build_rag_graph()


# ==========================================================================
# 7. Main Execution Interface
# ==========================================================================
def execute_rag_task(query: str, session_id: str, file_context: str = "", has_file: bool = False) -> Dict[str, Any]:
    try:
        # 🚀 로그: 전체 프로세스 시작
        logger.info("="*50)
        logger.info(f"🚀 [LangGraph Start] New Request (Session: {session_id})")
        logger.info("="*50)

        initial_state = {
            "question": query,
            "session_id": session_id,
            "file_context": file_context if file_context else "",
            "has_file": has_file,
            "intent": "GENERAL",
            "answer": "",
            "attempts": 0,
            "feedback": ""
        }

        # 그래프 실행
        result = rag_graph.invoke(initial_state)
        
        final_intent = result.get("intent", "GENERAL")
        final_answer = result.get("answer", "No Answer")
        attempts = result.get("attempts", 1)

        if attempts > 1:
            logger.info(f"✨ [Done] 품질 향상 루프 완료 (총 {attempts}회 시도)")
        else:
            logger.info("✨ [Done] 단일 시도 완료")

        return {
            "intent": final_intent,
            "answer": final_answer
        }

    except Exception as e:
        logger.error(f"❌ [LangGraph Error] Execution Failed: {e}")
        return {"intent": "ERROR", "answer": f"시스템 오류 발생: {e}"}