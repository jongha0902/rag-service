import os
import logging
import torch
import asyncio
from datetime import datetime, timedelta
from typing import TypedDict, Dict, Any, Literal, List, Optional

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
logger.setLevel(logging.INFO)


# ==========================================================================
# 0. Prompts
# ==========================================================================
SQL_SYSTEM_PROMPT = """
    You are an expert Oracle SQL architect.
    Generate SQL following this strict 7-step reasoning process:

    1. Analyze Rule: Extract calculations/conditions from rules.
    2. Identify Data Elements: List required fields (e.g., SMP, Generators).
    3. Map to Schema: Find tables/columns in the provided schema.
    4. Plan JOINs: Describe JOIN logic.
    5. Draft SQL: Write SELECT, JOIN, WHERE, GROUP BY.
    6. Verify: Check if columns exist in schema.
    7. Final SQL: Output ONLY the SQL code block.

    If schema info is missing, state "MISSING SCHEMA".
"""

VALIDATOR_SYSTEM_PROMPT = """
    당신은 엄격한 AI 답변 감사관(Auditor)입니다.
    제공된 [검색된 근거 문서]를 바탕으로 AI의 [답변]이 정확한지 검증하십시오.

    ### 검증 기준 (Checklist):
    1. **근거 일치 여부 (Groundedness):** 답변의 모든 내용은 오직 [검색된 근거 문서]에 포함된 정보여야 합니다. 문서에 없는 내용을 지어냈다면 FAIL입니다.
    2. **질문 해결 여부 (Relevance):** 사용자의 질문에 대해 동문서답하지 않고 명확한 결론을 제시했습니까?
    3. **형식 준수 (Format):** (SQL 생성 요청인 경우) 유효한 SQL 구문이 코드 블록으로 포함되어 있습니까?
    4. **회피성 답변 방지:** "문서에 없습니다"라고 답해야 할 상황이 아닌데도 불필요하게 "모르겠습니다"라고 하지 않았습니까?

    ### 평가 결과 출력 형식:
    STATUS: [PASS] 또는 [FAIL]
    REASON: [FAIL인 경우, 구체적으로 문서의 어느 부분과 불일치하는지 또는 무엇이 부족한지 설명]
"""


# ==========================================================================
# 1. 전역 변수 & 설정
# ==========================================================================
embeddings = None
db_schema_vectorstore = None
doc_vectorstore = None

# 세션 저장소 (In-Memory -> Redis 등으로 교체 권장)
store = {}
SESSION_TIMEOUT_MINUTES = 60

# LLM 초기화
llm = ChatOllama(
    model=Config.OLLAMA_MODEL,
    temperature=0.1,
    base_url=Config.OLLAMA_BASE_URL
)


# ==========================================================================
# 2. 세션 및 유틸리티 (Async)
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
            await asyncio.sleep(600) # 초
            now = datetime.now()
            expired = [sid for sid, data in store.items()
                       if now - data["last_access"] > timedelta(minutes=SESSION_TIMEOUT_MINUTES)]
            for sid in expired:
                del store[sid]
            if expired:
                logger.info(f"🧹 만료된 세션 {len(expired)}개 삭제됨")
        except Exception as e:
            logger.error(f"세션 청소 오류: {e}")

# ⚡ [Async] LLM 호출 헬퍼
async def ainvoke_chain_with_history(system_prompt: str, user_question: str, context: str, session_id: str):
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
    # 비동기 호출 (.ainvoke)
    return await chain_with_hist.ainvoke(
        {"question": user_question, "context": context},
        config={"configurable": {"session_id": session_id}}
    )

# ⚡ [Async] 벡터 검색 헬퍼 (CPU 블로킹 방지)
async def async_similarity_search(vectorstore, query, k=5):
    if not vectorstore:
        return []
    # FAISS 검색은 CPU 연산이므로 별도 스레드에서 실행
    return await asyncio.to_thread(vectorstore.similarity_search, query, k=k)


# ==========================================================================
# 3. 벡터스토어 초기화
# ==========================================================================
def initialize_all_vectorstores():
    """
    초기화는 앱 구동 시 1회 실행되므로 Sync 유지 가능하나,
    필요 시 async로 변경 후 main.py lifespan에서 await 할 수 있음.
    여기서는 기존 구조 유지를 위해 Sync로 두되, 내부 로직만 최적화.
    """
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
# 4. Intent Classifier & Logic Helpers (Async)
# ==========================================================================
async def classify_intent_logic(question: str, has_file=False, file_snippet=None, feedback=None) -> str:
    file_info = "No File"
    if has_file:
        snippet = file_snippet[:300] if file_snippet else ""
        file_info = f"File Uploaded. Snippet: '{snippet}...'"

    # 🧩 Priority 4: 복합 의도 및 Feedback 반영
    feedback_ctx = ""
    if feedback:
        feedback_ctx = f"NOTE: Previous attempt failed. Reason: '{feedback}'. Please Re-Classify carefully."

    router_prompt = f"""
    You are an AI Intent Router.
    [Context] Query: "{question}"
    [File Info] {file_info}
    [Feedback] {feedback_ctx}

    Classify into ONE category:

    1. FILE_ONLY: Question *solely* about the uploaded file content.
    2. VERSION_COMPARE: Compare uploaded file vs existing rules.
    3. CROSS_CHECK: 
       - Requires BOTH Rule Documents AND DB Schema.
       - Complex queries like "Find rule for X and Query Y from DB".
    4. DB_DESIGN: Create/Model new tables/DDL.
    5. CODE_ANALYSIS: Raw code text provided.
    6. DB_SCHEMA: Searching tables, columns, or generating SQL.
    7. RULE_DOC: General regulation/rule questions.
    8. GENERAL: Casual chat.

    Output ONLY category name.
    """
    try:
        result = await llm.ainvoke(router_prompt)
        intent = result.content.strip()
        valid = ["FILE_ONLY", "VERSION_COMPARE", "CROSS_CHECK", "DB_DESIGN", "CODE_ANALYSIS", "DB_SCHEMA", "RULE_DOC", "GENERAL"]
        
        # 가장 매칭되는 의도 찾기
        for v in valid:
            if v in intent: return v
        
        return "FILE_ONLY" if has_file else "GENERAL"
    except Exception:
        return "FILE_ONLY" if has_file else "GENERAL"


async def extract_keyword(question: str):
    res = await llm.ainvoke(f"질문: '{question}' 핵심 키워드 하나만 추출. 없으면 FALSE")
    return res.content.strip()


async def generate_sql_step_by_step(question: str, rule_context: str, db_context: str, session_id: str):
    prompt = f"""
        [사용자 질문] {question}
        [규정] {rule_context}
        [DB 스키마] {db_context}
    """
    return await ainvoke_chain_with_history(SQL_SYSTEM_PROMPT, question, prompt, session_id)


# ==========================================================================
# 5. Handler Functions (Async) - Return Dict with 'answer' and 'context'
# ==========================================================================
def log_task_start(name: str, attempts: int):
    prefix = "▶️ [First]" if attempts == 0 else f"🔄 [Retry {attempts}]"
    logger.info(f"{prefix} Node 실행: {name}")

async def rag_for_db_design(question: str, session_id="default"):
    rule_docs = await async_similarity_search(doc_vectorstore, question, k=5)
    rule_ctx = "\n".join([d.page_content for d in rule_docs])

    db_docs = await async_similarity_search(db_schema_vectorstore, question, k=5)
    db_ctx = "\n".join([d.page_content for d in db_docs])

    # 문맥 저장
    full_ctx = f"[Rule]\n{rule_ctx}\n\n[DB Schema]\n{db_ctx}"

    sql_result = await generate_sql_step_by_step(question, rule_ctx, db_ctx, session_id)
    
    system = "당신은 수석 DB 아키텍트입니다. 규정 기반으로 신규 테이블 DDL과 설계 근거를 설명하세요."
    modeling_result = await ainvoke_chain_with_history(system, question, full_ctx, session_id)

    return {
        "answer": f"📌 [SQL Draft]\n{sql_result}\n\n📌 [Design]\n{modeling_result}",
        "context": full_ctx
    }

async def rag_for_uploaded_files(question, file_context, session_id):
    # 긴 파일은 임시 처리 (실제로는 Chunking 필요)
    used_context = file_context[:10000] + "..." if len(file_context) > 10000 else file_context
    ans = await ainvoke_chain_with_history("파일 내용 분석", question, used_context, session_id)
    return {"answer": ans, "context": used_context}

async def rag_for_version_comparison(question, file_context, session_id):
    search_q = question if len(question) > 5 else "변경"
    old_docs = await async_similarity_search(doc_vectorstore, search_q, k=5)
    old_ctx = "\n".join([d.page_content for d in old_docs])
    
    full_ctx = f"[OLD Rules]\n{old_ctx}\n\n[NEW File]\n{file_context[:5000]}..."
    
    ans = await ainvoke_chain_with_history(
        "기존 규정과 신규 파일 비교", question, full_ctx, session_id
    )
    return {"answer": ans, "context": full_ctx}

async def rag_for_cross_check(question, session_id, file_context=None):
    # ⚡ 병렬 검색 (Rule + DB)
    rule_task = async_similarity_search(doc_vectorstore, question, k=5)
    db_task = async_similarity_search(db_schema_vectorstore, question, k=5)
    
    rule_docs, db_schema_docs = await asyncio.gather(rule_task, db_task)
    
    rule_ctx = "\n".join([d.page_content for d in rule_docs])
    db_ctx = "\n".join([d.page_content for d in db_schema_docs])
    
    kw = await extract_keyword(question)
    if kw != "FALSE":
        db_ctx += "\n" + search_db_metadata(kw)

    file_info = f"[FILE]\n{file_context[:2000]}" if file_context else ""
    full_ctx = f"{file_info}\n\n[규정]\n{rule_ctx}\n\n[DB 스키마]\n{db_ctx}"

    ans = await ainvoke_chain_with_history(
        "규정(Rule)과 DB 스키마 간의 정합성/매핑 분석", question, full_ctx, session_id
    )
    return {"answer": ans, "context": full_ctx}

async def analyze_code_context(question, full_context, session_id):
    ans = await ainvoke_chain_with_history("코드 분석", question, full_context, session_id)
    return {"answer": ans, "context": full_context}

async def rag_for_db_schema(question, session_id="default"):
    # 1. SQL 생성 요청인 경우
    if any(kw in question.lower() for kw in ["sql", "쿼리", "select", "ddl"]):
        rule_docs = await async_similarity_search(doc_vectorstore, question, k=5)
        db_docs = await async_similarity_search(db_schema_vectorstore, question, k=5)
        
        rule_ctx = "\n".join([d.page_content for d in rule_docs])
        db_ctx = "\n".join([d.page_content for d in db_docs])
        full_ctx = f"[Rule]\n{rule_ctx}\n\n[DB Schema]\n{db_ctx}"
        
        ans = await generate_sql_step_by_step(question, rule_ctx, db_ctx, session_id)
        return {"answer": ans, "context": full_ctx}

    # 2. 일반 DB 질문인 경우
    docs = await async_similarity_search(db_schema_vectorstore, question, k=8)
    full_ctx = "\n".join([d.page_content for d in docs])
    
    ans = await ainvoke_chain_with_history("DB 전문가", question, full_ctx, session_id)
    return {"answer": ans, "context": full_ctx}

async def rag_for_rules(question, session_id):
    docs = await async_similarity_search(doc_vectorstore, question, k=10)
    full_ctx = "\n".join([d.page_content for d in docs])
    ans = await ainvoke_chain_with_history("규정 전문가", question, full_ctx, session_id)
    return {"answer": ans, "context": full_ctx}

async def ask_llm_general(question, session_id):
    ans = await ainvoke_chain_with_history("도움이 되는 AI", question, "", session_id)
    return {"answer": ans, "context": "General Chat (No specific context)"}


# ==========================================================================
# 6. LangGraph Definition (Async & Enhanced Flow)
# ==========================================================================

class AgentState(TypedDict):
    question: str
    session_id: str
    file_context: str
    has_file: bool
    intent: str
    answer: str
    attempts: int
    feedback: str
    context: str # 👈 [필수] 검증 단계에서 사용할 검색된 근거 문서

def enhance_query_with_feedback(state: AgentState) -> str:
    query = state["question"]
    if state["attempts"] > 0 and state.get("feedback"):
        logger.info(f"🔄 [Loop] 질문 개선(피드백 반영): '{state['feedback']}'")
        return f"{query}\n[Feedback to reflect]: {state['feedback']}\nPlease Improve answer."
    return query

# ⚡ [Async] Router Node
async def router_node(state: AgentState):
    query = state["question"]
    
    # 재시도인 경우 attempts 유지, 아니면 0으로 초기화하지 않음 (그래프 진입 시 초기화됨)
    current_attempts = state.get("attempts", 0)
    feedback = state.get("feedback", "")
    
    # 피드백이 있으면 질문을 수정하거나 Intent를 변경할 수 있음
    intent = await classify_intent_logic(query, state["has_file"], state["file_context"], feedback)
    
    logger.info(f"🔀 [Router] Intent: {intent} (Attempts: {current_attempts})")
    
    return {
        "intent": intent,
        "attempts": current_attempts, # 카운트 유지
        "feedback": "" # 라우팅 후 피드백 초기화 (새 시도 시작)
    }

# ⚡ [Async] Task Nodes (Dict Unpacking)
async def file_only_node(state: AgentState):
    log_task_start("FILE_ONLY", state["attempts"])
    q = enhance_query_with_feedback(state)
    res = await rag_for_uploaded_files(q, state["file_context"], state["session_id"])
    return {"answer": res["answer"], "context": res["context"], "attempts": state["attempts"] + 1}

async def version_compare_node(state: AgentState):
    log_task_start("VERSION_COMPARE", state["attempts"])
    q = enhance_query_with_feedback(state)
    res = await rag_for_version_comparison(q, state["file_context"], state["session_id"])
    return {"answer": res["answer"], "context": res["context"], "attempts": state["attempts"] + 1}

async def cross_check_node(state: AgentState):
    log_task_start("CROSS_CHECK", state["attempts"])
    q = enhance_query_with_feedback(state)
    res = await rag_for_cross_check(q, state["session_id"], state["file_context"])
    return {"answer": res["answer"], "context": res["context"], "attempts": state["attempts"] + 1}

async def db_design_node(state: AgentState):
    log_task_start("DB_DESIGN", state["attempts"])
    q = enhance_query_with_feedback(state)
    res = await rag_for_db_design(q, state["session_id"])
    return {"answer": res["answer"], "context": res["context"], "attempts": state["attempts"] + 1}

async def code_analysis_node(state: AgentState):
    log_task_start("CODE_ANALYSIS", state["attempts"])
    q = enhance_query_with_feedback(state)
    res = await analyze_code_context(q, state["file_context"], state["session_id"])
    return {"answer": res["answer"], "context": res["context"], "attempts": state["attempts"] + 1}

async def db_schema_node(state: AgentState):
    log_task_start("DB_SCHEMA", state["attempts"])
    q = enhance_query_with_feedback(state)
    res = await rag_for_db_schema(q, state["session_id"])
    return {"answer": res["answer"], "context": res["context"], "attempts": state["attempts"] + 1}

async def rule_doc_node(state: AgentState):
    log_task_start("RULE_DOC", state["attempts"])
    q = enhance_query_with_feedback(state)
    res = await rag_for_rules(q, state["session_id"])
    return {"answer": res["answer"], "context": res["context"], "attempts": state["attempts"] + 1}

async def general_node(state: AgentState):
    log_task_start("GENERAL", state["attempts"])
    res = await ask_llm_general(state["question"], state["session_id"])
    return {"answer": res["answer"], "context": res["context"], "attempts": state["attempts"] + 1}

# ⚡ [Async] Validator Node
async def validator_node(state: AgentState):
    current_answer = state["answer"]
    intent = state["intent"]
    
    if intent == "GENERAL" or len(current_answer) < 10:
        return {"feedback": "PASS"}

    # Validator에게 Context 제공
    val_prompt = f"""
    [질문]: {state['question']}
    [검색된 근거 문서/Context]:
    {state['context']} 

    [AI 답변]: 
    {current_answer}
    """
    
    try:
        # 검증도 비동기 실행
        result = await ainvoke_chain_with_history(VALIDATOR_SYSTEM_PROMPT, "Evaluate", val_prompt, "validator_session")
        
        if "FAIL" in result:
            reason = result.split("REASON:")[-1].strip() if "REASON:" in result else "Low Quality"
            logger.warning(f"⚠️ [Validator] REJECTED: {reason}")
            return {"feedback": reason}
        else:
            return {"feedback": "PASS"}
            
    except Exception as e:
        logger.error(f"Validator Error: {e}")
        return {"feedback": "PASS"}

# 🔄 Conditional Logic (Retry to Router)
def should_retry_or_end(state: AgentState) -> Literal["retry", "end"]:
    feedback = state.get("feedback", "PASS")
    attempts = state["attempts"]
    MAX_RETRIES = 2

    if feedback == "PASS":
        logger.info("🏁 [Edge] 검증 통과 -> 종료")
        return "end"
    
    if attempts > MAX_RETRIES:
        logger.info(f"🛑 [Edge] 최대 재시도({MAX_RETRIES}) 초과 -> 종료")
        return "end"
    
    # 🔄 Priority 3: 실패 시 Router로 보내서 의도 재분석 기회 부여
    logger.info(f"🔙 [Edge] 재시도 필요 (Feedback: {feedback}) -> Router로 회귀")
    return "retry"

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

    # 🔄 Validator -> Router (Retry) or End
    workflow.add_conditional_edges(
        "validator",
        should_retry_or_end,
        {
            "end": END,
            "retry": "router"  # 실패 시 Router로 돌아감
        }
    )

    return workflow.compile()

rag_graph = build_rag_graph()


# ==========================================================================
# 7. Main Execution Interface (Async)
# ==========================================================================
async def execute_rag_task(query: str, session_id: str, file_context: str = "", has_file: bool = False) -> Dict[str, Any]:
    try:
        logger.info(f"🚀 [Async RAG] New Request (Session: {session_id})")

        initial_state = {
            "question": query,
            "session_id": session_id,
            "file_context": file_context if file_context else "",
            "has_file": has_file,
            "intent": "GENERAL",
            "answer": "",
            "attempts": 0,
            "feedback": "",
            "context": "" # 👈 [필수] 초기 상태에 context 키 추가
        }

        # 비동기 그래프 실행 (.ainvoke)
        result = await rag_graph.ainvoke(initial_state)
        
        return {
            "intent": result.get("intent", "GENERAL"),
            "answer": result.get("answer", "No Answer")
        }

    except Exception as e:
        logger.exception("LangGraph Execution Failed")
        return {"intent": "ERROR", "answer": f"시스템 오류 발생: {e}"}