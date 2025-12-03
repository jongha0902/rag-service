import os
import logging
import torch
import asyncio
from datetime import datetime, timedelta
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

from utils.config import Config

# DB 메타데이터 검색 모듈
try:
    from utils.db_full_schema import get_full_db_schema, search_db_metadata, get_all_table_names
except ImportError:
    def get_full_db_schema(): return []
    def search_db_metadata(k): return ""
    def get_all_table_names(): return ""

logger = logging.getLogger(__name__)


# ==========================================================================
# 0. Step-by-Step SQL Generator (완전 통합)
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
# 2. 세션 관리
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


# ==========================================================================
# 3. 공통 LLM 호출
# ==========================================================================
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
# 4. 벡터스토어 초기화
# ==========================================================================
def initialize_all_vectorstores():
    global embeddings, db_schema_vectorstore, doc_vectorstore
    logger.info("🚀 초기화 시작…")

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
            logger.info("DB Schema VectorStore 로드 완료")
        except Exception as e:
            logger.error(f"❌ DB Schema 로드 실패: {e}")
    else:
        docs = get_full_db_schema()
        if docs:
            lc_docs = [Document(page_content=d["content"], metadata={"name": d["name"]}) for d in docs]
            splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            db_schema_vectorstore = FAISS.from_documents(splitter.split_documents(lc_docs), embeddings)
            db_schema_vectorstore.save_local(Config.DB_SCHEMA_VECTORSTORE_PATH)
            logger.info("DB Schema VectorStore 생성 완료")


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
            logger.info("📘 Rule Doc 로드 완료")
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
                logger.info("Rule Doc VectorStore 생성 완료")


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
# 5. Intent Router
# ==========================================================================
def classify_intent(question: str, has_file=False, file_snippet=None) -> str:
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


# ==========================================================================
# 6. SQL Generator Function (핵심)
# ==========================================================================
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
# 7. Handlers
# ==========================================================================

# ------------------------
# DB_DESIGN Handler (강화)
# ------------------------
def rag_for_db_design(question: str, session_id="default"):
    rule_ctx = ""
    if doc_vectorstore:
        rule_docs = doc_vectorstore.similarity_search(question, k=5)
        rule_ctx = "\n".join([d.page_content for d in rule_docs])

    db_ctx = ""
    if db_schema_vectorstore:
        db_docs = db_schema_vectorstore.similarity_search(question, k=5)
        db_ctx = "\n".join([d.page_content for d in db_docs])

    # Step-by-Step SQL 생성
    sql_result = generate_sql_step_by_step(question, rule_ctx, db_ctx, session_id)

    # 테이블 구조 모델링
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


# ------------------------
# FILE_ONLY
# ------------------------
def rag_for_uploaded_files(question, file_context, session_id):
    if len(file_context) < 30000:
        return invoke_chain_with_history("파일 내용을 분석하세요.", question, file_context, session_id)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([file_context])
    temp_vs = FAISS.from_documents(docs, embeddings)
    res = temp_vs.similarity_search(question or "요약", k=8)
    context = "\n".join([d.page_content for d in res])
    return invoke_chain_with_history("파일 발췌본 기반", question, context, session_id)


# ------------------------
# VERSION_COMPARE
# ------------------------
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


# ------------------------
# CROSS_CHECK
# ------------------------
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


# ------------------------
# CODE_ANALYSIS
# ------------------------
def analyze_code_context(question, full_context, session_id):
    return invoke_chain_with_history("코드 분석 전문가", question, full_context, session_id)


# ------------------------
# DB_SCHEMA (SQL 자동 감지 강화)
# ------------------------
def rag_for_db_schema(question, session_id="default"):
    # 🔥 SQL 요청 자동 감지
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

    # 일반 DB 검색 로직
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


# ------------------------
# RULE_DOC
# ------------------------
def rag_for_rules(question, session_id):
    context = ""
    if doc_vectorstore:
        docs = doc_vectorstore.similarity_search(question, k=10)
        context = "\n".join([d.page_content for d in docs])
    return invoke_chain_with_history("규정 전문가", question, context, session_id)


# ------------------------
# GENERAL
# ------------------------
def ask_llm_general(question, session_id):
    return invoke_chain_with_history("도움이 되는 AI", question, "", session_id)


# ==========================================================================
# 8. Executor
# ==========================================================================
def execute_rag_task(intent, query, session_id, file_context=None, has_file=False):
    try:
        if intent == "FILE_ONLY":
            return rag_for_uploaded_files(query, file_context, session_id)

        if intent == "VERSION_COMPARE":
            return rag_for_version_comparison(query, file_context, session_id)

        if intent == "CROSS_CHECK":
            return rag_for_cross_check(query, session_id, file_context if has_file else None)

        if intent == "DB_DESIGN":
            return rag_for_db_design(query, session_id)

        if intent == "CODE_ANALYSIS":
            q = query if has_file else "입력된 코드 분석"
            return analyze_code_context(q, file_context, session_id)

        if intent == "DB_SCHEMA":
            return rag_for_db_schema(query, session_id)

        if intent == "RULE_DOC":
            return rag_for_rules(query, session_id)

        return ask_llm_general(query, session_id)

    except Exception as e:
        logger.error(f"Task Error: {e}")
        return f"오류 발생: {e}"
