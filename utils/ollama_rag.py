import os
import logging
import torch
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import TypedDict, Dict, Any, Literal, Optional

# LangChain Imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from PyPDF2 import PdfReader

# LangGraph Imports
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
# 0. Prompts (XML Tagging for Safety)
# ==========================================================================
# [Issue 5] Prompt Injection 방지를 위한 XML 태그 분리 적용
SQL_SYSTEM_PROMPT = """
You are an expert Oracle SQL architect.
You MUST generate SQL through the following strict 7-step reasoning process based on the provided Context.

[Context Instructions]
- Use information inside <RULES>...</RULES> for business logic.
- Use information inside <DB_SCHEMA>...</DB_SCHEMA> for table/column mapping.

### STEP 1: Rule Analysis
### STEP 2: Data Element Extraction
### STEP 3: DB Schema Mapping (Identify tables/columns)
### STEP 4: JOIN Plan
### STEP 5: SQL Draft
### STEP 6: Validation
### STEP 7: Final SQL Output (Code block only)

You MUST follow all 7 steps exactly.
"""

# [Issue 3] JSON Output Enforcement
ROUTER_SYSTEM_PROMPT = """
You are an AI Intent Classifier.
Analyze the user's query and the provided file snippet (if any).

Classify the intent into EXACTLY ONE of the following categories:
- "FILE_ONLY": User asks about the uploaded file.
- "VERSION_COMPARE": User compares uploaded file vs existing rules.
- "CROSS_CHECK": Compare 'Market Rules' vs 'DB Schema'.
- "DB_DESIGN": Design schema, Create table, DDL.
- "CODE_ANALYSIS": Raw code/text provided.
- "DB_SCHEMA": General DB questions, SQL generation.
- "RULE_DOC": General Rule questions.
- "GENERAL": General chat.

You MUST respond with a valid JSON object:
{ "intent": "CATEGORY_NAME" }
"""

# [Issue 1] Stateless Validator Prompt
VALIDATOR_SYSTEM_PROMPT = """
You are an AI Quality Assurance Agent.
Evaluate the AI's response based on the User's Query and Intent.

Criteria:
1. Does it directly answer the question?
2. Is it free from hallucinations or 'I don't know' loops?
3. If SQL was requested, is SQL code present?

Output JSON ONLY:
{
    "status": "PASS" or "FAIL",
    "reason": "Short feedback if FAIL, otherwise empty"
}
"""


# ==========================================================================
# 1. 전역 변수 & 캐싱
# ==========================================================================
embeddings = None
db_schema_vectorstore = None
doc_vectorstore = None

# [Issue 8] History Limit Increase
MAX_HISTORY_LENGTH = 50 

# 세션 저장소
store = {}
SESSION_TIMEOUT_MINUTES = 60

# [Issue 10] File VectorStore Cache (In-Memory)
# session_id -> { "hash": str, "vectorstore": FAISS }
file_vs_cache = {}

# LLM 초기화
llm = ChatOllama(
    model=Config.OLLAMA_MODEL,
    temperature=0.1,
    base_url=Config.OLLAMA_BASE_URL,
    # [Issue 9] 타임아웃 설정 (필요시 조정)
    request_timeout=120.0 
)


# ==========================================================================
# 2. 세션 및 유틸리티 (Async Support)
# ==========================================================================
def get_session_history(session_id: str):
    now = datetime.now()
    if session_id not in store:
        store[session_id] = { "history": ChatMessageHistory(), "last_access": now }
    store[session_id]["last_access"] = now

    history = store[session_id]["history"]
    
    # [Issue 8] 히스토리 트리밍 로직 개선
    if len(history.messages) > MAX_HISTORY_LENGTH:
        # 오래된 메시지 절삭 (시스템 메시지 보존 로직 추가 가능하나 여기선 단순 절삭)
        history.messages = history.messages[-MAX_HISTORY_LENGTH:]

    return history

async def cleanup_expired_sessions():
    while True:
        try:
            await asyncio.sleep(600)
            now = datetime.now()
            expired = []
            for sid, data in store.items():
                if now - data["last_access"] > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
                    expired.append(sid)
            
            for sid in expired:
                del store[sid]
                # 파일 캐시도 함께 정리
                if sid in file_vs_cache:
                    del file_vs_cache[sid]
                    
            if expired:
                logger.info(f"🧹 [Cleanup] 만료된 세션 {len(expired)}개 정리 완료")
        except Exception as e:
            logger.error(f"세션 청소 오류: {e}")

# [Issue 9] Async Wrapper for Chain
async def ainvoke_chain_with_history(system_prompt: str, user_question: str, context: str, session_id: str):
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("system", "{context}"), # Context is now pre-formatted with XML tags
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
        chain = prompt | llm | StrOutputParser()
        chain_with_hist = RunnableWithMessageHistory(
            chain, get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history"
        )
        # Use ainvoke for non-blocking
        return await chain_with_hist.ainvoke(
            {"question": user_question, "context": context},
            config={"configurable": {"session_id": session_id}}
        )
    except Exception as e:
        logger.error(f"LLM Chain Error: {e}")
        return f"죄송합니다. 답변 생성 중 오류가 발생했습니다. ({str(e)})"


# ==========================================================================
# 3. 벡터스토어 및 파일 처리
# ==========================================================================
def initialize_all_vectorstores():
    global embeddings, db_schema_vectorstore, doc_vectorstore
    logger.info("🚀 [Init] 벡터 스토어 초기화 시작…")

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

    # Initialize DB Schema Store
    if not os.path.exists(Config.DB_SCHEMA_VECTORSTORE_PATH):
        os.makedirs(Config.DB_SCHEMA_VECTORSTORE_PATH, exist_ok=True)

    idx_path = os.path.join(Config.DB_SCHEMA_VECTORSTORE_PATH, "index.faiss")
    if os.path.exists(idx_path):
        try:
            db_schema_vectorstore = FAISS.load_local(Config.DB_SCHEMA_VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
            logger.info("✅ DB Schema VS Loaded")
        except: pass
    else:
        docs = get_full_db_schema()
        if docs:
            # Batch embedding could be added here
            lc_docs = [Document(page_content=d["content"], metadata={"name": d["name"]}) for d in docs]
            splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            db_schema_vectorstore = FAISS.from_documents(splitter.split_documents(lc_docs), embeddings)
            db_schema_vectorstore.save_local(Config.DB_SCHEMA_VECTORSTORE_PATH)

    # Initialize Doc Store
    if not os.path.exists(Config.DOC_VECTORSTORE_PATH):
        os.makedirs(Config.DOC_VECTORSTORE_PATH, exist_ok=True)
    
    doc_idx_path = os.path.join(Config.DOC_VECTORSTORE_PATH, "index.faiss")
    if os.path.exists(doc_idx_path):
        try:
            doc_vectorstore = FAISS.load_local(Config.DOC_VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
            logger.info("✅ Rule Doc VS Loaded")
        except: pass
    else:
        if os.path.exists(Config.PDF_FILE_PATH):
            # ...PDF loading logic...
            pass


# [Issue 10] Optimized File VectorStore Creation (Caching)
async def get_or_create_file_vectorstore(file_context: str, session_id: str):
    if not file_context or len(file_context) < 10:
        return None
        
    # Generate content hash
    content_hash = hashlib.md5(file_context.encode('utf-8')).hexdigest()
    
    # Check Cache
    if session_id in file_vs_cache:
        cached = file_vs_cache[session_id]
        if cached["hash"] == content_hash:
            logger.info("⚡ [Cache] Existing File VectorStore hit")
            return cached["vectorstore"]
    
    # Create New
    logger.info("🔨 [File] Creating new VectorStore from uploaded file...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([file_context])
    
    # Run in thread pool to avoid blocking event loop
    loop = asyncio.get_running_loop()
    vectorstore = await loop.run_in_executor(
        None, 
        lambda: FAISS.from_documents(docs, embeddings)
    )
    
    # Update Cache
    file_vs_cache[session_id] = {"hash": content_hash, "vectorstore": vectorstore}
    return vectorstore


# ==========================================================================
# 4. JSON Parsers & Structures
# ==========================================================================
class IntentOutput(BaseModel):
    intent: str = Field(description="One of the classification categories")

class ValidatorOutput(BaseModel):
    status: str = Field(description="PASS or FAIL")
    reason: str = Field(description="Reason for failure, empty if PASS")

intent_parser = JsonOutputParser(pydantic_object=IntentOutput)
validator_parser = JsonOutputParser(pydantic_object=ValidatorOutput)


# ==========================================================================
# 5. Node Logic (Handlers) with Error Handling & Optimization
# ==========================================================================

# [Issue 6] Hybrid Search Helper
def hybrid_db_search(query: str, k=5) -> str:
    """Keyword search first, then Vector search fallback"""
    context = ""
    # 1. Keyword Metadata Search (Precise)
    keyword = llm.invoke(f"Extract single main table name keyword from: {query}. If none, return FALSE").content.strip()
    if keyword != "FALSE" and len(keyword) > 2:
        meta_result = search_db_metadata(keyword)
        if meta_result:
            context += f"<METADATA_MATCH>\n{meta_result}\n</METADATA_MATCH>\n"
    
    # 2. Vector Search (Semantic)
    if db_schema_vectorstore:
        docs = db_schema_vectorstore.similarity_search(query, k=k)
        vec_ctx = "\n".join([d.page_content for d in docs])
        context += f"<VECTOR_MATCH>\n{vec_ctx}\n</VECTOR_MATCH>"
    
    return context

# --- Node Handlers ---

async def router_node(state: Dict):
    query = state["question"]
    file_snippet = state["file_context"][:500] if state["file_context"] else "No File"
    
    logger.info(f"🚦 [Router] Analyzing: {query[:50]}...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", ROUTER_SYSTEM_PROMPT),
        ("human", "Query: {query}\nFile Snippet: {snippet}")
    ])
    
    try:
        # [Issue 3] JSON Parsing & [Issue 9] Async
        chain = prompt | llm | intent_parser
        result = await chain.ainvoke({"query": query, "snippet": file_snippet})
        intent = result.get("intent", "GENERAL")
    except Exception as e:
        logger.error(f"Router Error: {e}")
        intent = "GENERAL" # Fallback
        
    logger.info(f"🔀 [Router] Decision: {intent}")
    
    # [Issue 2] Reset feedback loop on new routing decision
    return {"intent": intent, "attempts": 0, "feedback": ""}


async def db_schema_node(state: Dict):
    # [Issue 4] Try-Except Block
    try:
        q = enhance_query_with_feedback(state)
        
        # [Issue 6] Optimized Search
        db_ctx = hybrid_db_search(q)
        rule_ctx = ""
        if doc_vectorstore:
            docs = doc_vectorstore.similarity_search(q, k=3)
            rule_ctx = "\n".join([d.page_content for d in docs])
            
        # [Issue 5] Context Separation
        full_ctx = f"<RULES>\n{rule_ctx}\n</RULES>\n<DB_SCHEMA>\n{db_ctx}\n</DB_SCHEMA>"
        
        ans = await ainvoke_chain_with_history(SQL_SYSTEM_PROMPT, q, full_ctx, state["session_id"])
        return {"answer": ans, "attempts": state["attempts"] + 1}
    except Exception as e:
        return {"answer": f"DB 작업 중 오류가 발생했습니다: {e}", "attempts": state["attempts"] + 1}


async def file_only_node(state: Dict):
    try:
        q = enhance_query_with_feedback(state)
        
        # [Issue 10] Use Cached VectorStore
        vs = await get_or_create_file_vectorstore(state["file_context"], state["session_id"])
        if vs:
            docs = vs.similarity_search(q, k=5)
            ctx = "\n".join([d.page_content for d in docs])
        else:
            ctx = state["file_context"][:2000] # Fallback to raw text if too short

        ans = await ainvoke_chain_with_history(
            "You are a file analysis assistant. Answer based on the provided context.", 
            q, f"<FILE_CONTENT>\n{ctx}\n</FILE_CONTENT>", state["session_id"]
        )
        return {"answer": ans, "attempts": state["attempts"] + 1}
    except Exception as e:
        return {"answer": f"파일 분석 중 오류: {e}", "attempts": state["attempts"] + 1}


async def general_node(state: Dict):
    try:
        ans = await ainvoke_chain_with_history("You are a helpful assistant.", state["question"], "", state["session_id"])
        return {"answer": ans, "attempts": state["attempts"] + 1}
    except Exception as e:
        return {"answer": f"오류: {e}", "attempts": state["attempts"] + 1}


# Placeholder nodes for brevity (apply similar async/try-except patterns)
async def version_compare_node(state): return await file_only_node(state)
async def cross_check_node(state): return await db_schema_node(state) # Reuse logic for now
async def db_design_node(state): return await db_schema_node(state)
async def code_analysis_node(state): return await file_only_node(state)
async def rule_doc_node(state): return await db_schema_node(state)


# ==========================================================================
# 6. Validator Node (Stateless & History Safe)
# ==========================================================================
async def validator_node(state: Dict):
    # [Issue 1] Use LLM directly, NO HISTORY
    current_answer = state["answer"]
    intent = state["intent"]
    
    # Fast pass for simple cases
    if intent == "GENERAL" or "오류" in current_answer:
        return {"feedback": "PASS"}

    prompt = ChatPromptTemplate.from_messages([
        ("system", VALIDATOR_SYSTEM_PROMPT),
        ("human", "Query: {query}\nIntent: {intent}\nAnswer: {answer}")
    ])
    
    try:
        chain = prompt | llm | validator_parser
        result = await chain.ainvoke({
            "query": state["question"],
            "intent": intent,
            "answer": current_answer
        })
        
        status = result.get("status", "PASS")
        reason = result.get("reason", "")
        
        if status == "FAIL":
            logger.warning(f"⚠️ [Validator] REJECTED: {reason}")
            return {"feedback": reason}
        else:
            logger.info("✅ [Validator] PASSED")
            return {"feedback": "PASS"}
            
    except Exception as e:
        logger.error(f"Validator Failed: {e}")
        return {"feedback": "PASS"} # Fail open


# ==========================================================================
# 7. Graph Logic & Building
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

def enhance_query_with_feedback(state: AgentState) -> str:
    query = state["question"]
    if state["attempts"] > 0 and state.get("feedback"):
        logger.info(f"🔄 [Loop] Feedback Injection: {state['feedback']}")
        return f"{query}\n\n[Previous Feedback]: {state['feedback']}\nPlease fix the answer based on this."
    return query

def should_retry(state: AgentState) -> Literal["router", "end"]:
    # [Issue 2] Loop back to ROUTER, not the task node
    # This allows the intent to change if the feedback implies a misunderstanding
    if state.get("feedback") == "PASS":
        return "end"
    
    if state["attempts"] >= 2:
        logger.info("🛑 Max retries reached.")
        return "end"
    
    logger.info("🔙 Retrying -> Sending back to Router")
    return "router" # Changed destination to Router

def build_rag_graph():
    workflow = StateGraph(AgentState)

    # Add Nodes (Async functions)
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

    workflow.set_entry_point("router")

    # Intent Mapping
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

    for node in intent_map.values():
        workflow.add_edge(node, "validator")

    # [Issue 2] & Visual Map
    workflow.add_conditional_edges(
        "validator", 
        should_retry, 
        {"end": END, "router": "router"}
    )

    return workflow.compile()

rag_graph = build_rag_graph()


# ==========================================================================
# 8. Main Execution Interface (Async)
# ==========================================================================
async def execute_rag_task(query: str, session_id: str, file_context: str = "", has_file: bool = False) -> Dict[str, Any]:
    try:
        logger.info(f"🚀 [Start] Session: {session_id}")
        
        initial_state = {
            "question": query,
            "session_id": session_id,
            "file_context": file_context,
            "has_file": has_file,
            "intent": "GENERAL",
            "answer": "",
            "attempts": 0,
            "feedback": ""
        }

        # [Issue 9] Use ainvoke for non-blocking execution
        result = await rag_graph.ainvoke(initial_state)
        
        return {
            "intent": result.get("intent", "GENERAL"),
            "answer": result.get("answer", "No Answer Generated")
        }

    except Exception as e:
        logger.critical(f"🔥 Graph Crash: {e}")
        return {"intent": "ERROR", "answer": f"Critical System Error: {e}"}