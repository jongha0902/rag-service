# utils/ollama_rag.py
import os
import requests
from utils.config import Config
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 경로 설정
VECTORSTORE_PATH = Config.VECTORSTORE_PATH
EMBEDDING_MODEL_PATH = Config.EMBEDDING_MODEL_PATH
PDF_PATH = Config.PDF_PATH
TXT_PATH = Config.TXT_PATH

vectorstore = None  # ✅ 전역 vectorstore 선언

# -----------------------------------------------------------------
# 👇 임베딩 모델을 전역으로 한 번만 로드합니다.
# -----------------------------------------------------------------
try:
    print("🧠 임베딩 모델 로드 중... (multilingual-e5-large)")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
    print("✅ 임베딩 모델 로드 완료.")
except Exception as e:
    print(f"🚫 FATAL: 임베딩 모델 로드 실패: {e}")
    embeddings = None
# -----------------------------------------------------------------


# PDF → 텍스트 변환
def extract_text_from_pdf(path: str) -> str:
    text = ""
    with open(path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                cleaned = page_text.replace("-\n", "").replace("\n", " ").strip()
                text += cleaned + "\n"
    print(f"[PDF] 전체 페이지 수: {len(reader.pages)} / 추출 완료")

    return text

def save_text(text: str, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# Vectorstore 구축/로딩 (타입 1용)
def create_vectorstore():
    global vectorstore
    global embeddings # 👈  전역 임베딩 사용
    
    if embeddings is None:
        print("🚫 ERROR: 임베딩 모델이 로드되지 않아 Vectorstore를 생성할 수 없습니다.")
        return None

    if not os.path.exists(TXT_PATH):
        print("📄 PDF에서 텍스트 추출 중...")
        extracted = extract_text_from_pdf(PDF_PATH)
        save_text(extracted, TXT_PATH)
    with open(TXT_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1600, chunk_overlap=300, separators=["\n\n", "\n", ".", " "]
    )

    documents = splitter.create_documents([raw_text])
    print(f"📑 문서 분할: 전체 {len(documents)}개")
    filtered_docs = [doc for doc in documents if len(doc.page_content.strip()) > 80]
    print(f"✅ 필터 후 문서: {len(filtered_docs)}개")
    
    # 👈 전역 임베딩을 사용하므로 이 라인 삭제
    # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)

    if not os.path.exists(VECTORSTORE_PATH):
        print("💾 FAISS 벡터스토어 생성 중...")
        vectorstore = FAISS.from_documents(filtered_docs, embeddings)
        vectorstore.save_local(VECTORSTORE_PATH)
    else:
        print("📦 FAISS 벡터스토어 로드 중...")
        vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)

    return vectorstore

def build_context_for_question(question: str, k=20, score_threshold=0.4):
    # 👈 vectorstore가 None일 때 방어 코드
    if vectorstore is None:
        return "ERROR: 전력거래시장 Vectorstore가 로드되지 않았습니다."

    results = vectorstore.similarity_search_with_score(question, k=k)

    # score 오름차순 정렬
    results.sort(key=lambda x: x[1])

    # score 필터링
    filtered = [(doc, score) for doc, score in results if score <= score_threshold]

    if not filtered:
        return "관련 문서를 찾을 수 없습니다."

    context_parts = [
        f"[score={score:.4f}]\n{doc.page_content}"
        for doc, score in filtered
    ]

    return "\n\n".join(context_parts)

custom_prompt = PromptTemplate.from_template(
"""
너는 '전력거래시장 규칙' 전문 기반의 RAG QA 전문가다.
아래 [문서 내용]을 참고하여 [질문]에 대해 아래의 답변 형식과 규칙에 따라 답변하라.

[답변 형식 예시]
1. 정의
(질문에 해당하는 공식/수식이 있으면 반드시 그대로 복사해 답변하라. 문서 내에서 일치하는 공식/수식이 없으면 "문서에서 확인할 수 없습니다."라고 답변하라.)

2. 설명
A. (조건/예외명)
(수식 및 설명)
...

[답변 작성 규칙]
- 1번 항목(정의)에는 질문에 해당하는 공식/수식만 반드시 그대로 복사해 답변하라.
- 중간 변수(ex: MP, GP, TLF 등)는 그대로 두고, 내부 수식은 확장하지 말라.
- 공식/수식은 반드시 문서 내 등장한 그 모습 그대로 복사해라.
- 추론, 변형, 해석, 요약, 다른 용어로의 변환 모두 절대 하지 말라.
- 답변 마지막에 **[END]**만 출력하라.

[문서 내용]
{context}

[질문]
{question}
"""
)

custom_prompt2 = PromptTemplate.from_template(
"""
너는 '전력거래시장 규칙' 전문 기반의 RAG QA 전문가다.
아래 [문서 내용]을 바탕으로 [질문]에 대해 답변하라.
공식/수식은 절대 변경하지말고, 문서 내용에 나온 그대로 답변하라.

[문서 내용]
{context}

[질문]
{question}
"""
)


def query_ollama(prompt: str, model: str = "gpt-oss-20b") -> str:
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.4,           # 창의성 최소화, 결정적 답변
            #"top_k": 40,                  # 후보군 좁게
            #"top_p": 0.7,                # 확률 상위 85%만 후보
            #"repeat_penalty": 1.15,       # 반복 억제
            #"presence_penalty": 1.2,      # 중복 억제
            #"frequency_penalty": 1.1,     # 자주 등장 단어 억제
            #"penalize_newline": True,     # 줄바꿈 반복 억제
            "num_predict": 2024,           # 충분한 길이
            "num_ctx": 32768,              # 가능한 한 크게 (모델 한계까지)
            "stop": ["[END]", "<|end_of_text|>"] # 필요시 프롬프트 종료 문자 지정
        }
    }

    try:
        res = requests.post(url, headers=headers, json=data, timeout=180)
        res.raise_for_status()
        answer = res.json().get("response", "").strip()
        return answer if answer else "문서에서 관련된 정보를 찾을 수 없습니다."
    except requests.exceptions.RequestException as e:
        return f"🚫 Ollama 오류: {e}"

def clean_ollama_answer(raw_answer: str):
    stop_tokens = ["[END]", "<|end_of_text|>"]
    min_idx = len(raw_answer)

    for token in stop_tokens:
        idx = raw_answer.find(token)
        if idx != -1:
            min_idx = min(min_idx, idx + len(token))

    return raw_answer[:min_idx].strip()

# --- (기존 타입 1 함수) ---
def rag_with_ollama(question: str, query_type: str):

    if query_type in ("0", "1"):
        context_str = build_context_for_question(question, k=12)
    else:
        context_str = ""

    if str(query_type) == "0":
        prompt = custom_prompt.format(context=context_str, question=question)
    elif str(query_type) == "1":
        prompt = custom_prompt2.format(context=context_str, question=question)
    else:
        prompt = f"[질문]\n{question}"

    

    print(f"\n📝 최종 Prompt (타입 1):\n{prompt}\n")
    answer = query_ollama(prompt)
    print(f"\n🔍 질문: {question}\n💡 답변: {answer}")

    return {
        "rag_context": context_str,
        "answer": clean_ollama_answer(answer)
    }

# -----------------------------------------------------------------
# 👇  타입 2 (파일 RAG)를 위한 프롬프트와 함수
# -----------------------------------------------------------------

# ✅ 3. 파일 RAG를 위한 새 프롬프트 (custom_prompt3)
custom_prompt3 = PromptTemplate.from_template(
"""
너는 주어진 [문서 내용]을 바탕으로 [질문]에 대해 답변하는 QA 전문가다.
[문서 내용]을 벗어나는 내용은 답변하지 말고, 내용을 요약하거나 추론하여 답변하라.

[문서 내용]
{context}

[질문]
{question}
"""
)

# -----------------------------------------------------------------
# 👇 'rag_with_context' (타입 2) 함수 로직 전체 수정
# -----------------------------------------------------------------
def rag_with_context(question: str, context: str):
    """
    업로드된 파일에서 추출한 텍스트(context)를 기반으로
    "인메모리(In-memory) RAG"를 수행합니다. (Context Stuffing 대신)
    """
    global embeddings # 👈 전역 임베딩 사용
    
    if embeddings is None:
        return {
            "rag_context": "ERROR: Embedding 모델이 로드되지 않았습니다.",
            "answer": "Embedding 모델 로드 실패로 답변할 수 없습니다."
        }

    print(f"🧠 'Type 2' (파일 RAG) 시작. 원본 텍스트 {len(context)}자.")
    
    # 1. 텍스트 분할 (Split)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1600, chunk_overlap=300, separators=["\n\n", "\n", ".", " "]
    )
    documents = splitter.create_documents([context])
    print(f"📑 업로드된 파일 {len(documents)}개 청크로 분할됨.")

    if not documents:
        return {
            "rag_context": "N/A",
            "answer": "파일을 분할(Chunking)했으나 유효한 텍스트가 없습니다."
        }

    # 2. 인메모리 Vectorstore 생성 (Embed & Store)
    print("💾 인메모리(In-memory) FAISS 벡터스토어 생성 중...")
    try:
        # 👈 [핵심] 업로드된 문서로 실시간(임시) 벡터스토어 생성
        temp_vectorstore = FAISS.from_documents(documents, embeddings)
        print("📦 인메모리 벡터스토어 생성 완료.")
    except Exception as e:
        print(f"🚫 ERROR: 인메모리 벡터스토어 생성 실패: {e}")
        return {
            "rag_context": f"Error: {e}",
            "answer": f"파일 벡터스토어 생성 중 오류가 발생했습니다: {e}"
        }

    # 3. 질문이 없으면 요약으로 처리
    if not question or question.strip() == "":
        question = "제공된 문서의 내용을 요약해줘."

    # 4. 검색 (Retrieve) - 관련된 청크(조각) 찾기
    print(f"🔍 벡터 검색 수행 (질문: {question})...")
    k_val = 12 # 12개의 관련 조각을 검색
    
    results = temp_vectorstore.similarity_search_with_score(question, k=k_val)
    
    # 5. 컨텍스트 생성 (Build Context)
    results.sort(key=lambda x: x[1]) # Score 기준 정렬 (낮을수록 좋음)
    
    # (임계값 필터링 - 필요시 활성화)
    # score_threshold = 0.5 
    # filtered = [(doc, score) for doc, score in results if score <= score_threshold]
    filtered = results # (일단 Top-K 모두 사용)
    
    if not filtered:
        context_str = "관련 문서를 찾을 수 없습니다."
    else:
        context_parts = [
            f"[score={score:.4f}]\n{doc.page_content}"
            for doc, score in filtered
        ]
        context_str = "\n\n".join(context_parts)
        
    print(f"✅ RAG 컨텍스트 생성 완료 (총 {len(context_str)}자).")

    # 6. LLM에 질문 (Generate)
    prompt = custom_prompt3.format(context=context_str, question=question)

    #print(f"\n📝 최종 Prompt (파일 RAG):\n{prompt}\n")
    
    # Ollama 호출 (기존 함수 재활용)
    answer = query_ollama(prompt)
    
    print(f"\n🔍 질문: {question}\n💡 답변: {answer}")

    return {
        "rag_context": context_str, # (디버깅/참고용으로 컨텍스트 반환)
        "answer": clean_ollama_answer(answer)
    }

# -----------------------------------------------------------------
# 👇 타입 2 (파일 없음)를 위한 LLM 직접 호출 함수
# -----------------------------------------------------------------
def ask_llm_only(question: str):
    """
    RAG 없이 질문(prompt)만 LLM에 직접 전달하여 답변을 받습니다.
    """
    
    # RAG가 없는 간단한 프롬프트 형식
    prompt = f"[질문]\n{question}"
    
    print(f"\n📝 최종 Prompt (LLM Only):\n{prompt}\n")

    # Ollama 호출 (기존 함수 재활용)
    answer = query_ollama(prompt)
    
    print(f"\n🔍 질문: {question}\n💡 답변: {answer}")

    # 프론트엔드가 동일한 {answer, rag_context} 형식을 기대할 수 있으므로
    # 'rag_context'는 비워두고 형식을 맞춥니다.
    return {
        "rag_context": "N/A (LLM 직접 호출)",
        "answer": clean_ollama_answer(answer)
    }