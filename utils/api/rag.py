# utils/api/rag.py

from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List, Optional
import logging
import io
import zipfile

# ---------------------------------------------------- 
# 👇 RAG 관련 함수
# ----------------------------------------------------
# classify_intent 필요 없음. execute_rag_task만 임포트
from utils.ollama_rag import execute_rag_task

# ----------------------------------------------------
# 👇 다양한 파일 파싱을 위한 라이브러리 임포트
# ----------------------------------------------------
try:
    import pandas as pd
except ImportError:
    pd = None
    logging.warning("pandas가 설치되지 않았습니다. 엑셀 파일 처리가 비활성화됩니다.")

try:
    import openpyxl
except ImportError:
    openpyxl = None
    logging.warning("openpyxl이 설치되지 않았습니다. .xlsx 파일 처리가 비활성화됩니다.")

try:
    import xlrd
except ImportError:
    xlrd = None
    logging.warning("xlrd가 설치되지 않았습니다. .xls 파일 처리가 비활성화됩니다.")

try:
    from PyPDF2 import PdfReader
    from PyPDF2.errors import FileNotDecryptedError
except ImportError:
    PdfReader = None
    FileNotDecryptedError = None
    logging.warning("PyPDF2가 설치되지 않았습니다. .pdf 파일 처리가 비활성화됩니다.")

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
    logging.warning("BeautifulSoup4가 설치되지 않았습니다. .xml, .jsp, .html 파일 처리가 비활성화됩니다.")
# ----------------------------------------------------


router = APIRouter()
logger = logging.getLogger(__name__)
# ----------------------------------------------------


# ----------------------------------------------------
# 👇 파일 확장자별 텍스트 추출 헬퍼 함수
# ----------------------------------------------------
async def read_file_content(f: UploadFile) -> str:
    """
    업로드된 파일(UploadFile)을 받아, 확장자에 맞는 파서를 사용해 텍스트를 추출합니다.
    """
    filename = f.filename.lower()
    content_bytes = await f.read()

    try:
        # 1️⃣ XLSX (엑셀)
        if filename.endswith('.xlsx'):
            if not pd or not openpyxl:
                raise ImportError("pandas/openpyxl이 설치되지 않아 .xlsx 파일을 읽을 수 없습니다.")
            try:
                with pd.ExcelFile(io.BytesIO(content_bytes), engine='openpyxl') as xls:
                    sheets = []
                    for sheet_name in xls.sheet_names:
                        df = pd.read_excel(xls, sheet_name=sheet_name)
                        sheet_text = f"--- 시트: {sheet_name} ---\n{df.to_string(index=False)}"
                        sheets.append(sheet_text)
                    return "\n\n".join(sheets)
            except zipfile.BadZipFile:
                raise HTTPException(status_code=400, detail=f"'{f.filename}'은 손상되었거나 암호화된 .xlsx 파일입니다.")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f".xlsx 파일 처리 중 오류: {e}")

        # 2️⃣ XLS
        elif filename.endswith('.xls'):
            if not pd or not xlrd:
                raise ImportError("pandas/xlrd가 설치되지 않아 .xls 파일을 읽을 수 없습니다.")
            try:
                with pd.ExcelFile(io.BytesIO(content_bytes), engine='xlrd') as xls:
                    sheets = []
                    for sheet_name in xls.sheet_names:
                        df = pd.read_excel(xls, sheet_name=sheet_name)
                        sheets.append(f"--- 시트: {sheet_name} ---\n{df.to_string(index=False)}")
                    return "\n\n".join(sheets)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f".xls 파일 처리 중 오류: {e}")

        # 3️⃣ HTML/XML/JSP
        elif filename.endswith(('.xml', '.jsp', '.html')):
            if not BeautifulSoup:
                raise ImportError("BeautifulSoup4가 설치되지 않아 .xml/.jsp/.html 파일을 읽을 수 없습니다.")
            try:
                try:
                    text = content_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    text = content_bytes.decode('cp949')
                soup = BeautifulSoup(text, 'lxml')
                return soup.get_text(separator="\n", strip=True)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"HTML/XML 파일 처리 중 오류: {e}")

        # 4️⃣ PDF
        elif filename.endswith('.pdf'):
            if not PdfReader:
                raise ImportError("PyPDF2가 설치되지 않아 .pdf 파일을 읽을 수 없습니다.")
            try:
                reader = PdfReader(io.BytesIO(content_bytes))
                if reader.is_encrypted:
                    raise HTTPException(status_code=400, detail=f"'{f.filename}'은 암호화된 PDF입니다.")
                pdf_texts = [page.extract_text() or "" for page in reader.pages]
                return "\n\n".join(pdf_texts)
            except FileNotDecryptedError:
                raise HTTPException(status_code=400, detail=f"'{f.filename}'은 암호화된 PDF입니다.")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"PDF 처리 중 오류: {e}")

        # 5️⃣ 기본 텍스트 파일
        else:
            try:
                return content_bytes.decode('utf-8')
            except UnicodeDecodeError:
                return content_bytes.decode('cp949')

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception(f"파일 파싱 중 오류 ({filename}): {e}")
        raise HTTPException(status_code=400, detail=f"'{f.filename}' 처리 중 오류 발생: {e}")


# ----------------------------------------------------
# 👇 메인 RAG 엔드포인트
# ----------------------------------------------------
@router.post("/ask")
async def ask_question(
    query: str = Form(...),
    session_id: str = Form(...),
    file: Optional[List[UploadFile]] = File(None)
):
    try:
        combined_context = ""
        has_file = False
        
        # 1. 파일 처리
        if file and len(file) > 0:
            has_file = True
            logger.info(f"📂 [파일 수신] {len(file)}개 처리 중...")
            full_text = []
            for f in file:
                txt = await read_file_content(f)
                full_text.append(txt)
            combined_context = "\n\n".join(full_text)
            
            # (기존 file_snippet 생성 로직 제거됨: Router Node에서 직접 처리함)

        # 2. 코드 붙여넣기 감지 (파일 없을 때)
        elif len(query) > 300 or any(k in query[:200] for k in ["import ", "def ", "class ", "function "]):
            # 텍스트로 된 코드가 들어온 경우 파일 컨텍스트로 취급
            combined_context = query
            # has_file은 False로 두어 Router가 CODE_ANALYSIS로 판단하게 유도하거나 
            # 필요하다면 True로 설정 가능. 여기서는 False 유지 (Router가 텍스트만 보고 판단)

        # 3. RAG 실행 엔진 호출 (LangGraph 실행)
        # execute_rag_task가 내부에서 Intent 분류와 처리를 모두 수행하고 결과를 반환함
        result = execute_rag_task(
            query=query,
            session_id=session_id,
            file_context=combined_context,
            has_file=has_file
        )

        # 4. 결과 응답
        intent = result.get("intent", "UNKNOWN")
        answer = result.get("answer", "")
        
        logger.info(f"🤖 [Router Result] Intent: {intent} (Session: {session_id})")

        return JSONResponse(status_code=200, content={"intent": intent, "answer": answer})

    except Exception as e:
        logger.exception("API Error")
        raise HTTPException(status_code=500, detail=str(e))