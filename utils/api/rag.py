# utils/api/rag.py

from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List, Optional
import logging
import io
import re
import zipfile

# ----------------------------------------------------
# 👇 RAG 관련 함수 (비동기) 
# ----------------------------------------------------
from utils.ollama_rag import execute_rag_task

# ----------------------------------------------------
# 👇 필수 라이브러리 체크 및 로드
# ----------------------------------------------------
logger = logging.getLogger(__name__)

# 데이터 처리용 (Pandas)
try:
    import pandas as pd
except ImportError:
    pd = None
    logger.warning("⚠️ 'pandas'가 설치되지 않았습니다. 엑셀 파일 처리가 제한됩니다.")

try:
    import openpyxl
except ImportError:
    openpyxl = None
    logger.warning("⚠️ 'openpyxl'이 설치되지 않았습니다. .xlsx 파일 처리가 제한됩니다.")

# PDF 처리용 (PyPDF2 유지)
try:
    from PyPDF2 import PdfReader
    from PyPDF2.errors import FileNotDecryptedError
except ImportError:
    PdfReader = None
    FileNotDecryptedError = None
    logger.warning("⚠️ 'PyPDF2'가 설치되지 않았습니다. PDF 파일 처리가 제한됩니다.")

# 인코딩 감지용
try:
    import chardet
except ImportError:
    chardet = None
    logger.warning("⚠️ 'chardet'이 설치되지 않았습니다. 텍스트 인코딩 자동 감지가 비활성화됩니다.")

# HTML/XML 파싱
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
    logger.warning("⚠️ 'beautifulsoup4'가 설치되지 않았습니다. 웹 문서 처리가 제한됩니다.")


router = APIRouter()

# ----------------------------------------------------
# 👇 설정: 파일 제한 및 정규식
# ----------------------------------------------------
MAX_FILE_COUNT = 3
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# 코드 감지 정규식 (import, def, class 등으로 시작하는 패턴)
CODE_PATTERN = re.compile(
    r"^\s*(import\s+\w+|from\s+\w+|class\s+\w+|def\s+\w+|const\s+\w+|let\s+\w+|function\s+\w+)",
    re.MULTILINE
)


# ----------------------------------------------------
# 👇 헬퍼 함수
# ----------------------------------------------------
def detect_encoding(content: bytes) -> str:
    """chardet을 사용해 인코딩 자동 감지, 실패 시 utf-8 기본값"""
    if chardet:
        result = chardet.detect(content)
        encoding = result.get('encoding')
        if encoding:
            return encoding
    return 'utf-8'


async def read_file_content(f: UploadFile) -> str:
    """
    확장자별 최적화된 파서로 텍스트 추출.
    - PDF: PyPDF2
    - Excel: pandas (to_csv)
    - Text: chardet
    """
    filename = f.filename.lower()
    
    # 1. 파일 크기 체크 (read 전)
    content_bytes = await f.read()
    if len(content_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400, 
            detail=f"파일 '{f.filename}'이 너무 큽니다. (최대 {MAX_FILE_SIZE_MB}MB)"
        )

    try:
        # 1️⃣ XLSX / XLS (엑셀)
        if filename.endswith(('.xlsx', '.xls')):
            if not pd:
                raise HTTPException(status_code=503, detail="서버에 pandas가 설치되지 않아 엑셀 파일을 읽을 수 없습니다.")
            
            engine = 'openpyxl' if filename.endswith('.xlsx') else 'xlrd'
            if filename.endswith('.xlsx') and not openpyxl:
                 raise HTTPException(status_code=503, detail="서버에 openpyxl이 설치되지 않았습니다.")
            
            try:
                # 메모리 효율을 위해 BytesIO 사용
                with pd.ExcelFile(io.BytesIO(content_bytes), engine=engine) as xls:
                    sheets = []
                    for sheet_name in xls.sheet_names:
                        df = pd.read_excel(xls, sheet_name=sheet_name)
                        # to_csv 사용 (탭 구분자)
                        sheet_text = f"--- Sheet: {sheet_name} ---\n{df.to_csv(index=False, sep='\t')}"
                        sheets.append(sheet_text)
                    return "\n\n".join(sheets)
            except Exception as e:
                logger.error(f"Excel parsing error: {e}")
                raise HTTPException(status_code=400, detail=f"엑셀 파일 처리 중 오류: {str(e)}")

        # 2️⃣ PDF (PyPDF2)
        elif filename.endswith('.pdf'):
            if not PdfReader:
                raise HTTPException(status_code=503, detail="서버에 PyPDF2가 설치되지 않아 PDF를 읽을 수 없습니다.")
            try:
                reader = PdfReader(io.BytesIO(content_bytes))
                
                # 암호화 체크
                if reader.is_encrypted:
                     raise HTTPException(status_code=400, detail=f"'{f.filename}'은 암호화된 PDF입니다.")

                text_list = []
                for page in reader.pages:
                    txt = page.extract_text() or ""
                    if txt.strip():
                        text_list.append(txt)
                return "\n\n".join(text_list)

            except FileNotDecryptedError:
                 raise HTTPException(status_code=400, detail=f"'{f.filename}'은 암호화된 PDF입니다.")
            except Exception as e:
                logger.error(f"PDF parsing error: {e}")
                raise HTTPException(status_code=400, detail=f"PDF 처리 중 오류: {str(e)}")

        # 3️⃣ HTML/XML/JSP
        elif filename.endswith(('.xml', '.jsp', '.html', '.htm')):
            if not BeautifulSoup:
                raise HTTPException(status_code=503, detail="서버에 BeautifulSoup이 없어 웹 문서를 읽을 수 없습니다.")
            try:
                encoding = detect_encoding(content_bytes)
                text = content_bytes.decode(encoding, errors='replace')
                soup = BeautifulSoup(text, 'html.parser')
                return soup.get_text(separator="\n", strip=True)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"HTML/XML 처리 오류: {str(e)}")

        # 4️⃣ 일반 텍스트 및 코드 파일
        else:
            try:
                encoding = detect_encoding(content_bytes)
                return content_bytes.decode(encoding, errors='replace')
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"텍스트 디코딩 오류: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"파일 파싱 중 알 수 없는 오류 ({filename})")
        raise HTTPException(status_code=400, detail=f"'{f.filename}' 처리 실패: {str(e)}")


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
        if file:
            if len(file) > MAX_FILE_COUNT:
                raise HTTPException(status_code=400, detail=f"파일은 최대 {MAX_FILE_COUNT}개까지만 업로드 가능합니다.")
            
            has_file = True
            logger.info(f"📂 [File Upload] {len(file)} files received. Processing...")
            
            full_text_list = []
            for f in file:
                # 비동기 파일 읽기 (await)
                txt = await read_file_content(f)
                full_text_list.append(f"filename: {f.filename}\n{txt}")
            
            combined_context = "\n\n".join(full_text_list)

        # 2. 코드/텍스트 붙여넣기 감지
        elif len(query) > 300 or CODE_PATTERN.search(query):
            logger.info("💻 [Code/Text Detected] Query treated as context.")
            combined_context = query
            # has_file=False 유지 -> Router가 텍스트만 보고 CODE_ANALYSIS 등 판단

        # 3. RAG 실행 엔진 호출 (비동기 await)
        result = await execute_rag_task(
            query=query,
            session_id=session_id,
            file_context=combined_context,
            has_file=has_file
        )

        # 4. 결과 응답
        intent = result.get("intent", "UNKNOWN")
        answer = result.get("answer", "")
        
        logger.info(f"🤖 [Response] Intent: {intent} (Session: {session_id})")

        return JSONResponse(status_code=200, content={"intent": intent, "answer": answer})

    except HTTPException as he:
        logger.warning(f"⚠️ HTTPException: {he.detail}")
        raise he
    except Exception as e:
        logger.exception("❌ API Internal Error")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")