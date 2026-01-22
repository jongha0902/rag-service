import logging
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from utils.api.rag import router as rag_router
from utils.ollama_rag import initialize_all_vectorstores, cleanup_expired_sessions

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ✅ 앱 시작 시 통합 초기화 수행 (DB스키마 & 규정문서 로드)
    logger.info("🚀 [통합 서비스] 앱 시작: 벡터 스토어 및 리소스 초기화...")

    initialize_all_vectorstores()

    # asyncio.create_task를 사용하여 메인 스레드를 차단하지 않고 실행
    cleanup_task = asyncio.create_task(cleanup_expired_sessions())

    yield
    logger.info("🛑 앱 종료")

app = FastAPI(lifespan=lifespan, title="Unified Smart RAG Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rag_router)