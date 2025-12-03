import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # --- [Oracle 설정] ---
    ORA_USER: str
    ORA_PASSWORD: str
    ORA_DSN: str
    ORA_LIB_DIR: str

    # --- [벡터 스토어 및 파일 경로 설정] ---
    DB_SCHEMA_VECTORSTORE_PATH: str       # Oracle 스키마 벡터스토어
    DOC_VECTORSTORE_PATH: str             # 전력거래 규정 벡터스토어
    
    # --- [데이터 파일 경로] ---
    PDF_FILE_PATH: str
    TXT_FILE_PATH: str
    DB_PATH: str                # SQLite DB 경로

    # --- [모델 설정] ---
    EMBEDDING_MODEL_PATH: str
    OLLAMA_BASE_URL: str
    OLLAMA_MODEL: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore" # 정의되지 않은 변수가 있어도 에러 안 나게 처리

try:
    Config = Settings()
except Exception as e:
    print(f"⚠️ [설정 주의] .env 파일 로드 중 오류 또는 기본값 사용: {e}")
    # 필요시 여기서 raise를 하지 않고 기본값으로 진행하거나, 로그만 남김
    Config = Settings()