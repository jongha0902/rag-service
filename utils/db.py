import logging
import os
import sqlite3
import oracledb
from contextlib import contextmanager
from utils.config import Config

logger = logging.getLogger(__name__)

class DatabaseError(Exception):
    pass

# ---------------------------------------------------------
# ✅ Oracle DB (Project 1)
# ---------------------------------------------------------
try:
    if Config.ORA_LIB_DIR and os.path.exists(Config.ORA_LIB_DIR):
        oracledb.init_oracle_client(lib_dir=Config.ORA_LIB_DIR)
        logger.info(f"✅ Oracle Client 초기화 완료: {Config.ORA_LIB_DIR}")
    else:
        logger.warning("⚠️ Oracle Client 경로가 없거나 잘못되었습니다. Oracle 기능이 제한될 수 있습니다.")
except Exception as e:
    logger.warning(f"⚠️ Oracle Client 초기화 실패: {e}")

@contextmanager
def get_oracle_conn():
    conn = None
    try:
        conn = oracledb.connect(
            user=Config.ORA_USER,
            password=Config.ORA_PASSWORD,
            dsn=Config.ORA_DSN
        )
        yield conn
        conn.commit()
    except Exception as e:
        if conn: conn.rollback()
        logger.error(f"Oracle Error: {e}")
        raise DatabaseError(f"[Oracle 오류] {e}")
    finally:
        if conn: conn.close()

# ---------------------------------------------------------
# ✅ SQLite DB (Project 2)
# ---------------------------------------------------------
@contextmanager
def get_sqlite_conn():
    # 경로가 없으면 디렉토리 생성
    os.makedirs(os.path.dirname(Config.DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(Config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise DatabaseError(f"[SQLite 오류] {e}")
    finally:
        conn.close()