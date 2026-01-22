import logging
import json
import sys
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from utils.db import DatabaseError

# 🔧 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ 공통 오류 응답 함수
def build_error_response(status_code: int, message: str, detail: any = None):
    content = {"message": message}
    if detail:
        content["detail"] = detail
    return JSONResponse(
        status_code=status_code,
        content=content,
        media_type="application/json; charset=utf-8"
    )

# ✅ 예외 발생 시 API 로그 기록 함수 (모든 핸들러에서 호출)
# async def log_exception_usage(request: Request, status_code: int, message: str, exc_detail: str = ""):
#     try:
#         login_id = getattr(request.state, "login_id", None)

#         if request.method in ("POST", "PUT", "PATCH"):
#             body_bytes = await request.body()
#             body = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
#             if not login_id:
#                 login_id = body.get("login_id", "unknown")
#             request_data = body
#         else:
#             query_params = dict(request.query_params)
#             if not login_id:
#                 login_id = query_params.get("login_id", "unknown")
#             request_data = query_params

#         # ❗ 간략화된 에러 정보
#         if not exc_detail:
#             exc_type, exc_value, _ = sys.exc_info()
#             error_detail = f"{exc_type.__name__}: {exc_value}"
#         else:
#             error_detail = exc_detail

#         if request.url.path == "/auth/login":
#             login_id = request_data.get("user_id", "unknown")
        
#         if request_data.get("password"):
#             request_data["password"] = password_hash_key(request_data.get("password"))

#         log_api_usage(
#             login_id or "unknown",
#             request.url.path,
#             request.method,
#             request_data,
#             {"error": message, "detail": error_detail},
#             status_code
#         )
#     except Exception as e:
#         logger.warning(f"[log_api_usage 실패] {e}")

# ✅ 예외 핸들러 정의

# 1. FastAPI 내장 HTTPException
async def handle_http_exception(request: Request, exc: HTTPException):
    logger.warning(f"[HTTPException] {exc.detail}")
    #await log_exception_usage(request, exc.status_code, str(exc.detail))
    return build_error_response(exc.status_code, exc.detail)

# 2. RequestValidationError (요청 유효성 오류)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    logger.warning(f"[ValidationError] {exc.errors()}")
    #await log_exception_usage(request, 422, "요청 데이터가 유효하지 않습니다.", f"{type(exc).__name__}: {exc}")
    return build_error_response(422, "요청 데이터가 유효하지 않습니다.", exc.errors())

# 3. TypeError (잘못된 타입 등 런타임 오류)
async def handle_type_error(request: Request, exc: TypeError):
    logger.error(f"[TypeError] {str(exc)}")
    #await log_exception_usage(request, 400, "잘못된 타입 또는 인자가 전달되었습니다.", f"{type(exc).__name__}: {exc}")
    return build_error_response(400, "잘못된 타입 또는 인자가 전달되었습니다.", str(exc))

# 4. 예기치 못한 일반 예외
async def handle_unexpected_exception(request: Request, exc: Exception):
    logger.exception("[Unhandled Exception] 예외 발생")
    #await log_exception_usage(request, 500, "서버 내부 오류가 발생했습니다.", f"{type(exc).__name__}: {exc}")
    return build_error_response(500, "서버 내부 오류가 발생했습니다.", str(exc))

# 5. DatabaseError (DB 관련 사용자 정의 예외)
async def handle_database_error(request: Request, exc: DatabaseError):
    logger.error(f"[DatabaseError] {str(exc)}")
    #await log_exception_usage(request, 500, "데이터베이스 오류가 발생했습니다.", str(exc))
    return build_error_response(500, "데이터베이스 오류가 발생했습니다.", str(exc))
