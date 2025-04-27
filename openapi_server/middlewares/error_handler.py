"""
错误处理中间件
"""
import traceback
from typing import Callable, Any
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..schemas.response import InternalServerError

class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    全局错误处理中间件
    捕获所有未处理的异常，返回500错误响应
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
            
        except Exception as e:
            # 记录详细的错误信息
            error_detail = {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            print("[Error]", error_detail)  # 在实际环境中应该使用proper logging
            
            # 返回500错误响应
            return JSONResponse(
                status_code=500,
                content={
                    "code": 1,
                    "message": str(e) if str(e) else "服务器内部错误"
                }
            ) 