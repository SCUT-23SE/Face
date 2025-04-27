"""
请求日志中间件
"""
import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

class RequestLoggerMiddleware(BaseHTTPMiddleware):
    """
    请求日志中间件
    记录每个请求的处理时间和基本信息
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 记录请求开始时间
        start_time = time.time()
        
        # 获取请求信息
        path = request.url.path
        method = request.method
        
        try:
            # 处理请求
            response = await call_next(request)
            
            # 计算处理时间
            process_time = (time.time() - start_time) * 1000
            
            # 记录成功请求
            print(f"[{method}] {path} - {response.status_code} - {process_time:.2f}ms")
            
            return response
            
        except Exception as e:
            # 计算处理时间
            process_time = (time.time() - start_time) * 1000
            
            # 记录失败请求
            print(f"[{method}] {path} - 500 - {process_time:.2f}ms - Error: {str(e)}")
            
            # 重新抛出异常，让错误处理中间件处理
            raise 