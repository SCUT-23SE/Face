import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError, HTTPException
from openapi_server.controllers.face_controller import router
from openapi_server.exceptions import (
    validation_exception_handler,
    http_exception_handler,
    general_exception_handler
)
from openapi_server.middlewares.error_handler import ErrorHandlerMiddleware
from openapi_server.middlewares.request_logger import RequestLoggerMiddleware

app = FastAPI(
    title="Face Recognition API",
    description="人脸识别 API",
    version="1.0.0"
)

# 注册中间件
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(RequestLoggerMiddleware)

# 注册异常处理器
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# 注册路由
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    ) 