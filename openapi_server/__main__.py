#!/usr/bin/env python3

import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError, HTTPException
from .controllers.face_controller import router
from .exceptions import (
    validation_exception_handler,
    http_exception_handler,
    general_exception_handler
)

app = FastAPI(
    title="Face Recognition API",
    description="人脸识别 API",
    version="1.0.0"
)

# 注册异常处理器
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# 注册路由
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(
        "openapi_server.__main__:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
