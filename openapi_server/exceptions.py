"""
自定义异常处理器
"""
from fastapi import Request, status, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from typing import List, Dict, Any

def validation_exception_handler(request: Request, exc: RequestValidationError):
    """处理请求验证错误"""
    errors: List[Dict[str, Any]] = exc.errors()
    if not errors:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "code": 1,
                "message": "无效的请求参数"
            }
        )
    
    # 获取第一个错误
    error = errors[0]
    field_name = error.get("loc", ["未知字段"])[-1]
    error_type = error.get("type", "unknown")
    error_msg = error.get("msg", "无效的请求参数")
    
    # 根据错误类型提供更详细的错误信息
    if error_type == "missing":
        message = f"缺少必要参数: {field_name}"
    elif error_type == "type_error":
        message = f"参数类型错误: {field_name} 必须是 {error.get('ctx', {}).get('expected_type', '未知类型')}"
    elif error_type == "value_error":
        message = f"参数值错误: {field_name} {error_msg}"
    elif error_type == "json_invalid":
        message = "JSON格式错误: 请检查请求体格式是否正确"
    else:
        message = f"{field_name}: {error_msg}"
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "code": 1,
            "message": message
        }
    )

def http_exception_handler(request: Request, exc: HTTPException):
    """处理 HTTP 异常"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": 1,
            "message": exc.detail
        }
    )

def general_exception_handler(request: Request, exc: Exception):
    """处理一般异常"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "code": 1,
            "message": str(exc) if str(exc) else "服务器内部错误"
        }
    ) 