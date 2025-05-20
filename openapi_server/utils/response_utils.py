"""
响应工具类
"""
from typing import Dict, Any, Tuple, Optional

def get_status_code(error_type: str) -> int:
    """
    获取错误类型对应的HTTP状态码
    
    Args:
        error_type: 错误类型
        
    Returns:
        int: HTTP状态码
    """
    status_map = {
        "unauthorized": 401,
        "not_found": 404,
        "bad_request": 400,
        "internal_error": 500
    }
    return status_map.get(error_type, 500)

def error_response(message: str, error_type: str = "bad_request") -> Tuple[Dict[str, Any], int]:
    """
    生成错误响应
    
    Args:
        message: 错误信息
        error_type: 错误类型
        
    Returns:
        Tuple[Dict[str, Any], int]: 错误响应和状态码
    """
    return {
        "code": "1",
        "message": message
    }, get_status_code(error_type)

def success_response(data: Any) -> Dict[str, Any]:
    """
    生成成功响应
    
    Args:
        data: 响应数据
        
    Returns:
        Dict[str, Any]: 成功响应
    """
    return {
        "code": "0",
        "data": data
    }

def validate_request(request: Any) -> Optional[Tuple[Dict[str, Any], int]]:
    """
    验证请求参数
    
    Args:
        request: 请求对象
        
    Returns:
        Optional[Tuple[Dict[str, Any], int]]: 如果验证失败返回错误响应和状态码，否则返回None
    """
    if not hasattr(request, 'userId') or not request.userId or request.userId <= 0:
        return error_response("用户ID必须大于0")
    
    if hasattr(request, 'faceImageBase64') and not request.faceImageBase64:
        return error_response("图片数据不能为空")
        
    if hasattr(request, 'faceImagesBase64'):
        if not request.faceImagesBase64:
            return error_response("图片列表不能为空")
        if len(request.faceImagesBase64) > 10:
            return error_response("图片数量超过限制，最多支持10张图片")
    
    return None 