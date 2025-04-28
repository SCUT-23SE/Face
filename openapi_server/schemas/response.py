"""
响应数据模型
"""
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field

class BaseResponse(BaseModel):
    """基础响应模型"""
    code: str = Field(..., description="响应码，0表示成功，1表示失败")

class SuccessWithData(BaseResponse):
    """带数据的成功响应"""
    data: Dict[str, Any] = Field(..., description="响应数据")

class ErrorResponse(BaseResponse):
    """错误响应"""
    message: str = Field(..., description="错误信息")

class FaceData(BaseModel):
    """人脸数据"""
    faceId: int = Field(..., description="人脸数据ID")
    userId: int = Field(..., description="用户ID")
    faceImageBase64: str = Field(..., description="人脸图像的Base64编码")
    createdAt: int = Field(..., description="创建时间（Unix时间戳，单位：秒）")
    updatedAt: int = Field(..., description="最后更新时间（Unix时间戳，单位：秒）")

class VerifyResult(BaseModel):
    """验证结果"""
    isMatch: bool = Field(..., description="人脸是否匹配")

# 预定义的错误响应模型
class BadRequest(ErrorResponse):
    """请求参数错误"""
    code: Literal["1"] = "1"
    message: str = Field(..., description="请求参数错误信息")

class Unauthorized(ErrorResponse):
    """未授权错误"""
    code: Literal["1"] = "1"
    message: str = Field(..., description="未授权错误信息")

class NotFound(ErrorResponse):
    """资源未找到错误"""
    code: Literal["1"] = "1"
    message: str = Field(..., description="资源未找到错误信息")

class InternalServerError(ErrorResponse):
    """服务器内部错误"""
    code: Literal["1"] = "1"
    message: str = Field(..., description="服务器内部错误信息") 