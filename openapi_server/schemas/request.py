"""
请求数据模型
"""
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict, validator
from fastapi import HTTPException, status

class CreateOrUpdateFaceRequest(BaseModel):
    """创建或更新人脸请求"""
    userId: Optional[int] = Field(None, description="用户ID")
    faceImageBase64: str = Field(..., description="Base64编码的人脸图片")

    model_config = ConfigDict(
        json_schema_extra={
            "properties": {
                "userId": {
                    "description": "用户ID"
                }
            }
        }
    )

    @validator('userId')
    def validate_user_id(cls, v):
        if v is None:
            raise ValueError("用户ID不能为空")
        if not isinstance(v, int):
            raise ValueError("用户ID必须是整数")
        if v <= 0:
            raise ValueError("用户ID必须大于0")
        return v

class VerifyFaceRequest(BaseModel):
    """验证人脸请求"""
    userId: Optional[int] = Field(None, description="用户ID")
    faceImagesBase64: List[str] = Field(
        default_factory=list, 
        description="Base64编码的人脸图片列表，最多10张"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "properties": {
                "userId": {
                    "description": "用户ID"
                }
            }
        }
    )

    @validator('userId')
    def validate_user_id(cls, v):
        if v is None:
            raise ValueError("用户ID不能为空")
        if not isinstance(v, int):
            raise ValueError("用户ID必须是整数")
        if v <= 0:
            raise ValueError("用户ID必须大于0")
        return v

    @validator('faceImagesBase64')
    def validate_face_images(cls, v):
        if not v:
            raise ValueError("图片列表不能为空")
        if len(v) > 10:
            raise ValueError("图片数量超过限制，最多支持10张图片")
        return v 