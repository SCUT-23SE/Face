"""
请求数据模型
"""
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict
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

    def __init__(self, **data):
        super().__init__(**data)
        if self.userId is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="用户ID不能为空"
            )
        if not isinstance(self.userId, int):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="用户ID必须是整数"
            )
        if self.userId <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="用户ID必须大于0"
            )

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

    def __init__(self, **data):
        super().__init__(**data)
        if self.userId is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="用户ID不能为空"
            )
        if not isinstance(self.userId, int):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="用户ID必须是整数"
            )
        if self.userId <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="用户ID必须大于0"
    ) 