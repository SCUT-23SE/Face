"""
人脸控制器
"""
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..database import get_db
from ..services.face_service import FaceService
from ..schemas.request import CreateOrUpdateFaceRequest, VerifyFaceRequest
from ..schemas.response import (
    SuccessWithData,
    FaceData,
    VerifyResult,
    NotFound,
    BadRequest,
    InternalServerError,
    Unauthorized
)

# 创建API路由实例
router = APIRouter(prefix="/users/me", tags=["Face"])

@router.get(
    "/face",
    response_model=SuccessWithData,
    responses={
        401: {"model": Unauthorized},
        404: {"model": NotFound},
        500: {"model": InternalServerError}
    }
)
def get_user_face(
    userId: int,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    获取当前用户的人脸数据
    
    Args:
        userId: 用户ID
        db: 数据库会话
        
    Returns:
        Dict[str, Any]: 包含人脸数据的响应
    """
    try:
        face_data = FaceService.get_user_face(db, userId)
        if not face_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="当前用户尚未设置人脸数据"
            )
        
        return {
            "code": "0",
            "data": FaceData(**face_data).model_dump()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取人脸数据时发生异常: {str(e)}"
        )

@router.put(
    "/face",
    response_model=SuccessWithData,
    responses={
        400: {"model": BadRequest},
        401: {"model": Unauthorized},
        500: {"model": InternalServerError}
    }
)
def create_or_update_face(
    request: CreateOrUpdateFaceRequest,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    创建或更新当前用户的人脸数据
    
    Args:
        request: 请求数据，包含user_id和face_image_base64
        db: 数据库会话
        
    Returns:
        Dict[str, Any]: 包含更新后人脸数据的响应
    """
    # 验证用户ID
    if request.userId <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户ID必须大于0"
        )

    # 验证图片数据
    if not request.faceImageBase64:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="图片数据不能为空"
        )

    try:
        face_data = FaceService.create_or_update_face(
            db,
            request.userId,
            request.faceImageBase64
        )
        return {
            "code": "0",
            "data": FaceData(**face_data).model_dump()
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"保存人脸数据时发生异常: {str(e)}"
        )

@router.post(
    "/face/verify",
    response_model=SuccessWithData,
    responses={
        400: {"model": BadRequest},
        401: {"model": Unauthorized},
        404: {"model": NotFound},
        500: {"model": InternalServerError}
    }
)
async def verify_face(
    request: VerifyFaceRequest,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    验证人脸数据
    
    Args:
        request: 请求数据，包含user_id和face_images_base64
        db: 数据库会话
        
    Returns:
        Dict[str, Any]: 包含验证结果的响应
    """
    # 验证图片列表
    if not request.faceImagesBase64:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="图片列表不能为空"
        )
    if len(request.faceImagesBase64) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="图片数量超过限制"
        )

    try:
        # 验证人脸
        result = await FaceService.verify_face(
            db,
            request.userId,
            request.faceImagesBase64
        )
        
        return {
            "code": "0",
            "data": result
        }
    except ValueError as e:
        if "用户未设置人脸数据" in str(e):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e)
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"人脸验证时发生异常: {str(e)}"
        )