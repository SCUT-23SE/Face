"""
人脸控制器
"""
from typing import Dict, Any, Union
from fastapi import APIRouter, Depends, Response
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
    Unauthorized,
    ErrorResponse
)
from ..utils.response_utils import (
    success_response,
    error_response,
    validate_request,
    get_status_code
)

# 创建API路由实例
router = APIRouter(prefix="/users/me", tags=["Face"])

@router.get(
    "/face",
    response_model=Union[SuccessWithData, ErrorResponse],
    responses={
        401: {"model": Unauthorized},
        404: {"model": NotFound},
        500: {"model": InternalServerError}
    }
)
def get_user_face(
    userId: int,
    response: Response,
    db: Session = Depends(get_db)
) -> Union[SuccessWithData, ErrorResponse]:
    """
    获取当前用户的人脸数据
    
    Args:
        userId: 用户ID
        response: FastAPI响应对象
        db: 数据库会话
        
    Returns:
        Union[SuccessWithData, ErrorResponse]: 响应数据
    """
    try:
        # 验证用户ID
        if userId <= 0:
            response.status_code = 400
            return BadRequest(message="用户ID必须大于0")
            
        face_data = FaceService.get_user_face(db, userId)
        if not face_data:
            response.status_code = 404
            return NotFound(message="当前用户尚未设置人脸数据")
        
        return SuccessWithData(
            code="0",
            data=FaceData(**face_data).model_dump()
        )
        
    except Exception as e:
        response.status_code = 500
        return InternalServerError(message=f"获取人脸数据时发生异常: {str(e)}")

@router.put(
    "/face",
    response_model=Union[SuccessWithData, ErrorResponse],
    responses={
        400: {"model": BadRequest},
        401: {"model": Unauthorized},
        500: {"model": InternalServerError}
    }
)
def create_or_update_face(
    request: CreateOrUpdateFaceRequest,
    response: Response,
    db: Session = Depends(get_db)
) -> Union[SuccessWithData, ErrorResponse]:
    """
    创建或更新当前用户的人脸数据
    
    Args:
        request: 请求数据，包含user_id和face_image_base64
        response: FastAPI响应对象
        db: 数据库会话
        
    Returns:
        Union[SuccessWithData, ErrorResponse]: 响应数据
    """
    try:
        print(f"收到注册请求: userId={request.userId}, faceImageBase64长度={len(request.faceImageBase64) if request.faceImageBase64 else 0}")
        
        # 验证请求参数
        validation_result = validate_request(request)
        if validation_result:
            print(f"请求验证失败: {validation_result}")
            response.status_code = 400
            return BadRequest(message=validation_result[0]["message"])

        face_data = FaceService.create_or_update_face(
            db,
            request.userId,
            request.faceImageBase64
        )
        print(f"注册成功: faceId={face_data['faceId']}")
        return SuccessWithData(
            code="0",
            data=FaceData(**face_data).model_dump()
        )
        
    except ValueError as e:
        print(f"注册失败(ValueError): {str(e)}")
        response.status_code = 400
        return BadRequest(message=str(e))
    except Exception as e:
        print(f"注册失败(Exception): {str(e)}")
        response.status_code = 500
        return InternalServerError(message=f"保存人脸数据时发生异常: {str(e)}")

@router.post(
    "/face/verify",
    response_model=Union[SuccessWithData, ErrorResponse],
    responses={
        400: {"model": BadRequest},
        401: {"model": Unauthorized},
        404: {"model": NotFound},
        500: {"model": InternalServerError}
    }
)
async def verify_face(
    request: VerifyFaceRequest,
    response: Response,
    db: Session = Depends(get_db)
) -> Union[SuccessWithData, ErrorResponse]:
    """
    验证人脸数据
    
    Args:
        request: 请求数据，包含user_id和face_images_base64
        response: FastAPI响应对象
        db: 数据库会话
        
    Returns:
        Union[SuccessWithData, ErrorResponse]: 响应数据
    """
    try:
        # 验证请求参数
        validation_result = validate_request(request)
        if validation_result:
            response.status_code = 400
            return BadRequest(message=validation_result[0]["message"])

        # 验证人脸
        result = await FaceService.verify_face(
            db,
            request.userId,
            request.faceImagesBase64
        )
        
        return SuccessWithData(
            code="0",
            data=VerifyResult(isMatch=result).model_dump()
        )
        
    except ValueError as e:
        if "用户未设置人脸数据" in str(e):
            response.status_code = 404
            return NotFound(message=str(e))
        response.status_code = 400
        return BadRequest(message=str(e))
    except Exception as e:
        response.status_code = 500
        return InternalServerError(message=f"人脸验证时发生异常: {str(e)}")