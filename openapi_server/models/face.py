"""
人脸数据模型
"""
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
import time

from ..database import Base

class Face(Base):
    """人脸数据表"""
    __tablename__ = "faces"

    # 人脸数据ID
    faceid = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # 用户ID
    userid = Column(Integer, index=True, nullable=True)
    
    # 人脸图像的Base64编码
    faceImageBase64 = Column(String, nullable=False)
    
    # 创建时间
    createdAt = Column(Integer, default=lambda: int(time.time()))
    
    # 更新时间
    updatedAt = Column(Integer, default=lambda: int(time.time()), onupdate=lambda: int(time.time())) 