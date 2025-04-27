"""
人脸数据仓储类
"""
from typing import Optional, List
from sqlalchemy.orm import Session

from .base_repository import BaseRepository
from ..models.face import Face

class FaceRepository(BaseRepository[Face]):
    """
    人脸数据仓储操作
    """
    def __init__(self):
        super().__init__(Face)
    
    def get_by_userid(self, db: Session, userid: int) -> List[Face]:
        """
        根据用户ID获取人脸数据列表
        
        Args:
            db: 数据库会话
            userid: 用户ID
            
        Returns:
            List[Face]: 人脸数据列表
        """
        return db.query(self.model).filter(self.model.userid == userid).all()
    
    def get_latest_face(self, db: Session, userid: int) -> Optional[Face]:
        """
        获取用户最新的人脸数据
        
        Args:
            db: 数据库会话
            userid: 用户ID
            
        Returns:
            Optional[Face]: 最新的人脸数据，如果不存在则返回None
        """
        return db.query(self.model)\
            .filter(self.model.userid == userid)\
            .order_by(self.model.createdAt.desc())\
            .first() 