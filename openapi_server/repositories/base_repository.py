"""
基础仓储类
"""
from typing import Generic, TypeVar, Type, Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import inspect
from ..database import Base

ModelType = TypeVar("ModelType", bound=Base)

class BaseRepository(Generic[ModelType]):
    """
    提供基本的数据库操作
    """
    def __init__(self, model: Type[ModelType]):
        self.model = model
        self.primary_key = inspect(model).primary_key[0].name

    def get(self, db: Session, id: int) -> Optional[ModelType]:
        """
        根据ID获取单个记录
        """
        return db.query(self.model).filter(getattr(self.model, self.primary_key) == id).first()

    def get_multi(self, db: Session, *, skip: int = 0, limit: int = 100) -> List[ModelType]:
        """
        获取多条记录
        """
        return db.query(self.model).offset(skip).limit(limit).all()

    def create(self, db: Session, *, obj_in: dict) -> ModelType:
        """
        创建记录
        """
        db_obj = self.model(**obj_in)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(self, db: Session, *, db_obj: ModelType, obj_in: dict) -> ModelType:
        """
        更新记录
        """
        for field, value in obj_in.items():
            setattr(db_obj, field, value)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def delete(self, db: Session, *, id: int) -> ModelType:
        """
        删除记录
        """
        obj = db.query(self.model).get(id)
        db.delete(obj)
        db.commit()
        return obj 