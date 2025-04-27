"""
数据库连接和会话管理
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session

from .config import Config

# 创建SQLAlchemy引擎
engine = create_engine(
    Config.DATABASE_URL,
    connect_args={"check_same_thread": False} if Config.DATABASE_URL.startswith("sqlite") else {}
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建模型基类
Base = declarative_base()

def get_db() -> Session:
    """
    获取数据库会话
    
    Returns:
        Session: SQLAlchemy数据库会话
    """
    db = SessionLocal()
    try:
        return db
    finally:
        db.close() 