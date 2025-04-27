"""
应用程序配置设置
"""
import os
from typing import Dict, Any

class Config:
    # 数据库设置
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sql_app.db")
    
    # API设置
    API_V1_STR = "/api/v1"
    
    # 安全设置
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    
    # 应用设置
    DEBUG = True
    PROJECT_NAME = "Face Recognition API"
    
    @classmethod
    def get_settings(cls) -> Dict[str, Any]:
        """
        获取所有设置
        
        Returns:
            Dict[str, Any]: 所有配置设置的字典
        """
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        } 