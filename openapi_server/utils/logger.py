import logging
import os
import functools
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional, Callable

class FaceLogger:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FaceLogger, cls).__new__(cls)
            cls._instance._initialize_logger()
        return cls._instance
    
    def _initialize_logger(self):
        """初始化日志记录器"""
        self.logger = logging.getLogger('face_service')
        self.logger.setLevel(logging.INFO)
        
        # 创建日志目录
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # 创建文件处理器
        log_file = os.path.join(log_dir, f'face_service_{datetime.now().strftime("%Y%m%d")}.log')
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        
        # 设置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
    
    def log_face_verification(
        self,
        user_id: int,
        success: bool,
        details: Dict[str, Any],
        error: Optional[str] = None,
        failure_reason: Optional[str] = None
    ):
        """记录人脸验证日志"""
        log_data = {
            'user_id': user_id,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'details': details
        }
        
        if error:
            log_data['error'] = error
        if failure_reason:
            log_data['failure_reason'] = failure_reason
            
        if success:
            self.logger.info(f"Face verification successful: {log_data}")
        else:
            self.logger.warning(f"Face verification failed: {log_data}")
    
    def log_face_registration(
        self,
        user_id: int,
        success: bool,
        details: Dict[str, Any],
        error: Optional[str] = None
    ):
        """记录人脸注册日志"""
        log_data = {
            'user_id': user_id,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'details': details
        }
        
        if error:
            log_data['error'] = error
            
        if success:
            self.logger.info(f"Face registration successful: {log_data}")
        else:
            self.logger.warning(f"Face registration failed: {log_data}")
    
    def log_error(self, error: str, context: Dict[str, Any], failure_reason: Optional[str] = None):
        """记录错误日志"""
        log_data = {
            'error': error,
            'timestamp': datetime.now().isoformat(),
            'context': context
        }
        if failure_reason:
            log_data['failure_reason'] = failure_reason
        self.logger.error(f"Error occurred: {log_data}")

def extract_failure_reason(func_result: Any, func_name: str) -> Optional[str]:
    """从函数结果中提取失败原因"""
    if func_name == 'check_liveness':
        if not func_result:
            return "活体检测失败"
    elif func_name == 'verify_face':
        if not func_result:
            return "人脸验证失败"
    return None

def log_face_verification(func: Callable):
    """人脸验证日志装饰器"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            result = await func(*args, **kwargs)
            failure_reason = extract_failure_reason(result, func.__name__)
            face_logger.log_face_verification(
                user_id=kwargs.get('user_id'),
                success=result,
                details={
                    'args': str(args),
                    'kwargs': str(kwargs),
                    'result': result
                },
                failure_reason=failure_reason
            )
            return result
        except Exception as e:
            failure_reason = str(e)
            face_logger.log_error(
                str(e),
                {
                    'function': func.__name__,
                    'args': str(args),
                    'kwargs': str(kwargs),
                    'traceback': traceback.format_exc()
                },
                failure_reason=failure_reason
            )
            raise
    return wrapper

def log_face_registration(func: Callable):
    """人脸注册日志装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            face_logger.log_face_registration(
                user_id=kwargs.get('user_id'),
                success=True,
                details={
                    'args': str(args),
                    'kwargs': str(kwargs),
                    'result': result
                }
            )
            return result
        except Exception as e:
            face_logger.log_error(
                str(e),
                {
                    'function': func.__name__,
                    'args': str(args),
                    'kwargs': str(kwargs),
                    'traceback': traceback.format_exc()
                }
            )
            raise
    return wrapper

def log_liveness_check(func: Callable):
    """活体检测日志装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            if not result:
                failure_reason = extract_failure_reason(result, func.__name__)
                face_logger.log_error(
                    "活体检测失败",
                    {
                        'function': func.__name__,
                        'args': str(args),
                        'kwargs': str(kwargs),
                        'result': result
                    },
                    failure_reason=failure_reason
                )
            return result
        except Exception as e:
            face_logger.log_error(
                str(e),
                {
                    'function': func.__name__,
                    'args': str(args),
                    'kwargs': str(kwargs),
                    'traceback': traceback.format_exc()
                }
            )
            raise
    return wrapper

# 创建全局日志记录器实例
face_logger = FaceLogger() 