"""
人脸服务类
"""
from typing import Optional, List, Dict, Any, Tuple, Set
from sqlalchemy.orm import Session
import base64
import cv2
import numpy as np
import io
import os
import time
import torch
import insightface
from PIL import Image
from insightface.utils import face_align
from arcface import Arcface
from fastapi import HTTPException, status
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import threading
from collections import deque
import hashlib
from fastapi.responses import JSONResponse
from fastapi import Request
from fastapi.exceptions import RequestValidationError

from ..repositories import face_repository
from ..models.face import Face
from ..schemas.request import CreateOrUpdateFaceRequest, VerifyFaceRequest
from ..schemas.response import FaceData, VerifyResult
from openapi_server.utils.logger import face_logger, log_face_verification, log_face_registration, log_liveness_check

# 初始化模型
face_recognition_model = Arcface()
face_detect_model = insightface.app.FaceAnalysis(name="buffalo_l", root="insightface_models/")
face_detect_model.prepare(ctx_id=0, det_size=(640, 640))

# 创建线程池
thread_pool = ThreadPoolExecutor(max_workers=4)

# 创建缓存锁
cache_lock = threading.Lock()

# 创建LRU缓存
class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.lru = deque()
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[np.ndarray]:
        with self.lock:
            if key in self.cache:
                self.lru.remove(key)
                self.lru.append(key)
                return self.cache[key]
            return None

    def put(self, key: str, value: np.ndarray):
        with self.lock:
            if key in self.cache:
                self.lru.remove(key)
            elif len(self.cache) >= self.capacity:
                oldest = self.lru.popleft()
                del self.cache[oldest]
            self.cache[key] = value
            self.lru.append(key)

# 创建缓存实例
face_detect_cache = LRUCache(100)
image_hash_cache = LRUCache(100)

@lru_cache(maxsize=1000)
def get_image_hash(image_bytes: bytes) -> str:
    """计算图像哈希值用于缓存"""
    return hashlib.md5(image_bytes).hexdigest()

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """预处理图像以提高性能"""
    # 调整图像大小以加快处理速度
    height, width = image.shape[:2]
    if width > 640 or height > 640:
        scale = min(640/width, 640/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image

def batch_face_detect(images: List[np.ndarray]) -> List[Optional[np.ndarray]]:
    """批量处理人脸检测"""
    results = [None] * len(images)
    
    def process_single_image(idx: int, image: np.ndarray):
        try:
            image_hash = get_image_hash(image.tobytes())
            cached_result = face_detect_cache.get(image_hash)
            if cached_result is not None:
                results[idx] = cached_result
                return

            # 预处理图像
            processed_image = preprocess_image(image)
            faces = face_detect_model.get(processed_image)
            
            if len(faces) <= 0:
                return
                
            face = faces[0]
            aligned_face = face_align.norm_crop(processed_image, landmark=face.kps, image_size=112)
            face_detect_cache.put(image_hash, aligned_face)
            results[idx] = aligned_face
        except Exception:
            pass

    # 并行处理图像
    futures = []
    for i, image in enumerate(images):
        futures.append(thread_pool.submit(process_single_image, i, image))
    
    # 等待所有任务完成
    for future in as_completed(futures):
        try:
            future.result()
        except Exception:
            continue
    
    return results

def calculate_eye_aspect_ratio(landmarks):
    """计算眼睛纵横比（EAR）"""
    if landmarks is None or len(landmarks) < 5:
        return 0
    
    try:
        left_eye = landmarks[0:2]
        right_eye = landmarks[2:4]
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        return (left_ear + right_ear) / 2.0
    except Exception:
        return 0

def calculate_ear(eye_points):
    """计算单只眼睛的纵横比"""
    try:
        if len(eye_points) < 2:
            return 0
        width = np.linalg.norm(eye_points[0] - eye_points[1])
        height = width * 0.3
        return height / width
    except Exception:
        return 0

def calculate_mouth_aspect_ratio(landmarks):
    """计算嘴部纵横比（MAR）"""
    if landmarks is None or len(landmarks) < 5:
        return 0
    
    try:
        left_mouth = landmarks[3]
        right_mouth = landmarks[4]
        width = np.linalg.norm(left_mouth - right_mouth)
        mouth_center = (left_mouth + right_mouth) / 2
        nose = landmarks[2]
        height = np.linalg.norm(mouth_center - nose)
        return height / width
    except Exception:
        return 0

@lru_cache(maxsize=1000)
def calculate_image_similarity(img1_bytes: bytes, img2_bytes: bytes) -> float:
    """计算两张图片的相似度"""
    try:
        img1 = np.frombuffer(img1_bytes, dtype=np.uint8).reshape((112, 112, 3))
        img2 = np.frombuffer(img2_bytes, dtype=np.uint8).reshape((112, 112, 3))
        
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        ssim_score = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)[0][0]
        mse = np.mean((gray1 - gray2) ** 2)
        mse_score = 1.0 / (1.0 + mse)
        
        return 0.7 * ssim_score + 0.3 * mse_score
    except Exception:
        return 0.0

@log_liveness_check
def check_liveness(image: np.ndarray, previous_frames: List[np.ndarray] = None) -> bool:
    """检查图片是否为真实人脸（防照片攻击）"""
    try:
        image = preprocess_image(image)
        faces = face_detect_model.get(image)
        if len(faces) == 0:
            print("活体检测失败: 未检测到人脸")
            return False
            
        face = faces[0]
        aligned_face = face_align.norm_crop(image, landmark=face.kps, image_size=112)
        landmarks = face.kps
        
        if landmarks is None:
            print("活体检测失败: 未检测到关键点")
            return False
        
        # 图片质量特征分析
        gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness_std = np.std(gray)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.count_nonzero(edges) / (edges.shape[0] * edges.shape[1])
        face_std = np.std(gray)
        
        # 人脸关键点检查
        if len(landmarks) >= 5:
            left_eye = landmarks[0]
            right_eye = landmarks[2]
            nose = landmarks[4]
            left_mouth = landmarks[3]
            right_mouth = landmarks[4]
            
            eye_distance = np.linalg.norm(left_eye - right_eye)
            left_eye_nose = np.linalg.norm(left_eye - nose)
            right_eye_nose = np.linalg.norm(right_eye - nose)
            mouth_width = np.linalg.norm(left_mouth - right_mouth)
            
            eye_nose_ratio = (left_eye_nose + right_eye_nose) / (2 * eye_distance)
            mouth_eye_ratio = mouth_width / eye_distance
            
            # 降低人脸比例检查的严格度
            face_ratio_check = 0.4 < eye_nose_ratio < 1.8 and 0.6 < mouth_eye_ratio < 1.8
            print(f"人脸比例检查: eye_nose_ratio={eye_nose_ratio:.2f}, mouth_eye_ratio={mouth_eye_ratio:.2f}, 结果={face_ratio_check}")
            
            if not face_ratio_check:
                face_logger.log_error("人脸比例检查失败", {
                    "eye_nose_ratio": eye_nose_ratio,
                    "mouth_eye_ratio": mouth_eye_ratio
                })
        else:
            face_ratio_check = False
            face_logger.log_error("关键点数量不足", {"landmarks_count": len(landmarks)})
            print("活体检测失败: 关键点数量不足")
        
        # 其他特征检查
        lbp = cv2.calcHist([gray], [0], None, [256], [0, 256])
        lbp_std = np.std(lbp)
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_std = np.std(gradient_magnitude)
        
        contrast = np.std(gray) / np.mean(gray)
        
        if len(landmarks) >= 5:
            face_width = face.bbox[2] - face.bbox[0]
            face_height = face.bbox[3] - face.bbox[1]
            face_aspect_ratio = face_width / face_height
            depth_check = 0.7 < face_aspect_ratio < 1.3
            print(f"深度检查: face_aspect_ratio={face_aspect_ratio:.2f}, 结果={depth_check}")
            
            if not depth_check:
                face_logger.log_error("深度检查失败", {"face_aspect_ratio": face_aspect_ratio})
        else:
            depth_check = False
            
        face_hsv = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2HSV)
        saturation_std = np.std(face_hsv[:,:,1])
        value_std = np.std(face_hsv[:,:,2])
        material_check = saturation_std > 20 and value_std > 20
        print(f"材质检查: saturation_std={saturation_std:.2f}, value_std={value_std:.2f}, 结果={material_check}")
        
        if not material_check:
            face_logger.log_error("材质检查失败", {
                "saturation_std": saturation_std,
                "value_std": value_std
            })
        
        light_regions = np.array_split(gray, 4)
        light_vars = [np.var(region) for region in light_regions]
        light_var_std = np.std(light_vars)
        light_check = light_var_std > 100
        print(f"光照检查: light_var_std={light_var_std:.2f}, 结果={light_check}")
        
        if not light_check:
            face_logger.log_error("光照检查失败", {"light_var_std": light_var_std})
        
        # 动态特征检测
        dynamic_check = False
        if previous_frames and len(previous_frames) >= 1:
            # 检查图片相似度
            similarities = []
            for prev_frame in previous_frames:
                similarity = calculate_image_similarity(aligned_face.tobytes(), prev_frame.tobytes())
                similarities.append(similarity)
            
            # 如果与任何一帧的相似度都过高，则判定为失败
            if any(similarity > 0.95 for similarity in similarities):
                face_logger.log_error("图片相似度过高", {"similarities": similarities})
                print("活体检测失败: 图片相似度过高")
                return False
            
            # 张嘴检测
            current_mouth_aspect_ratio = calculate_mouth_aspect_ratio(landmarks)
            mouth_ratios = []
            
            for frame in previous_frames:
                frame_faces = face_detect_model.get(frame)
                if len(frame_faces) > 0 and len(frame_faces[0].kps) >= 5:
                    mouth_ratio = calculate_mouth_aspect_ratio(frame_faces[0].kps)
                    mouth_ratios.append(mouth_ratio)
            
            if mouth_ratios:
                # 计算当前帧与所有之前帧的嘴部比例差异
                mouth_diffs = [abs(current_mouth_aspect_ratio - ratio) for ratio in mouth_ratios]
                max_diff = max(mouth_diffs)
                avg_diff = sum(mouth_diffs) / len(mouth_diffs)
                
                # 使用最大差异和平均差异来判断张嘴动作
                mouth_check = max_diff > 0.25 or avg_diff > 0.15
                print(f"张嘴检测: current_ratio={current_mouth_aspect_ratio:.4f}, max_diff={max_diff:.4f}, avg_diff={avg_diff:.4f}, 结果={mouth_check}")
                
                if not mouth_check:
                    face_logger.log_error("张嘴检测失败", {
                        "current_ratio": current_mouth_aspect_ratio,
                        "max_diff": max_diff,
                        "avg_diff": avg_diff,
                        "mouth_ratios": mouth_ratios
                    })
            else:
                mouth_check = False
                face_logger.log_error("张嘴检测失败: 无法计算嘴部比例", {"previous_frames": len(previous_frames)})
                print("张嘴检测失败: 无法计算嘴部比例")
                
            # 检查图片之间的差异
            frame_diffs = []
            for prev_frame in previous_frames:
                prev_faces = face_detect_model.get(prev_frame)
                if len(prev_faces) > 0:
                    prev_aligned = face_align.norm_crop(prev_frame, landmark=prev_faces[0].kps, image_size=112)
                    if prev_aligned.shape == aligned_face.shape:
                        diff = cv2.absdiff(aligned_face, prev_aligned)
                        frame_diffs.append(np.mean(diff))
            
            # 降低帧差异检查的阈值
            frame_diff_check = any(diff > 1.0 for diff in frame_diffs) if frame_diffs else False
            print(f"帧差异检查: diffs={[f'{d:.4f}' for d in frame_diffs]}, 结果={frame_diff_check}")
            
            if not frame_diff_check:
                face_logger.log_error("帧差异检查失败", {"frame_diffs": frame_diffs})
            
            # 检查关键点变化
            landmark_changes = []
            for prev_frame in previous_frames:
                prev_faces = face_detect_model.get(prev_frame)
                if len(prev_faces) > 0 and len(prev_faces[0].kps) >= 5:
                    prev_landmarks = prev_faces[0].kps
                    landmark_diff = np.mean(np.abs(landmarks - prev_landmarks))
                    landmark_changes.append(landmark_diff)
            
            # 降低关键点变化检查的阈值
            landmark_change_check = any(change > 0.6 for change in landmark_changes)
            print(f"关键点变化检查: changes={[f'{c:.4f}' for c in landmark_changes]}, 结果={landmark_change_check}")
            
            if not landmark_change_check:
                face_logger.log_error("关键点变化检查失败", {"landmark_changes": landmark_changes})
            
            # 使用所有帧进行动态特征检测
            dynamic_check = mouth_check and (frame_diff_check or landmark_change_check)
            print(f"动态特征检测结果: {dynamic_check}")
            
            if not dynamic_check:
                face_logger.log_error("动态特征检测失败", {
                    "mouth_check": mouth_check,
                    "frame_diff_check": frame_diff_check,
                    "landmark_change_check": landmark_change_check
                })
        else:
            face_logger.log_error("没有足够的帧进行动态特征检测", {"previous_frames": len(previous_frames) if previous_frames else 0})
            print("活体检测失败: 没有足够的帧进行动态特征检测")
            return False
        
        # 特征判断
        QUALITY_CHECKS = {
            # 图像清晰度检查 - 基于论文 "Face Anti-Spoofing Using Image Quality Assessment"
            'laplacian_var': laplacian_var > 100,  # 提高清晰度要求，确保图像细节丰富
            
            # 亮度检查 - 基于论文 "Face Liveness Detection Using Image Quality Assessment"
            'brightness_std': 35 < brightness_std < 85,  # 更严格的亮度范围，避免过暗或过亮
            
            # 边缘密度检查 - 基于论文 "Face Anti-Spoofing Using Texture Analysis"
            'edge_density': 0.015 < edge_density < 0.22,  # 更严格的边缘密度范围，确保清晰的边缘特征
            
            # 人脸区域标准差 - 基于论文 "Face Liveness Detection Using Image Quality Assessment"
            'face_std': 35 < face_std < 85,  # 更严格的标准差范围，确保人脸区域质量
            
            # 人脸比例检查 - 保持原有逻辑
            'face_ratio': face_ratio_check,
            
            # 纹理特征检查 - 基于论文 "Face Anti-Spoofing Using Texture Analysis"
            'texture': lbp_std > 60,  # 提高纹理特征要求，确保丰富的纹理信息
            
            # 梯度标准差检查 - 基于论文 "Face Liveness Detection Using Image Quality Assessment"
            'gradient': 25 < gradient_std < 75,  # 更严格的梯度范围，确保图像细节
            
            # 对比度检查 - 基于论文 "Face Anti-Spoofing Using Image Quality Assessment"
            'contrast': 0.25 < contrast < 0.75,  # 更严格的对比度范围，确保合适的图像对比度
            
            # 深度检查 - 保持原有逻辑
            'depth': depth_check,
            
            # 材质检查 - 基于论文 "Face Anti-Spoofing Using Material Analysis"
            'material': material_check,
            
            # 光照检查 - 基于论文 "Face Liveness Detection Using Image Quality Assessment"
            'light': light_check,
            
            # 动态特征检测 - 保持原有逻辑
            'dynamic': dynamic_check,
        }
        
        print("\n质量检查结果:")
        for check, result in QUALITY_CHECKS.items():
            print(f"{check}: {result}")
            if not result:
                face_logger.log_error(f"质量检查失败: {check}", {"check_value": locals().get(check, None)})
        
        # 分离动态特征检测和其他检查
        dynamic_check_result = QUALITY_CHECKS.pop('dynamic')
        other_checks = list(QUALITY_CHECKS.values())
        
        # 计算其他检查的失败数量
        failed_checks = sum(1 for check in other_checks if not check)
        other_checks_passed = failed_checks <= 2
        
        # 最终结果需要动态特征检测通过，且其他检查最多失败两个
        return dynamic_check_result and other_checks_passed
        
    except Exception as e:
        face_logger.log_error("活体检测过程发生异常", {"error": str(e)})
        return False

def calculate_head_pose(landmarks):
    """计算头部姿态（欧拉角）"""
    if landmarks is None or len(landmarks) < 5:
        return np.zeros(3)
    
    # 使用关键点计算简单的头部姿态
    # 这里使用简化的方法，实际应用中可以使用更复杂的3D重建方法
    nose = landmarks[2]
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    
    # 计算眼睛中点
    eye_center = (left_eye + right_eye) / 2
    
    # 计算头部姿态（简化为2D角度）
    dx = nose[0] - eye_center[0]
    dy = nose[1] - eye_center[1]
    
    # 计算偏航角（yaw）和俯仰角（pitch）
    yaw = np.arctan2(dx, dy)
    pitch = np.arctan2(dy, np.sqrt(dx*dx + dy*dy))
    
    return np.array([yaw, pitch, 0])  # roll角设为0

@lru_cache(maxsize=1000)
def decode_base64_to_image(encoded_string: str) -> Image.Image:
    """将base64编码的图片转换为PIL Image对象"""
    try:
        encoded_string = re.sub(r'\s+', '', encoded_string)
        padding = len(encoded_string) % 4
        if padding:
            encoded_string += '=' * (4 - padding)
            
        image_data = base64.b64decode(encoded_string)
        return Image.open(io.BytesIO(image_data))
    except Exception as e:
        raise ValueError(f"无效的base64图片数据: {str(e)}")

def face_detect(image: np.ndarray) -> Optional[np.ndarray]:
    """检测人脸并返回对齐后的人脸图像"""
    image_hash = get_image_hash(image.tobytes())
    cached_result = face_detect_cache.get(image_hash)
    if cached_result is not None:
        return cached_result
        
    faces = face_detect_model.get(image)
    
    if len(faces) <= 0:
        return None
    else:
        face = faces[0]
        aligned_face = face_align.norm_crop(image, landmark=face.kps, image_size=112)
        
        with cache_lock:
            face_detect_cache.put(image_hash, aligned_face)
            
        return aligned_face

def process_single_image(face_image_base64: str, registered_face: Image.Image) -> Tuple[bool, Optional[Exception]]:
    """处理单张图片的验证"""
    try:
        face_image_base64 = re.sub(r'\s+', '', face_image_base64)
        padding = len(face_image_base64) % 4
        if padding:
            face_image_base64 += '=' * (4 - padding)
            
        verify_image = decode_base64_to_image(face_image_base64)
        cv_verify = np.array(verify_image)
        cv_verify = cv2.cvtColor(cv_verify, cv2.COLOR_RGB2BGR)
        
        if not check_liveness(cv_verify):
            raise ValueError("活体检测失败，请确保使用真实人脸")
        
        verify_face = face_detect(cv_verify)
        if verify_face is None:
            raise ValueError("待验证的图片中未检测到人脸")
        
        verify_face = cv2.cvtColor(verify_face, cv2.COLOR_BGR2RGB)
        verify_face = Image.fromarray(verify_face)
        
        probability = face_recognition_model.detect_image(registered_face, verify_face)
        
        return probability <= 1.18, None
    except Exception as e:
        return False, e

def batch_process_images(face_images_base64: List[str], registered_face: Image.Image) -> List[Tuple[bool, Optional[Exception]]]:
    """批量处理图片验证"""
    cv_images = []
    for face_image_base64 in face_images_base64:
        try:
            face_image_base64 = re.sub(r'\s+', '', face_image_base64)
            padding = len(face_image_base64) % 4
            if padding:
                face_image_base64 += '=' * (4 - padding)
                
            verify_image = decode_base64_to_image(face_image_base64)
            cv_verify = np.array(verify_image)
            cv_verify = cv2.cvtColor(cv_verify, cv2.COLOR_RGB2BGR)
            cv_images.append(cv_verify)
        except Exception as e:
            raise ValueError(f"图片格式错误: {str(e)}")
    
    image_hashes = set()
    for image in cv_images:
        if image is not None:
            image_hash = get_image_hash(image.tobytes())
            image_hashes.add(image_hash)
    
    if len(image_hashes) <= 1:
        raise ValueError("检测到相同的图片，请确保使用不同的图片")
    
    detected_faces = batch_face_detect(cv_images)
    
    results = []
    for i, (cv_image, detected_face) in enumerate(zip(cv_images, detected_faces)):
        if cv_image is None or detected_face is None:
            raise ValueError("图片中未检测到人脸")
            
        try:
            previous_frames = cv_images[:i] if i > 0 else []
            if not check_liveness(cv_image, previous_frames):
                raise ValueError("活体检测失败，请确保使用真实人脸")
                
            verify_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGB)
            verify_face = Image.fromarray(verify_face)
            
            probability = face_recognition_model.detect_image(registered_face, verify_face)
            results.append((probability <= 1.18, None))
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"图片处理失败: {str(e)}")
    
    return results

class FaceService:
    """人脸服务实现类"""
    
    # 定义相似度阈值
    SIMILARITY_THRESHOLD = 0.6
    
    @staticmethod
    def get_user_face(db: Session, user_id: int) -> Optional[Dict[str, Any]]:
        """
        获取用户的人脸数据
        
        Args:
            db: 数据库会话
            user_id: 用户ID
            
        Returns:
            Optional[Dict[str, Any]]: 人脸数据字典，不存在则返回None
            
        Raises:
            Exception: 数据库操作异常
        """
        try:
            face = face_repository.get_latest_face(db, user_id)
            if not face:
                return None
                
            return {
                "faceId": face.faceid,
                "userId": face.userid,
                "faceImageBase64": face.faceImageBase64,
                "createdAt": face.createdAt,
                "updatedAt": face.updatedAt
            }
        except Exception as e:
            raise Exception(f"获取人脸数据失败: {str(e)}")

    @log_face_registration
    def create_or_update_face(db: Session, user_id: int, face_image_base64: str) -> Dict[str, Any]:
        """创建或更新用户的人脸数据"""
        try:
            if face_image_base64 is None:
                raise ValueError("空图片数据")
                
            # 清理base64字符串
            face_image_base64 = re.sub(r'\s+', '', face_image_base64)
            padding = len(face_image_base64) % 4
            if padding:
                face_image_base64 += '=' * (4 - padding)
            
            try:
                # 解码Base64字符串
                image_data = base64.b64decode(face_image_base64)
            except Exception as e:
                raise ValueError("图片数据格式错误，无法解码Base64")
            
            try:
                # 将二进制数据转换为numpy数组
                nparr = np.frombuffer(image_data, np.uint8)
                cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                raise ValueError("无法解码图片数据")
            
            if cv_image is None:
                raise ValueError("无法解码图片数据")
            
            # 预处理图像
            cv_image = preprocess_image(cv_image)
            
            # 只进行基本的人脸检测，不进行活体检测
            aligned_face = face_detect(cv_image)
            if aligned_face is None:
                raise ValueError("图片中未检测到人脸")
            
            # 查找或创建人脸记录
            face = db.query(Face).filter(Face.userid == user_id).first()
            if not face:
                face = Face(userid=user_id)
                db.add(face)
            
            # 更新人脸数据
            face.faceImageBase64 = face_image_base64
            db.commit()
            db.refresh(face)
            
            return {
                "faceId": face.faceid,
                "userId": face.userid,
                "faceImageBase64": face.faceImageBase64,
                "createdAt": face.createdAt,
                "updatedAt": face.updatedAt
            }
        except ValueError as e:
            raise ValueError(str(e))
        except Exception as e:
            raise Exception(f"保存人脸数据时发生异常: {str(e)}")

    @staticmethod
    async def batch_process_images(face_images_base64: List[str]) -> List[Dict[str, Any]]:
        """
        批量处理图片
        
        Args:
            face_images_base64: Base64编码的图片列表
            
        Returns:
            List[Dict[str, Any]]: 处理结果列表，每个结果包含face_detected和face_encoding
        """
        results = []
        for img_base64 in face_images_base64:
            try:
                # 解码图片
                image = decode_base64_to_image(img_base64)
                cv_image = np.array(image)
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                
                # 检测人脸
                face = face_detect(cv_image)
                if face is not None:
                    # 进行活体检测
                    if check_liveness(cv_image):
                        # 获取人脸编码
                        face_encoding = face_recognition_model.get_face_encoding(face)
                        results.append({
                            "face_detected": True,
                            "face_encoding": face_encoding
                        })
                    else:
                        results.append({
                            "face_detected": False,
                            "error": "活体检测失败"
                        })
                else:
                    results.append({
                        "face_detected": False,
                        "error": "未检测到人脸"
                    })
            except Exception as e:
                results.append({
                    "face_detected": False,
                    "error": str(e)
                })
        return results

    @staticmethod
    def _calculate_similarity(encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        """
        计算两个人脸编码的相似度
        
        Args:
            encoding1: 第一个人脸编码
            encoding2: 第二个人脸编码
            
        Returns:
            float: 相似度分数
        """
        return face_recognition_model.calculate_similarity(encoding1, encoding2)

    @log_face_verification
    async def verify_face(db: Session, user_id: int, face_images_base64: List[str]) -> bool:
        """验证人脸数据"""
        try:
            # 获取用户的人脸数据
            face_data = FaceService.get_user_face(db, user_id)
            if not face_data:
                face_logger.log_error("用户未设置人脸数据", {"user_id": user_id})
                raise ValueError("用户未设置人脸数据")
            
            # 验证图片数量
            if not face_images_base64:
                face_logger.log_error("图片列表为空", {"user_id": user_id})
                raise ValueError("图片列表不能为空")
            if len(face_images_base64) > 10:
                face_logger.log_error("图片数量超过限制", {
                    "user_id": user_id,
                    "image_count": len(face_images_base64)
                })
                raise ValueError("图片数量超过限制，最多支持10张图片")
            
            # 预处理所有图片
            cv_images = []
            for img_base64 in face_images_base64:
                try:
                    image = decode_base64_to_image(img_base64)
                    cv_image = np.array(image)
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                    cv_images.append(cv_image)
                except Exception as e:
                    face_logger.log_error("图片预处理失败", {
                        "user_id": user_id,
                        "error": str(e)
                    })
                    continue
            
            if not cv_images:
                face_logger.log_error("未能成功处理任何图片", {"user_id": user_id})
                raise ValueError("未能成功处理任何图片")
            
            # 进行活体检测
            liveness_results = []
            for i, cv_image in enumerate(cv_images):
                previous_frame = [cv_images[i-1]] if i > 0 else []
                liveness_result = check_liveness(cv_image, previous_frame)
                liveness_results.append(liveness_result)
            
            if not any(liveness_results):
                face_logger.log_error("所有图片均未通过活体检测", {
                    "user_id": user_id,
                    "total_images": len(cv_images),
                    "liveness_results": liveness_results
                })
                raise ValueError("所有图片均未通过活体检测，请确保使用真实人脸并完成张嘴动作")
            
            # 获取注册人脸图片
            try:
                registered_base64 = re.sub(r'\s+', '', face_data["faceImageBase64"])
                padding = len(registered_base64) % 4
                if padding:
                    registered_base64 += '=' * (4 - padding)
                    
                registered_image = decode_base64_to_image(registered_base64)
                cv_registered = np.array(registered_image)
                cv_registered = cv2.cvtColor(cv_registered, cv2.COLOR_RGB2BGR)
                registered_face = face_detect(cv_registered)
                if registered_face is None:
                    face_logger.log_error("注册的人脸图片中未检测到人脸", {"user_id": user_id})
                    raise ValueError("注册的人脸图片中未检测到人脸")
                registered_face = cv2.cvtColor(registered_face, cv2.COLOR_BGR2RGB)
                registered_face = Image.fromarray(registered_face)
            except Exception as e:
                face_logger.log_error("处理注册人脸图片时发生错误", {
                    "user_id": user_id,
                    "error": str(e)
                })
                raise ValueError(f"处理注册人脸图片时发生错误: {str(e)}")
            
            # 批量进行人脸验证
            for i, cv_image in enumerate(cv_images):
                if liveness_results[i]:  # 只验证通过活体检测的图片
                    try:
                        verify_face = face_detect(cv_image)
                        if verify_face is not None:
                            verify_face = cv2.cvtColor(verify_face, cv2.COLOR_BGR2RGB)
                            verify_face = Image.fromarray(verify_face)
                            probability = face_recognition_model.detect_image(registered_face, verify_face)
                            print(f"图片 {i+1} 人脸比对结果: probability={float(probability):.4f}, 是否匹配={probability <= 1.18}")
                            
                            if probability <= 1.18:  # 如果匹配成功，立即返回
                                face_logger.log_face_verification(
                                    user_id=user_id,
                                    success=True,
                                    details={
                                        "image_index": i,
                                        "probability": float(probability),
                                        "threshold": 1.18
                                    }
                                )
                                print(f"图片 {i+1} 验证通过，提前返回")
                                return True
                            else:
                                face_logger.log_face_verification(
                                    user_id=user_id,
                                    success=False,
                                    details={
                                        "image_index": i,
                                        "probability": float(probability),
                                        "threshold": 1.18
                                    }
                                )
                    except Exception as e:
                        face_logger.log_error(f"图片 {i+1} 人脸比对失败", {
                            "user_id": user_id,
                            "error": str(e)
                        })
                        print(f"图片 {i+1} 人脸比对失败: {str(e)}")
                        continue
            
            face_logger.log_face_verification(
                user_id=user_id,
                success=False,
                details={
                    "total_images": len(cv_images),
                    "liveness_passed": sum(liveness_results)
                }
            )
            print("所有图片验证均未通过")
            return False
            
        except ValueError as e:
            face_logger.log_error("人脸验证参数错误", {
                "user_id": user_id,
                "error": str(e)
            })
            raise ValueError(str(e))
        except Exception as e:
            face_logger.log_error("人脸验证发生异常", {
                "user_id": user_id,
                "error": str(e)
            })
            raise Exception(f"人脸验证发生异常: {str(e)}")

    @staticmethod
    def _validate_face_image(face_image_base64: str) -> bool:
        """验证Base64编码的人脸图像"""
        try:
            face_image_base64 = re.sub(r'\s+', '', face_image_base64)
            padding = len(face_image_base64) % 4
            if padding:
                face_image_base64 += '=' * (4 - padding)
                
            image_data = base64.b64decode(face_image_base64)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                return False

            if image.shape == (1, 1, 3):
                return True

            # 预处理图像
            image = preprocess_image(image)
            
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            return len(faces) > 0

        except Exception:
            return False 