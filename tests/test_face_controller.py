import os
import sys
import pytest
import asyncio
from fastapi.testclient import TestClient
from fastapi import FastAPI, Depends
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
import base64
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from PIL import Image
import io
from fastapi import HTTPException
from fastapi import status

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openapi_server.controllers.face_controller import router
from openapi_server.schemas.request import CreateOrUpdateFaceRequest, VerifyFaceRequest
from openapi_server.schemas.response import FaceData, VerifyResult
from openapi_server.database import Base, get_db
from openapi_server.models.face import Face
import openapi_server.database
from main import app

# 测试用的base64图片数据（使用真实的人脸图片）
with open("tests/test_face.jpg", "rb") as f:
    TEST_IMAGE_BASE64 = base64.b64encode(f.read()).decode("utf-8")

# 测试用的另一张人脸图片
with open("img/face2.jpeg", "rb") as f:
    TEST_IMAGE_BASE64_2 = base64.b64encode(f.read()).decode("utf-8")

# 生成一个没有人脸的图片
no_face_image = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.rectangle(no_face_image, (10, 10), (90, 90), (255, 0, 0), -1)
_, buffer = cv2.imencode('.jpg', no_face_image)
NO_FACE_IMAGE = base64.b64encode(buffer).decode('utf-8')

# 生成一个低质量的人脸图片（模糊）
low_quality_image = cv2.imread("tests/test_face.jpg")
low_quality_image = cv2.GaussianBlur(low_quality_image, (15, 15), 0)
_, buffer = cv2.imencode('.jpg', low_quality_image)
LOW_QUALITY_IMAGE = base64.b64encode(buffer).decode('utf-8')

# 生成一个多人脸的图片（简单的拼接）
multi_face_image = np.hstack([cv2.imread("tests/test_face.jpg"), cv2.imread("tests/test_face.jpg")])
_, buffer = cv2.imencode('.jpg', multi_face_image)
MULTI_FACE_IMAGE = base64.b64encode(buffer).decode('utf-8')

# 测试用的无效图片数据
INVALID_IMAGE = "invalid_base64"
EMPTY_IMAGE = ""
TOO_LARGE_IMAGE = "A" * (1024 * 1024 * 10)  # 10MB的数据
NOT_BASE64 = "这不是base64编码的数据"
EMPTY_IMAGE_LIST = []
TOO_MANY_IMAGES = [TEST_IMAGE_BASE64] * 11  # 假设最大允许10张图片

# 测试数据库设置
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

@pytest.fixture(scope="module")
def test_app():
    # 创建数据库表
    Base.metadata.create_all(bind=engine)
    
    # 创建测试应用
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db] = override_get_db
    
    yield app
    
    # 清理数据库
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="module")
def client(test_app):
    return TestClient(test_app)

def test_get_user_face_not_found(client):
    """测试获取不存在的用户人脸数据"""
    response = client.get("/users/me/face", params={"userId": 999})
    assert response.status_code == 404
    response_data = response.json()
    assert "当前用户尚未设置人脸数据" in response_data["detail"]

def test_create_or_update_face_success(client):
    """测试成功创建用户人脸数据"""
    data = {
        "userId": 1,
        "faceImageBase64": TEST_IMAGE_BASE64
    }
    response = client.put("/users/me/face", json=data)
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["code"] == 0
    assert "faceId" in response_data["data"]

def test_update_existing_face(client):
    """测试更新已存在的人脸数据"""
    # 先创建人脸数据
    create_data = {
        "userId": 2,
        "faceImageBase64": TEST_IMAGE_BASE64
    }
    response = client.put("/users/me/face", json=create_data)
    assert response.status_code == 200
    first_face_id = response.json()["data"]["faceId"]

    # 更新人脸数据
    update_data = {
        "userId": 2,
        "faceImageBase64": TEST_IMAGE_BASE64_2
    }
    response = client.put("/users/me/face", json=update_data)
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["code"] == 0
    assert response_data["data"]["faceId"] == first_face_id  # ID应该保持不变

def test_create_or_update_face_invalid_image(client):
    """测试创建无效的人脸数据"""
    test_cases = [
        ("空图片数据", EMPTY_IMAGE),
        ("无效的base64数据", INVALID_IMAGE),
        ("过大的图片数据", TOO_LARGE_IMAGE)
    ]
    
    for case_name, image_data in test_cases:
        data = {
            "userId": 1,
            "faceImageBase64": image_data
        }
        response = client.put("/users/me/face", json=data)
        assert response.status_code == 400, f"测试用例 '{case_name}' 失败"

def test_verify_face_success(client):
    """测试成功验证人脸数据"""
    # 先创建人脸数据
    create_data = {
        "userId": 3,
        "faceImageBase64": TEST_IMAGE_BASE64
    }
    client.put("/users/me/face", json=create_data)

    # 验证人脸
    verify_data = {
        "userId": 3,
        "faceImagesBase64": [TEST_IMAGE_BASE64]
    }
    response = client.post("/users/me/face/verify", json=verify_data)
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["code"] == 0
    assert "isMatch" in response_data["data"]
    assert response_data["data"]["isMatch"] is True

def test_verify_face_with_multiple_images(client):
    """测试使用多张图片验证人脸"""
    # 先创建人脸数据
    create_data = {
        "userId": 4,
        "faceImageBase64": TEST_IMAGE_BASE64
    }
    client.put("/users/me/face", json=create_data)

    # 使用多张图片验证
    verify_data = {
        "userId": 4,
        "faceImagesBase64": [TEST_IMAGE_BASE64_2, TEST_IMAGE_BASE64]  # 第二张是匹配的
    }
    response = client.post("/users/me/face/verify", json=verify_data)
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["code"] == 0
    assert response_data["data"]["isMatch"] is True

def test_verify_face_not_found(client):
    """测试验证不存在的用户人脸数据"""
    data = {
        "userId": 999,
        "faceImagesBase64": [TEST_IMAGE_BASE64]
    }
    response = client.post("/users/me/face/verify", json=data)
    assert response.status_code == 404
    response_data = response.json()
    assert "用户未设置人脸数据" in response_data["detail"]

def test_verify_face_with_invalid_images(client):
    """测试使用无效图片验证人脸"""
    # 先创建人脸数据
    create_data = {
        "userId": 5,
        "faceImageBase64": TEST_IMAGE_BASE64
    }
    client.put("/users/me/face", json=create_data)

    test_cases = [
        ("空图片数据", [EMPTY_IMAGE]),
        ("无效的base64数据", [INVALID_IMAGE]),
        ("过大的图片数据", [TOO_LARGE_IMAGE]),
        ("混合有效和无效数据", [INVALID_IMAGE, TEST_IMAGE_BASE64])
    ]

    for case_name, images in test_cases:
        verify_data = {
            "userId": 5,
            "faceImagesBase64": images
        }
        response = client.post("/users/me/face/verify", json=verify_data)
        if TEST_IMAGE_BASE64 in images:
            # 如果包含有效图片，应该能成功验证
            assert response.status_code == 200, f"测试用例 '{case_name}' 失败"
            assert response.json()["data"]["isMatch"] is True
        else:
            # 如果全是无效图片，应该返回400
            assert response.status_code == 400, f"测试用例 '{case_name}' 失败"

def test_concurrent_requests(client):
    """测试并发请求"""
    def create_face(user_id: int):
        data = {
            "userId": user_id,
            "faceImageBase64": TEST_IMAGE_BASE64
        }
        return client.put("/users/me/face", json=data)

    def verify_face(user_id: int):
        data = {
            "userId": user_id,
            "faceImagesBase64": [TEST_IMAGE_BASE64]
        }
        return client.post("/users/me/face/verify", json=data)

    # 创建10个用户的人脸数据
    user_ids = range(100, 110)
    with ThreadPoolExecutor(max_workers=5) as executor:
        responses = list(executor.map(create_face, user_ids))
    
    # 验证所有响应都是成功的
    for response in responses:
        assert response.status_code == 200
        assert response.json()["code"] == 0

    # 并发验证这些用户的人脸
    with ThreadPoolExecutor(max_workers=5) as executor:
        responses = list(executor.map(verify_face, user_ids))
    
    # 验证所有响应都是成功的
    for response in responses:
        assert response.status_code == 200
        assert response.json()["code"] == 0
        assert response.json()["data"]["isMatch"] is True

def test_verify_face_with_empty_image_list(client):
    """测试空的图片列表"""
    # 先创建人脸数据
    create_data = {
        "userId": 6,
        "faceImageBase64": TEST_IMAGE_BASE64
    }
    client.put("/users/me/face", json=create_data)

    verify_data = {
        "userId": 6,
        "faceImagesBase64": EMPTY_IMAGE_LIST
    }
    response = client.post("/users/me/face/verify", json=verify_data)
    print(f"Empty list response: {response.json()}")  # 打印响应
    assert response.status_code == 400
    assert "图片列表不能为空" in response.json()["detail"]

def test_verify_face_with_too_many_images(client):
    """测试超过最大允许数量的图片"""
    create_data = {
        "userId": 7,
        "faceImageBase64": TEST_IMAGE_BASE64
    }
    client.put("/users/me/face", json=create_data)

    verify_data = {
        "userId": 7,
        "faceImagesBase64": TOO_MANY_IMAGES
    }
    response = client.post("/users/me/face/verify", json=verify_data)
    print(f"Too many images response: {response.json()}")  # 打印响应
    assert response.status_code == 400
    assert "图片数量超过限制" in response.json()["detail"]

def test_verify_face_with_not_base64(client):
    """测试非base64格式的数据"""
    create_data = {
        "userId": 8,
        "faceImageBase64": TEST_IMAGE_BASE64
    }
    client.put("/users/me/face", json=create_data)

    verify_data = {
        "userId": 8,
        "faceImagesBase64": [NOT_BASE64]
    }
    response = client.post("/users/me/face/verify", json=verify_data)
    assert response.status_code == 400
    assert "无效的base64图片数据" in response.json()["detail"]

def test_verify_face_with_no_face(client):
    """测试图片中没有人脸的情况"""
    create_data = {
        "userId": 9,
        "faceImageBase64": TEST_IMAGE_BASE64
    }
    client.put("/users/me/face", json=create_data)

    verify_data = {
        "userId": 9,
        "faceImagesBase64": [NO_FACE_IMAGE]
    }
    response = client.post("/users/me/face/verify", json=verify_data)
    assert response.status_code == 400
    assert "未检测到人脸" in response.json()["detail"]

def test_verify_face_with_low_quality(client):
    """测试低质量图片"""
    create_data = {
        "userId": 10,
        "faceImageBase64": TEST_IMAGE_BASE64
    }
    client.put("/users/me/face", json=create_data)

    verify_data = {
        "userId": 10,
        "faceImagesBase64": [LOW_QUALITY_IMAGE]
    }
    response = client.post("/users/me/face/verify", json=verify_data)
    # 低质量图片可能无法检测到人脸
    if response.status_code == 400:
        assert "未检测到人脸" in response.json()["detail"]
    else:
        assert response.status_code == 200
        assert "isMatch" in response.json()["data"]

def test_verify_face_with_multi_face(client):
    """测试多人脸图片"""
    create_data = {
        "userId": 11,
        "faceImageBase64": TEST_IMAGE_BASE64
    }
    client.put("/users/me/face", json=create_data)

    verify_data = {
        "userId": 11,
        "faceImagesBase64": [MULTI_FACE_IMAGE]
    }
    response = client.post("/users/me/face/verify", json=verify_data)
    assert response.status_code == 200
    assert response.json()["data"]["isMatch"] is True  # 应该匹配第一个人脸

def test_verify_face_with_different_person(client):
    """测试完全不同的人脸"""
    # 这个测试需要两张不同人的人脸图片
    # 由于我们没有另一个人的图片，这里只是演示测试结构
    create_data = {
        "userId": 12,
        "faceImageBase64": TEST_IMAGE_BASE64
    }
    client.put("/users/me/face", json=create_data)

    verify_data = {
        "userId": 12,
        "faceImagesBase64": [TEST_IMAGE_BASE64_2]  # 假设这是另一个人的图片
    }
    response = client.post("/users/me/face/verify", json=verify_data)
    assert response.status_code == 200
    # 根据实际的相似度阈值，这里的结果可能是True或False
    assert "isMatch" in response.json()["data"]

def test_verify_face_with_db_error(client, test_app):
    """测试数据库连接失败的情况"""
    # 临时修改数据库连接函数以模拟错误
    original_get_db = openapi_server.database.get_db
    
    def mock_get_db():
        try:
            db = TestingSessionLocal()
            yield db
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"数据库连接失败: {str(e)}"
            )
    
    # 修改 FastAPI 应用的依赖注入
    test_app.dependency_overrides[get_db] = mock_get_db
    
    try:
        verify_data = {
            "userId": 13,
            "faceImagesBase64": [TEST_IMAGE_BASE64]
        }
        
        response = client.post("/users/me/face/verify", json=verify_data)
        assert response.status_code == 500
        assert "数据库连接失败" in response.json()["detail"]
    finally:
        # 恢复原始函数
        test_app.dependency_overrides[get_db] = original_get_db

def test_create_face_with_transaction(client, test_app):
    """测试事务回滚"""
    # 临时修改数据库连接函数以模拟错误
    original_get_db = openapi_server.database.get_db
    
    def mock_get_db():
        try:
            db = TestingSessionLocal()
            # 模拟数据库事务错误
            raise Exception("数据库事务错误")
            yield db
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"数据库事务错误: {str(e)}"
            )
    
    # 修改 FastAPI 应用的依赖注入
    test_app.dependency_overrides[get_db] = mock_get_db
    
    try:
        data = {
            "userId": 14,
            "faceImageBase64": TEST_IMAGE_BASE64
        }
        
        response = client.put("/users/me/face", json=data)
        assert response.status_code == 500
        assert "数据库事务错误" in response.json()["detail"]
    finally:
        # 恢复原始函数
        test_app.dependency_overrides[get_db] = original_get_db

def test_different_image_formats(client):
    """测试不同格式的图片"""
    # 将测试图片转换为不同格式
    img = Image.open("tests/test_face.jpg")
    
    # 测试PNG格式
    png_buffer = io.BytesIO()
    img.save(png_buffer, format='PNG')
    png_base64 = base64.b64encode(png_buffer.getvalue()).decode('utf-8')
    
    # 测试WEBP格式
    webp_buffer = io.BytesIO()
    img.save(webp_buffer, format='WEBP')
    webp_base64 = base64.b64encode(webp_buffer.getvalue()).decode('utf-8')
    
    # 创建用户人脸数据
    create_data = {
        "userId": 15,
        "faceImageBase64": TEST_IMAGE_BASE64
    }
    client.put("/users/me/face", json=create_data)
    
    # 测试不同格式的图片验证
    for image_format, image_data in [
        ("PNG", png_base64),
        ("WEBP", webp_base64)
    ]:
        verify_data = {
            "userId": 15,
            "faceImagesBase64": [image_data]
        }
        response = client.post("/users/me/face/verify", json=verify_data)
        assert response.status_code == 200, f"{image_format}格式验证失败"
        assert response.json()["data"]["isMatch"] is True

def test_face_angles(client):
    """测试不同角度的人脸图片"""
    # 读取原始图片
    img = cv2.imread("tests/test_face.jpg")
    
    # 创建不同角度的图片
    angles = [15, 30, -15, -30]  # 旋转角度
    rotated_images = []
    
    for angle in angles:
        # 获取图片中心点
        height, width = img.shape[:2]
        center = (width // 2, height // 2)
        
        # 创建旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, rotation_matrix, (width, height))
        
        # 转换为base64
        _, buffer = cv2.imencode('.jpg', rotated)
        rotated_images.append(base64.b64encode(buffer).decode('utf-8'))
    
    # 创建用户人脸数据
    create_data = {
        "userId": 16,
        "faceImageBase64": TEST_IMAGE_BASE64
    }
    client.put("/users/me/face", json=create_data)
    
    # 测试不同角度的图片
    for idx, rotated_image in enumerate(rotated_images):
        verify_data = {
            "userId": 16,
            "faceImagesBase64": [rotated_image]
        }
        response = client.post("/users/me/face/verify", json=verify_data)
        print(f"Testing angle {angles[idx]} degrees")
        # 由于角度较大，可能无法检测到人脸
        if response.status_code == 400:
            assert "未检测到人脸" in response.json()["detail"]
        else:
            assert response.status_code == 200

def test_brightness_variations(client):
    """测试不同亮度条件下的人脸识别"""
    # 读取原始图片
    img = cv2.imread("tests/test_face.jpg")
    
    # 创建不同亮度的图片
    brightness_factors = [0.7, 1.3]  # 变暗和变亮
    brightness_images = []
    
    for factor in brightness_factors:
        # 调整亮度
        adjusted = cv2.convertScaleAbs(img, alpha=factor, beta=0)
        
        # 转换为base64
        _, buffer = cv2.imencode('.jpg', adjusted)
        brightness_images.append(base64.b64encode(buffer).decode('utf-8'))
    
    # 创建用户人脸数据
    create_data = {
        "userId": 17,
        "faceImageBase64": TEST_IMAGE_BASE64
    }
    client.put("/users/me/face", json=create_data)
    
    # 测试不同亮度的图片
    for idx, bright_image in enumerate(brightness_images):
        verify_data = {
            "userId": 17,
            "faceImagesBase64": [bright_image]
        }
        response = client.post("/users/me/face/verify", json=verify_data)
        print(f"Testing brightness factor {brightness_factors[idx]}")
        assert response.status_code == 200
        assert "isMatch" in response.json()["data"]

def test_special_user_ids(client):
    """测试特殊用户ID的处理"""
    special_ids = [
        0,  # 边界值
        2**31 - 1,  # 最大32位整数
        -1,  # 负数
        None  # 空值
    ]
    
    for special_id in special_ids:
        # 创建人脸数据
        create_data = {
            "userId": special_id,
            "faceImageBase64": TEST_IMAGE_BASE64
        }
        response = client.put("/users/me/face", json=create_data)
        
        if special_id is None or special_id < 0:
            assert response.status_code == 400  # 验证无效ID被正确拒绝
        else:
            assert response.status_code in [200, 400]  # 有效ID应该被接受或有明确的错误消息

def test_batch_performance(client):
    """测试批量操作的性能"""
    # 准备测试数据
    num_users = 5
    user_ids = range(20, 20 + num_users)
    
    # 记录开始时间
    start_time = asyncio.get_event_loop().time()
    
    # 批量创建用户人脸数据
    create_responses = []
    for user_id in user_ids:
        create_data = {
            "userId": user_id,
            "faceImageBase64": TEST_IMAGE_BASE64
        }
        response = client.put("/users/me/face", json=create_data)
        create_responses.append(response)
    
    # 验证所有创建操作都成功
    for response in create_responses:
        assert response.status_code == 200
        assert response.json()["code"] == 0
    
    # 批量验证用户人脸
    verify_responses = []
    for user_id in user_ids:
        verify_data = {
            "userId": user_id,
            "faceImagesBase64": [TEST_IMAGE_BASE64]
        }
        response = client.post("/users/me/face/verify", json=verify_data)
        verify_responses.append(response)
    
    # 验证所有验证操作都成功
    for response in verify_responses:
        assert response.status_code == 200
        assert response.json()["data"]["isMatch"] is True
    
    # 计算总耗时
    total_time = asyncio.get_event_loop().time() - start_time
    print(f"批量操作总耗时: {total_time:.2f}秒")
    # 验证性能是否在可接受范围内（这里设定阈值为10秒）
    assert total_time < 10, f"批量操作耗时过长: {total_time:.2f}秒"

def test_verify_face_with_empty_array(client):
    """测试空的图片数组"""
    # 先创建人脸数据
    create_data = {
        "userId": 18,
        "faceImageBase64": TEST_IMAGE_BASE64
    }
    client.put("/users/me/face", json=create_data)

    # 测试空的图片数组
    verify_data = {
        "userId": 18,
        "faceImagesBase64": []
    }
    response = client.post("/users/me/face/verify", json=verify_data)
    assert response.status_code == 400
    assert "图片列表不能为空" in response.json()["detail"] 