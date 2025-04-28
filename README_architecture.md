# Face Recognition API

基于 FastAPI 和 ArcFace 的人脸识别系统，提供人脸注册、验证和考勤管理功能。

## 项目架构

### 1. 整体架构

项目采用三层架构设计：

-   API 层：处理 HTTP 请求和响应
-   服务层：实现业务逻辑
-   存储层：处理数据持久化

```
arcface-pytorch-main/
├── openapi_server/
│   ├── api/                    # API路由层
│   │   ├── face_api.py        # 人脸相关API
│   │   ├── user_api.py        # 用户相关API
│   │   └── attendance_api.py  # 考勤相关API
│   ├── controllers/           # 控制器层
│   │   ├── face_controller.py # 人脸控制器
│   │   ├── user_controller.py # 用户控制器
│   │   └── attendance_controller.py # 考勤控制器
│   ├── services/             # 服务层
│   │   ├── face_service.py   # 人脸服务
│   │   ├── user_service.py   # 用户服务
│   │   └── attendance_service.py # 考勤服务
│   ├── repositories/         # 存储层
│   │   ├── face_repository.py # 人脸数据仓库
│   │   ├── user_repository.py # 用户数据仓库
│   │   ├── attendance_repository.py # 考勤数据仓库
│   │   └── base_repository.py # 基础仓库类
│   ├── models/              # 数据模型
│   │   ├── face.py         # 人脸模型
│   │   ├── user.py         # 用户模型
│   │   └── attendance.py   # 考勤模型
│   ├── schemas/            # 数据模式
│   │   ├── request.py      # 请求模式
│   │   └── response.py     # 响应模式
│   ├── database.py         # 数据库配置
│   └── config.py           # 应用配置
├── tests/                  # 测试目录
│   ├── test_face_controller.py
│   ├── test_user_controller.py
│   └── test_attendance_controller.py
├── requirements.txt        # 项目依赖
└── README.md              # 项目文档
```

### 2. 核心组件

#### 2.1 API 层

-   使用 FastAPI 框架处理 HTTP 请求
-   提供 RESTful API 接口
-   实现请求参数验证和响应格式化

#### 2.2 服务层

-   实现核心业务逻辑
-   处理人脸检测、对齐和特征提取
-   管理用户认证和授权
-   处理考勤记录和统计

#### 2.3 存储层

-   使用 SQLAlchemy ORM 进行数据库操作
-   实现数据持久化
-   提供数据访问接口

### 3. 主要功能

#### 3.1 人脸管理

-   人脸注册：支持单张人脸图片注册
-   人脸验证：支持多张人脸图片验证
-   人脸更新：支持更新已注册的人脸数据

#### 3.2 用户管理

-   用户注册：创建新用户
-   用户登录：基于 JWT 的身份验证
-   用户信息管理：查询和更新用户信息

#### 3.3 考勤管理

-   考勤记录：记录用户考勤信息
-   考勤统计：统计用户考勤数据
-   考勤查询：查询历史考勤记录

### 4. 技术栈

-   后端框架：FastAPI
-   数据库：SQLite
-   ORM：SQLAlchemy
-   人脸识别：ArcFace
-   图像处理：OpenCV, PIL
-   测试框架：pytest

### 5. 配置说明

项目配置通过`config.py`管理，主要配置项包括：

-   数据库连接
-   API 设置
-   安全设置
-   应用设置

### 6. 开发环境设置

1. 克隆项目

```bash
git clone [repository_url]
cd arcface-pytorch-main
```

2. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安装依赖

```bash
pip install -r requirements.txt
```

4. 运行测试

```bash
pytest tests/
```

5. 启动服务

```bash
uvicorn main:app --reload
```

### 7. API 文档

启动服务后，可以通过以下地址访问 API 文档：

-   Swagger UI: http://localhost:8000/docs
-   ReDoc: http://localhost:8000/redoc

### 8. 安全考虑

-   使用 JWT 进行身份验证
-   密码使用 bcrypt 加密存储
-   敏感配置通过环境变量管理
-   实现请求速率限制
-   输入验证和清理

### 9. 性能优化

-   使用连接池管理数据库连接
-   实现缓存机制
-   优化人脸检测和特征提取算法
-   批量处理支持

### 10. 部署说明

1. 配置环境变量

```bash
export DATABASE_URL="sqlite:///./sql_app.db"
export SECRET_KEY="your-secret-key"
```

2. 启动服务

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 11. 维护和更新

-   定期更新依赖包
-   监控系统性能
-   备份数据库
-   更新安全补丁

### 12. 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

### 13. 许可证

[许可证类型]

### 14. 联系方式

[项目维护者信息]
