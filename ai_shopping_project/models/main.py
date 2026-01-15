from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sqlalchemy.dialects.sqlite import JSON

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    user_uuid = db.Column(db.String(64), unique=True)  # 用户的唯一标识
    group_id = db.Column(db.String(10))  # 实验分组: A, B, C, D
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class InteractionTurn(db.Model):
    __tablename__ = 'interaction_turns'
    id = db.Column(db.Integer, primary_key=True)
    session_uuid = db.Column(db.String(64), db.ForeignKey('experiment_session.session_uuid'))
    user_uuid = db.Column(db.String(64), db.ForeignKey('users.user_uuid'))

    sender = db.Column(db.String(10))  # 'user' 或 'ai'
    content = db.Column(db.Text)  # 聊天内容
    # 动态偏好量化指标
    # 1. 偏好向量 (存 JSON, 例如 {"price": 0.5, "specificity": 0.8})
    preference_vector = db.Column(db.JSON)
    # 2. 演化强度 (Drift): 这一轮与上一轮偏好的欧氏距离，代表变化的剧烈程度
    preference_drift = db.Column(db.Float, default=0.0)
    # 3. 当前聚焦维度 (用于定性分析，如 "price", "feature", "brand")
    focus_dimension = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    # --- 论文核心数据 ---
    turn_index = db.Column(db.Integer)  # 第几轮对话

    # 记录当时AI的状态（用于后续归因分析）
    ai_adaptability_level = db.Column(db.String(10))  # HIGH / LOW
    ai_calibration_level = db.Column(db.String(10))  # HIGH / LOW

    # 推荐商品字段
    recommended_products = db.Column(JSON, nullable=True)  # 存储推荐商品的JSON数据（AI回复时才有值，用户消息为None）

class ExperimentSession(db.Model):
    __tablename__ = 'experiment_session'
    id = db.Column(db.Integer, primary_key=True)

    session_uuid = db.Column(db.String(64), unique=True)
    user_uuid = db.Column(db.String(64), db.ForeignKey('users.user_uuid'))

    group_id = db.Column(db.String(10)) # A/B/C/D
    assigned_adaptivity = db.Column(db.String(10)) # HIGH/LOW
    assigned_calibration = db.Column(db.String(10)) # HIGH/LOW

    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)

class Product(db.Model):
    __tablename__ = 'products'  # 数据库表名：products
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.String(64), unique=True, nullable=False)  # 商品唯一标识（EAR001）
    product_name = db.Column(db.String(255), nullable=False)  # 商品名称
    price = db.Column(db.Float, nullable=False)  # 价格
    price_band = db.Column(db.String(10))  # 价格带（低/中/高）
    headset_type = db.Column(db.String(20))  # 耳机类型（头戴式/入耳式）
    core_function = db.Column(db.String(255))  # 核心功能（降噪,无线蓝牙）
    brand = db.Column(db.String(20))  # 品牌
    battery_life = db.Column(db.Integer)  # 续航时长
    sales_volume = db.Column(db.String(10))  # 销量（5000+）
    scenario = db.Column(db.String(20))  # 使用场景（通勤/游戏）
    created_at = db.Column(db.DateTime, default=datetime.utcnow)  # 入库时间


