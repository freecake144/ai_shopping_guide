import json
import random

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from models.main import db,User,InteractionTurn,ExperimentSession
from ai.logic import assign_group, get_ai_response, get_experiment_condition
import uuid
import os
from datetime import datetime
from flask_migrate import Migrate
from utils.preference_analyzer import PreferenceAnalyzer

app = Flask(
    __name__,
    static_folder='static',  # 你的 static 文件夹在项目根目录下
    static_url_path='/static'  # 前端访问静态文件的前缀（必须和前端src一致）
)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# 配置数据库路径
DB_DIR = os.environ.get('RAILWAY_VOLUME_MOUNT_PATH', os.path.join(BASE_DIR, 'data'))
os.makedirs(DB_DIR, exist_ok=True)  # 确保目录存在
DB_PATH = os.path.join(DB_DIR, 'experiment.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
app.secret_key = 'thesis_secret_key'  # 用于加密session

db.init_app(app)
migrate = Migrate(app, db)

# 初始化分析器
analyzer = PreferenceAnalyzer()

# 初始化数据库（第一次运行时自动创建文件）
with app.app_context():
    # 确保 data 目录存在
    if not os.path.exists('data'):
        os.makedirs('data')
    db.create_all()


@app.route('/')
def index():
    """实验着陆页：分配ID和分组"""
    # 确保user存在
    if 'user_uuid' not in session:
        user_uuid = str(uuid.uuid4())
        session['user_uuid'] = user_uuid

        # 创建用户并写入数据库
        user = User(
            user_uuid=user_uuid,
            created_at=datetime.utcnow()
        )
        db.session.add(user)
        db.session.commit()

    user_uuid = session['user_uuid']

    # 确保Session (实验会话)存在
    if 'session_id' not in session:
        session_uuid = str(uuid.uuid4())
        group_id = assign_group()  # A / B / C / D随机分组

        session['session_uuid'] = session_uuid
        session['group_id'] = group_id

        # 记录实验会话元数据
        user_uuid = session['user_uuid']
        # 获取当前组别的设定用于记录
        from ai.logic import get_ai_response
        adapt, calib = get_experiment_condition(group_id)

        exp_session = ExperimentSession(
            session_uuid=session_uuid,
            user_uuid=user_uuid,
            group_id=group_id,
            assigned_adaptivity = adapt,
            assigned_calibration = calib,
            start_time=datetime.utcnow()
        )
        db.session.add(exp_session)

        # 更新user表的分组信息（方便查询）
        current_user = User.query.filter_by(user_uuid=user_uuid).first()
        if current_user:
            current_user.group_id = group_id

        db.session.commit()

    return render_template('index.html')  # 欢迎页


@app.route('/register', methods=['GET', 'POST'])
def register():
    """注册页：可选邮箱收集"""
    if request.method == 'POST':
        email = request.form.get('email', '').strip()

        # 如果有邮箱，存入 User 模型（假设你的 User 模型已有 email 字段）
        # 如果没有，先加字段（见下文提示）
        if 'user_uuid' in session:
            user = User.query.filter_by(user_uuid=session['user_uuid']).first()
            if user:
                user.email = email or None  # 空字符串转 None
                db.session.commit()

        # 提交后直接跳转聊天页
        return redirect(url_for('chat_page'))

    return render_template('register.html')


@app.route('/chat')
def chat_page():
    """聊天主界面"""
    return render_template('chat.html')


@app.route('/api/send', methods=['POST'])
def api_send():
    # A. 获取前端传来的数据
    data = request.json
    user_msg = data.get('msg')

    # B. 获取 Session 中的实验状态
    user_uuid = session.get('user_uuid')
    session_uuid = session.get('session_uuid')
    group_id = session.get('group_id')

    # 安全检查：如果 Session 过期了，报错
    if not all([user_uuid, session_uuid, group_id]):
        return jsonify({'error': 'Session expired, please refresh'}), 400

    # C. 计算当前是第几轮 (Turn Index)
    # 逻辑：总记录数除以2 + 1。例如：0条记录->第1轮；2条记录->第2轮
    total_msgs = InteractionTurn.query.filter_by(session_uuid=session_uuid).count()
    current_turn_index = (total_msgs // 2) + 1

    # ===============================================================
    # D. [核心步骤] 计算动态偏好指标 (Thesis Metric Calculation)
    # ===============================================================

    # 1. 计算当前文本的特征向量
    current_vector = analyzer.compute_vector(user_msg)
    focus_dim = analyzer.identify_focus(user_msg)
    drift_score = 0.0  # 默认漂移为0

    # 2. 获取上一轮用户发言 (用于计算对比)
    # 注意：只找 sender='user' 的最近一条
    last_user_turn = InteractionTurn.query.filter_by(
        session_uuid=session_uuid,
        sender='user'
    ).order_by(InteractionTurn.turn_index.desc()).first()

    # 3. 如果有上一轮，计算 Drift (欧氏距离)
    if last_user_turn and last_user_turn.preference_vector:
        last_vector = last_user_turn.preference_vector
        # SQLAlchemy JSON类型通常自动转Dict，但为了保险起见处理一下
        if isinstance(last_vector, str):
            try:
                last_vector = json.loads(last_vector)
            except:
                last_vector = {}

        drift_score = analyzer.calculate_drift(current_vector, last_vector)

    # ===============================================================
    # E. 存储 USER 发言 (包含偏好数据)
    # ===============================================================
    user_turn = InteractionTurn(
        session_uuid=session_uuid,
        user_uuid=user_uuid,
        sender='user',
        content=user_msg,
        turn_index=current_turn_index,

        # 存入你的论文核心指标
        preference_vector=current_vector,
        preference_drift=drift_score,
        focus_dimension=focus_dim,

        # AI 的字段留空
        ai_adaptability_level=None,
        ai_calibration_level=None
    )
    db.session.add(user_turn)
    db.session.commit()  # 立即提交，防止后续出错导致用户输入丢失

    # ===============================================================
    # F. 调用 AI 逻辑 (Experiment Manipulation)
    # ===============================================================
    # 获取上一轮AI推荐过的商品
    last_ai_turn = InteractionTurn.query.filter_by(
        session_uuid=session_uuid,
        sender='ai'
    ).order_by(InteractionTurn.turn_index.desc()).first()

    previous_products = []
    if last_ai_turn and last_ai_turn.recommended_products:
        # 如果上一轮有推荐商品，提取出来
        # 此时取出来的是 list[dict] 格式
        previous_products = last_ai_turn.recommended_products

    # 将 previous_products 传入 get_ai_response
    assigned_adapt, assigned_calib = get_experiment_condition(group_id)
    ai_text, adapt_level, calib_level, recommended_products = get_ai_response(
        user_msg=user_msg,
        group_id=group_id,
        current_turn=current_turn_index,
        assigned_adaptivity=assigned_adapt,
        assigned_calibration=assigned_calib,
        previous_recommended_products=previous_products
    )

    # ===============================================================
    # G. 存储 AI 回复
    # ===============================================================
    ai_turn = InteractionTurn(
        session_uuid=session_uuid,
        user_uuid=user_uuid,
        sender='ai',
        content=ai_text,
        turn_index=current_turn_index,
        recommended_products=recommended_products,

        # 记录 AI 当时的实验状态 (方便做 ANOVA 分析)
        ai_adaptability_level=adapt_level,
        ai_calibration_level=calib_level,

        # AI 没有偏好向量，留空
        preference_vector=None,
        preference_drift=None,
        focus_dimension=None
    )
    db.session.add(ai_turn)
    db.session.commit()

    # H. 构造返回前端的数据
    frontend_products = []
    if recommended_products:
        for p in recommended_products:
            pid = p.get('product_id')

            frontend_products.append({
                'product_id': pid,
                'product_name': p.get('product_name'),
                'price': p.get('price'),
                'headset_type': p.get('category'),
                'core_function': p.get('core_function') or '',
                'image_url': f'/static/images/{pid}.jpg?v={datetime.utcnow().timestamp()}'
            })

    # J. 返回结果给前端
    return jsonify({'response': ai_text, 'products':frontend_products})

@app.route('/end')
def end_experiment():
    session_uuid = session.get('session_uuid')

    if session_uuid:
        exp_session = ExperimentSession.query.filter_by(
            session_uuid=session_uuid
        ).first()

        if exp_session:
            exp_session.end_time = datetime.utcnow()
            db.session.commit()

    # 清理浏览器 session（防止重复）
    session.clear()
    return render_template('end.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)





