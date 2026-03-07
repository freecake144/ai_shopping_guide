import copy
import random
import re
from typing import Tuple, List, Dict, Set

from utils.product_loader import (
    extract_product_core_info,
    load_products_from_csv,
    filter_products_by_involvement,
    get_matching_products,
    get_random_products,
)
from utils.deepseek_client import call_deepseek_with_products
from models.main import InteractionTurn, ExperimentSession


# =========================
# 1. 实验条件
# =========================
def get_experiment_condition(group_id: str):
    """
    2×2 实验条件映射
    A: LOW adaptivity + LOW calibration
    B: LOW adaptivity + HIGH calibration
    C: HIGH adaptivity + LOW calibration
    D: HIGH adaptivity + HIGH calibration
    """
    condition_map = {
        'A': ('LOW', 'LOW'),
        'B': ('LOW', 'HIGH'),
        'C': ('HIGH', 'LOW'),
        'D': ('HIGH', 'HIGH')
    }
    return condition_map.get(group_id, ('HIGH', 'HIGH'))


def assign_group():
    """随机分配实验组"""
    return random.choice(['A', 'B', 'C', 'D'])


# =========================
# 2. 文本预处理 / 需求识别
# =========================
def _normalize_text(text: str) -> str:
    return (text or "").strip().lower()


def _extract_budget(text: str):
    """
    尝试从中文表达里提取预算上限
    例如：
    - 500以内
    - 1000以下
    - 不超过800
    - 预算1500
    - 1000左右
    """
    if not text:
        return None

    patterns = [
        r'(\d+)\s*元?\s*以内',
        r'(\d+)\s*元?\s*以下',
        r'不超过\s*(\d+)',
        r'预算\s*(\d+)',
        r'(\d+)\s*左右',
        r'小于\s*(\d+)',
        r'低于\s*(\d+)',
    ]

    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                continue

    return None


def _extract_headset_type(text: str):
    """
    从用户文本中提取耳机类型
    """
    if not text:
        return None

    mapping = {
        "头戴": "头戴式",
        "头戴式": "头戴式",
        "入耳": "入耳式",
        "入耳式": "入耳式",
        "半入耳": "半入耳式",
        "半入耳式": "半入耳式",
    }

    for k, v in mapping.items():
        if k in text:
            return v
    return None


def _extract_core_function(text: str):
    """
    从用户文本提取核心功能
    这里尽量和CSV里的 core_function 值保持一致
    """
    if not text:
        return None

    candidates = [
        ("降噪", "降噪"),
        ("无线", "无线蓝牙"),
        ("蓝牙", "无线蓝牙"),
        ("续航", "长续航"),
        ("游戏", "游戏低延迟"),
        ("低延迟", "游戏低延迟"),
        ("音质", "高音质"),
        ("通话", "高清通话"),
        ("运动", "运动防水"),
        ("防水", "运动防水"),
    ]

    for keyword, normalized in candidates:
        if keyword in text:
            return normalized
    return None


def _extract_brand(text: str):
    """
    简单品牌识别，可按你的商品库继续补
    """
    if not text:
        return None

    brands = [
        "索尼", "sony",
        "苹果", "apple",
        "华为", "huawei",
        "小米", "xiaomi",
        "漫步者", "edifier",
        "bose", "博士",
        "jbl",
        "beats",
        "森海塞尔", "sennheiser",
        "oppo",
        "vivo",
    ]

    for brand in brands:
        if brand.lower() in text:
            # 返回原中文/常用形式
            brand_map = {
                "sony": "索尼",
                "apple": "苹果",
                "huawei": "华为",
                "xiaomi": "小米",
                "edifier": "漫步者",
                "博士": "Bose",
                "bose": "Bose",
                "jbl": "JBL",
                "beats": "Beats",
                "sennheiser": "森海塞尔",
                "oppo": "OPPO",
                "vivo": "vivo",
            }
            return brand_map.get(brand.lower(), brand)
    return None


def _detect_user_intent(user_msg: str) -> str:
    """
    粗粒度意图识别：
    - price_sensitive
    - comparison
    - exploration
    - recommendation
    """
    text = _normalize_text(user_msg)

    if any(k in text for k in ["对比", "比较", "区别", "哪个好", "哪款好", "怎么选"]):
        return "comparison"

    if any(k in text for k in ["便宜", "预算", "以内", "以下", "不超过", "性价比"]):
        return "price_sensitive"

    if any(k in text for k in ["随便看看", "都有哪些", "有什么", "推荐几款", "先看看"]):
        return "exploration"

    return "recommendation"


def _build_intent_details(user_msg: str) -> Dict:
    """
    提取结构化需求信息，供 HIGH calibration 使用
    """
    text = _normalize_text(user_msg)

    details = {}

    budget = _extract_budget(text)
    if budget is not None:
        details["max_price"] = budget

    headset_type = _extract_headset_type(text)
    if headset_type:
        details["headset_type"] = headset_type

    core_function = _extract_core_function(text)
    if core_function:
        details["core_function"] = core_function

    brand = _extract_brand(text)
    if brand:
        details["brand"] = brand

    return details


# =========================
# 3. 适应性操控（硬约束）
# =========================
def _need_clarification(user_msg: str, current_turn: int, intent_details: Dict) -> bool:
    """
    HIGH adaptivity 组在信息不足时必须追问
    判断标准尽量简单稳定：
    - 第一轮尤其严格
    - 文本很短
    - 几乎没有结构化需求
    """
    text = _normalize_text(user_msg)

    vague_patterns = [
        "想买耳机",
        "推荐耳机",
        "推荐一下",
        "耳机推荐",
        "买个耳机",
        "看看耳机",
        "有什么耳机",
        "推荐几款耳机",
    ]

    if len(text) <= 6:
        return True

    if current_turn == 1 and len(intent_details) == 0:
        return True

    if any(p in text for p in vague_patterns) and len(intent_details) <= 1:
        return True

    return False


def _build_clarifying_question(current_turn: int, intent_details: Dict) -> str:
    """
    统一追问模板，让 HIGH adaptivity 更稳定
    """
    if current_turn <= 1:
        return (
            "可以呀，我先帮你缩小一下范围。你主要是通勤、运动、游戏还是办公用呢？"
            "另外预算大概在什么区间，你更在意降噪、音质还是续航？"
        )

    if "max_price" not in intent_details:
        return "我再确认一下，你的预算大概是多少呢？这样我能帮你筛得更准一点。"

    if "headset_type" not in intent_details:
        return "我再确认一下，你更偏向头戴式、入耳式，还是半入耳式呢？"

    if "core_function" not in intent_details:
        return "明白了，我再确认一个重点：你更在意降噪、音质、续航、佩戴舒适度，还是通话效果呢？"

    return "我再确认一下，你还有特别在意的品牌或使用场景吗？这样我可以帮你筛得更贴合。"


# =========================
# 4. 历史记录辅助
# =========================
def _get_session_involvement(session_uuid: str) -> str:
    exp_session = ExperimentSession.query.filter_by(session_uuid=session_uuid).first()
    if not exp_session:
        return "high"
    # 兼容大小写
    return (exp_session.assigned_involvement or "high").lower()


def _get_history_ai_turns(session_uuid: str):
    return InteractionTurn.query.filter_by(
        session_uuid=session_uuid,
        sender='ai'
    ).order_by(InteractionTurn.turn_index.asc()).all()


def _get_history_product_ids(session_uuid: str) -> Set[str]:
    history_turns = _get_history_ai_turns(session_uuid)
    history_ids = set()

    for turn in history_turns:
        if turn.recommended_products:
            for p in turn.recommended_products:
                pid = p.get("product_id")
                if pid:
                    history_ids.add(pid)
    return history_ids


def _get_recent_history_products(session_uuid: str, max_n: int = 3) -> List[Dict]:
    """
    取最近几款历史推荐商品，必要时用于对比上下文
    """
    history_turns = _get_history_ai_turns(session_uuid)
    collected = []

    for turn in reversed(history_turns):
        if turn.recommended_products:
            for p in turn.recommended_products:
                if p not in collected:
                    collected.append(p)
                if len(collected) >= max_n:
                    return collected[:max_n]
    return collected[:max_n]


# =========================
# 5. 商品筛选（校准操控）
# =========================
def _safe_filter_products_by_ids(products: List[Dict], allowed_ids: Set[str]) -> List[Dict]:
    if not allowed_ids:
        return products
    return [p for p in products if p.get("product_id") in allowed_ids]


def _exclude_history_products(products: List[Dict], history_ids: Set[str]) -> List[Dict]:
    filtered = [p for p in products if p.get("product_id") not in history_ids]
    return filtered if filtered else products


def _high_calibration_select(
    base_products: List[Dict],
    user_intent: str,
    intent_details: Dict,
    top_n: int = 5
) -> List[Dict]:
    """
    HIGH calibration：按需求匹配
    product_loader.get_matching_products 会重新从全量CSV加载，
    所以这里要把结果再限制回当前涉入度商品池
    """
    matched = get_matching_products(user_intent, intent_details, top_n=top_n * 2)

    base_ids = {p.get("product_id") for p in base_products if p.get("product_id")}
    matched = _safe_filter_products_by_ids(matched, base_ids)

    if matched:
        return matched[:top_n]

    # 兜底：如果没匹配上，退化成当前涉入度池中的随机/前几项
    fallback = copy.deepcopy(base_products)
    random.shuffle(fallback)
    return fallback[:top_n]


def _low_calibration_select(
    base_products: List[Dict],
    top_n: int = 5
) -> List[Dict]:
    """
    LOW calibration：弱匹配/随机推荐
    """
    random_products = get_random_products(top_n=top_n * 2)
    base_ids = {p.get("product_id") for p in base_products if p.get("product_id")}
    random_products = _safe_filter_products_by_ids(random_products, base_ids)

    if len(random_products) >= top_n:
        return random_products[:top_n]

    # 如果随机结果不足，再从当前池补齐
    fallback_pool = copy.deepcopy(base_products)
    random.shuffle(fallback_pool)

    seen = {p.get("product_id") for p in random_products if p.get("product_id")}
    for p in fallback_pool:
        pid = p.get("product_id")
        if pid not in seen:
            random_products.append(p)
            seen.add(pid)
        if len(random_products) >= top_n:
            break

    return random_products[:top_n]


def _select_products_by_calibration(
    all_products: List[Dict],
    user_msg: str,
    calib_level: str,
    history_product_ids: Set[str],
    top_n: int = 5
) -> List[Dict]:
    user_intent = _detect_user_intent(user_msg)
    intent_details = _build_intent_details(user_msg)

    # 尽量避免一直重复推荐
    candidate_products = _exclude_history_products(all_products, history_product_ids)

    if calib_level == "HIGH":
        selected = _high_calibration_select(
            base_products=candidate_products,
            user_intent=user_intent,
            intent_details=intent_details,
            top_n=top_n
        )
    else:
        selected = _low_calibration_select(
            base_products=candidate_products,
            top_n=top_n
        )

    return selected[:top_n]


# =========================
# 6. 主函数
# =========================
def get_ai_response(
    user_msg: str,
    group_id: str,
    current_turn: int,
    assigned_adaptivity: str,
    assigned_calibration: str,
    session_uuid: str,
    previous_recommended_products: list = None
) -> Tuple[str, str, str, List[Dict]]:
    """
    返回:
    (
        ai_text,
        adapt_level,
        calib_level,
        core_products
    )
    """
    previous_recommended_products = previous_recommended_products or []

    adapt_level = (assigned_adaptivity or "HIGH").upper()
    calib_level = (assigned_calibration or "HIGH").upper()

    # 1) 读取当前session的涉入度，并过滤商品池
    all_products = load_products_from_csv()
    involvement = _get_session_involvement(session_uuid)
    all_products = filter_products_by_involvement(all_products, involvement)

    # 2) 提前识别用户需求
    user_intent = _detect_user_intent(user_msg)
    intent_details = _build_intent_details(user_msg)

    # 3) HIGH adaptivity：信息不足时，直接追问，不推荐商品
    if adapt_level == "HIGH" and _need_clarification(user_msg, current_turn, intent_details):
        ai_text = _build_clarifying_question(current_turn, intent_details)
        return ai_text, adapt_level, calib_level, []

    # 4) 获取历史推荐商品ID，避免重复
    history_product_ids = _get_history_product_ids(session_uuid)

    # 5) 根据校准水平做硬操控选商品
    selected_products = _select_products_by_calibration(
        all_products=all_products,
        user_msg=user_msg,
        calib_level=calib_level,
        history_product_ids=history_product_ids,
        top_n=5
    )

    # 6) 对 comparison 类意图，允许补一点历史商品做上下文
    # 但不要把全量历史商品都混进去
    if user_intent == "comparison":
        recent_history = _get_recent_history_products(session_uuid, max_n=2)
        merged = []
        seen = set()

        for p in recent_history + selected_products:
            pid = p.get("product_id")
            if pid and pid not in seen:
                merged.append(p)
                seen.add(pid)

        selected_products = merged[:5]

    # 7) 只把“最终筛出的少量商品”交给模型
    ai_text = call_deepseek_with_products(
        user_msg=user_msg,
        user_intent=user_intent,
        recommended_products=selected_products,
        adapt_level=adapt_level,
        calib_level=calib_level
    )

    # 8) 提取核心字段，供前端展示和数据库存储
    core_products = extract_product_core_info(selected_products[:5])

    return ai_text, adapt_level, calib_level, core_products



