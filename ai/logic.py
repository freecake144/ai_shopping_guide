import copy
import random
import re
from typing import Tuple, List, Dict, Set, Any

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
        "A": ("LOW", "LOW"),
        "B": ("LOW", "HIGH"),
        "C": ("HIGH", "LOW"),
        "D": ("HIGH", "HIGH"),
    }
    return condition_map.get(group_id, ("HIGH", "HIGH"))


def assign_group():
    return random.choice(["A", "B", "C", "D"])


# =========================
# 2. 基础工具
# =========================
def _normalize_text(text: str) -> str:
    return (text or "").strip().lower()


def _safe_json_dict(value: Any) -> Dict:
    return value if isinstance(value, dict) else {}


def _safe_json_list(value: Any) -> List:
    return value if isinstance(value, list) else []


def _dedup_products(products: List[Dict], max_n: int = None) -> List[Dict]:
    seen = set()
    result = []

    for p in products:
        if not isinstance(p, dict):
            continue
        pid = p.get("product_id")
        if not pid or pid in seen:
            continue
        seen.add(pid)
        result.append(p)
        if max_n and len(result) >= max_n:
            break

    return result


def _count_filled_slots(profile: Dict) -> int:
    count = 0
    if profile.get("max_price") is not None:
        count += 1
    if profile.get("headset_type"):
        count += 1
    if profile.get("brand"):
        count += 1
    if profile.get("core_functions"):
        count += 1
    if profile.get("scenarios"):
        count += 1
    return count


# =========================
# 3. 从文本抽取结构化需求
# =========================
def _extract_budget(text: str):
    if not text:
        return None

    patterns = [
        r"预算\s*(\d+)",
        r"(\d+)\s*元?\s*以内",
        r"(\d+)\s*元?\s*以下",
        r"(\d+)\s*块?\s*以内",
        r"不超过\s*(\d+)",
        r"低于\s*(\d+)",
        r"小于\s*(\d+)",
        r"控制在\s*(\d+)",
        r"(\d+)\s*左右",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None


def _extract_headset_type(text: str):
    if not text:
        return None

    mapping = {
        "头戴式": "头戴式",
        "头戴": "头戴式",
        "入耳式": "入耳式",
        "入耳": "入耳式",
        "半入耳式": "半入耳式",
        "半入耳": "半入耳式",
    }

    for k, v in mapping.items():
        if k in text:
            return v
    return None


def _extract_core_functions(text: str) -> List[str]:
    if not text:
        return []

    mapping = [
        ("主动降噪", "降噪"),
        ("降噪", "降噪"),
        ("无线", "无线蓝牙"),
        ("蓝牙", "无线蓝牙"),
        ("续航", "长续航"),
        ("长续航", "长续航"),
        ("游戏", "游戏低延迟"),
        ("低延迟", "游戏低延迟"),
        ("音质", "高音质"),
        ("重低音", "重低音"),
        ("通话", "高清通话"),
        ("麦克风", "高清通话"),
        ("运动", "运动防水"),
        ("防水", "运动防水"),
        ("舒适", "佩戴舒适"),
        ("佩戴", "佩戴舒适"),
    ]

    found = []
    for keyword, normalized in mapping:
        if keyword in text and normalized not in found:
            found.append(normalized)
    return found


def _extract_brand(text: str):
    if not text:
        return None

    brand_map = {
        "sony": "索尼",
        "索尼": "索尼",
        "apple": "苹果",
        "苹果": "苹果",
        "huawei": "华为",
        "华为": "华为",
        "xiaomi": "小米",
        "小米": "小米",
        "edifier": "漫步者",
        "漫步者": "漫步者",
        "bose": "Bose",
        "博士": "Bose",
        "jbl": "JBL",
        "beats": "Beats",
        "sennheiser": "森海塞尔",
        "森海塞尔": "森海塞尔",
        "oppo": "OPPO",
        "vivo": "vivo",
    }

    text_lower = text.lower()
    for raw, normalized in brand_map.items():
        if raw in text_lower:
            return normalized
    return None


def _extract_scenarios(text: str) -> List[str]:
    if not text:
        return []

    mapping = [
        ("通勤", "通勤"),
        ("地铁", "通勤"),
        ("上班", "办公"),
        ("办公", "办公"),
        ("开会", "办公"),
        ("学习", "学习"),
        ("运动", "运动"),
        ("跑步", "运动"),
        ("健身", "运动"),
        ("游戏", "游戏"),
        ("手游", "游戏"),
        ("打游戏", "游戏"),
        ("睡觉", "助眠"),
        ("日常", "日常"),
        ("出差", "通勤"),
    ]

    found = []
    for keyword, normalized in mapping:
        if keyword in text and normalized not in found:
            found.append(normalized)
    return found


def _build_intent_details(user_msg: str) -> Dict:
    text = _normalize_text(user_msg)
    details = {}

    budget = _extract_budget(text)
    if budget is not None:
        details["max_price"] = budget

    headset_type = _extract_headset_type(text)
    if headset_type:
        details["headset_type"] = headset_type

    brand = _extract_brand(text)
    if brand:
        details["brand"] = brand

    core_functions = _extract_core_functions(text)
    if core_functions:
        details["core_function"] = core_functions[0]
        details["core_functions"] = core_functions

    scenarios = _extract_scenarios(text)
    if scenarios:
        details["scenarios"] = scenarios

    return details


def _detect_user_intent(user_msg: str) -> str:
    text = _normalize_text(user_msg)

    if any(k in text for k in ["对比", "比较", "区别", "哪个好", "哪款好", "怎么选"]):
        return "comparison"

    if any(k in text for k in ["便宜", "预算", "以内", "以下", "不超过", "性价比"]):
        return "price_sensitive"

    if any(k in text for k in ["随便看看", "都有哪些", "有什么", "推荐几款", "先看看"]):
        return "exploration"

    return "recommendation"


# =========================
# 4. 会话历史 / 记忆
# =========================
def _get_session_involvement(session_uuid: str) -> str:
    exp_session = ExperimentSession.query.filter_by(session_uuid=session_uuid).first()
    if not exp_session:
        return "high"
    return (exp_session.assigned_involvement or "high").lower()


def _get_history_ai_turns(session_uuid: str):
    return InteractionTurn.query.filter_by(
        session_uuid=session_uuid,
        sender="ai"
    ).order_by(InteractionTurn.turn_index.asc()).all()


def _get_history_user_turns(session_uuid: str):
    return InteractionTurn.query.filter_by(
        session_uuid=session_uuid,
        sender="user"
    ).order_by(InteractionTurn.turn_index.asc()).all()


def _get_history_product_ids(session_uuid: str) -> Set[str]:
    history_turns = _get_history_ai_turns(session_uuid)
    history_ids = set()

    for turn in history_turns:
        products = _safe_json_list(turn.recommended_products)
        for p in products:
            pid = p.get("product_id")
            if pid:
                history_ids.add(pid)

    return history_ids


def _get_recent_history_products(session_uuid: str, max_n: int = 3) -> List[Dict]:
    history_turns = _get_history_ai_turns(session_uuid)
    collected = []

    for turn in reversed(history_turns):
        products = _safe_json_list(turn.recommended_products)
        for p in products:
            collected.append(p)

    return _dedup_products(collected, max_n=max_n)


def _count_stable_signals(session_uuid: str) -> Dict:
    """
    统计跨轮重复出现的偏好信号，避免一次提及就被当成稳定需求
    """
    history_user_turns = _get_history_user_turns(session_uuid)

    counts = {
        "budget": 0,
        "headset_type": 0,
        "brand": 0,
        "core_function": 0,
        "scenario": 0,
    }

    for turn in history_user_turns:
        details = _build_intent_details(_normalize_text(turn.content or ""))

        if details.get("max_price") is not None:
            counts["budget"] += 1
        if details.get("headset_type"):
            counts["headset_type"] += 1
        if details.get("brand"):
            counts["brand"] += 1
        if details.get("core_functions"):
            counts["core_function"] += 1
        if details.get("scenarios"):
            counts["scenario"] += 1

    return counts


def _build_user_memory_profile(session_uuid: str) -> Dict:
    history_user_turns = _get_history_user_turns(session_uuid)

    memory = {
        "max_price": None,
        "headset_type": None,
        "brand": None,
        "core_functions": [],
        "scenarios": [],
        "known_slots": [],
        "summary": "暂无明确历史需求",
    }

    for turn in history_user_turns:
        text = _normalize_text(turn.content or "")
        current_details = _build_intent_details(text)

        if current_details.get("max_price") is not None:
            memory["max_price"] = current_details["max_price"]

        if current_details.get("headset_type"):
            memory["headset_type"] = current_details["headset_type"]

        if current_details.get("brand"):
            memory["brand"] = current_details["brand"]

        for func in current_details.get("core_functions", []):
            if func not in memory["core_functions"]:
                memory["core_functions"].append(func)

        for s in current_details.get("scenarios", []):
            if s not in memory["scenarios"]:
                memory["scenarios"].append(s)

        vec = _safe_json_dict(turn.preference_vector)
        attrs = _safe_json_dict(vec.get("preferred_attributes"))

        for func in _safe_json_list(attrs.get("core_function")):
            if func not in memory["core_functions"]:
                memory["core_functions"].append(func)

        for s in _safe_json_list(attrs.get("scenario")):
            if s not in memory["scenarios"]:
                memory["scenarios"].append(s)

        headset_types = _safe_json_list(attrs.get("headset_type"))
        if not memory["headset_type"] and headset_types:
            memory["headset_type"] = headset_types[0]

        brands = _safe_json_list(attrs.get("brand"))
        if not memory["brand"] and brands:
            memory["brand"] = brands[0]

    if memory["max_price"] is not None:
        memory["known_slots"].append("budget")
    if memory["headset_type"]:
        memory["known_slots"].append("headset_type")
    if memory["brand"]:
        memory["known_slots"].append("brand")
    if memory["core_functions"]:
        memory["known_slots"].append("core_function")
    if memory["scenarios"]:
        memory["known_slots"].append("scenario")

    summary_parts = []
    if memory["max_price"] is not None:
        summary_parts.append(f"预算约{memory['max_price']}元")
    if memory["headset_type"]:
        summary_parts.append(f"偏好{memory['headset_type']}")
    if memory["brand"]:
        summary_parts.append(f"偏好品牌{memory['brand']}")
    if memory["core_functions"]:
        summary_parts.append(f"关注功能：{'、'.join(memory['core_functions'])}")
    if memory["scenarios"]:
        summary_parts.append(f"使用场景：{'、'.join(memory['scenarios'])}")

    if summary_parts:
        memory["summary"] = "；".join(summary_parts)

    return memory


def _merge_memory_with_current(memory: Dict, current_details: Dict) -> Dict:
    merged = {
        "max_price": current_details.get("max_price")
        if current_details.get("max_price") is not None
        else memory.get("max_price"),

        "headset_type": current_details.get("headset_type") or memory.get("headset_type"),
        "brand": current_details.get("brand") or memory.get("brand"),
        "core_functions": list(memory.get("core_functions", [])),
        "scenarios": list(memory.get("scenarios", [])),
    }

    for func in current_details.get("core_functions", []):
        if func not in merged["core_functions"]:
            merged["core_functions"].append(func)

    for s in current_details.get("scenarios", []):
        if s not in merged["scenarios"]:
            merged["scenarios"].append(s)

    merged["core_function"] = merged["core_functions"][0] if merged["core_functions"] else None

    known_slots = []
    if merged["max_price"] is not None:
        known_slots.append("budget")
    if merged["headset_type"]:
        known_slots.append("headset_type")
    if merged["brand"]:
        known_slots.append("brand")
    if merged["core_functions"]:
        known_slots.append("core_function")
    if merged["scenarios"]:
        known_slots.append("scenario")
    merged["known_slots"] = known_slots

    summary_parts = []
    if merged["max_price"] is not None:
        summary_parts.append(f"预算约{merged['max_price']}元")
    if merged["headset_type"]:
        summary_parts.append(f"偏好{merged['headset_type']}")
    if merged["brand"]:
        summary_parts.append(f"偏好品牌{merged['brand']}")
    if merged["core_functions"]:
        summary_parts.append(f"关注功能：{'、'.join(merged['core_functions'])}")
    if merged["scenarios"]:
        summary_parts.append(f"使用场景：{'、'.join(merged['scenarios'])}")
    merged["summary"] = "；".join(summary_parts) if summary_parts else "暂无明确需求"

    return merged


# =========================
# 5. 适应性操控：缺失追问 + 确认式追问
# =========================
def _need_clarification_from_memory(memory_profile: Dict, current_turn: int) -> bool:
    """
    只有明显信息不足时才追问
    降敏：第二轮后不轻易因为单一缺项就再次追问
    """
    missing = 0

    if memory_profile.get("max_price") is None:
        missing += 1
    if not memory_profile.get("headset_type"):
        missing += 1
    if not memory_profile.get("core_functions") and not memory_profile.get("scenarios"):
        missing += 1

    if current_turn <= 1:
        return missing >= 2

    return missing >= 3


def _build_targeted_clarifying_question(memory_profile: Dict) -> str:
    missing_budget = memory_profile.get("max_price") is None
    missing_type = not memory_profile.get("headset_type")
    missing_need = (not memory_profile.get("core_functions")) and (not memory_profile.get("scenarios"))

    if missing_budget and missing_type:
        return "我先确认两点，这样后面推荐会更贴合：你的预算大概是多少？另外你更偏向头戴式、入耳式还是半入耳式呢？"

    if missing_budget and missing_need:
        return "我再确认一下，你的预算大概是多少？另外你主要是通勤、运动、游戏还是办公使用，或者最在意降噪、音质、续航中的哪一项呢？"

    if missing_budget:
        return "我再确认一下你的预算大概是多少呢？这样我可以先帮你排除掉不合适的价格区间。"

    if missing_type:
        return "我再确认一下，你更偏向头戴式、入耳式还是半入耳式呢？这会直接影响后面推荐方向。"

    if missing_need:
        return "我再确认一下，你主要是什么场景用，或者最在意的是降噪、音质、续航里的哪一项呢？"

    return ""


def _need_confirmation_followup(memory_profile: Dict, current_turn: int, stable_counts: Dict) -> bool:
    """
    已经有初步需求，但还不够稳定时，在第2-3轮优先做确认式追问
    """
    if not (2 <= current_turn <= 3):
        return False

    detail_count = _count_filled_slots(memory_profile)

    stable_slot_count = sum(1 for v in stable_counts.values() if v >= 2)

    # 有初步轮廓，但稳定性还不足
    return 2 <= detail_count <= 4 and stable_slot_count < 2


def _build_confirmation_followup(memory_profile: Dict) -> str:
    if memory_profile.get("max_price") is not None and not memory_profile.get("headset_type"):
        return "你刚刚已经提到预算和使用方向了，我再确认一下：你更偏向头戴式、入耳式还是半入耳式呢？"

    if memory_profile.get("scenarios") and memory_profile.get("core_functions"):
        return "我大概明白你的方向了。我再确认一个取舍：在你的使用场景下，你会更优先考虑降噪/音质这些核心体验，还是更看重续航和佩戴舒适度呢？"

    if memory_profile.get("brand") is None:
        return "我再确认一下，你对品牌有没有明显偏好？比如更倾向索尼、Bose、苹果这类，还是更看重性价比？"

    return "我目前已经有一个初步判断了，不过为了推荐更贴合，我再确认一个点：你最不能妥协的那个条件是什么？"


# =========================
# 6. 停止指令
# =========================
def _is_explicit_finish_intent(user_msg: str) -> bool:
    text = _normalize_text(user_msg)
    finish_keywords = [
        "可以了", "就这样", "没问题了", "不用再推荐了", "不用推荐了",
        "我已经决定了", "决定好了", "就买这个", "买这个", "就这个",
        "可以结束了", "结束吧", "开始答题", "去答题", "填写问卷",
        "我要答题", "进入问卷"
    ]
    return any(k in text for k in finish_keywords)


def _is_need_clear_enough(memory_profile: Dict, current_turn: int, user_msg: str, stable_counts: Dict) -> bool:
    """
    降敏版停止条件：
    1. 用户明确表示结束/决定 -> 可以停
    2. 否则至少第3轮以后
    3. 且信息完整度更高 + 稳定信号足够
    """
    text = _normalize_text(user_msg)

    decision_keywords = [
        "就买", "下单", "决定", "买这个", "就这个",
        "直接买", "可以了", "够了", "不用再推荐"
    ]

    if any(k in text for k in decision_keywords):
        return True

    if current_turn < 3:
        return False

    detail_count = _count_filled_slots(memory_profile)
    stable_slot_count = sum(1 for v in stable_counts.values() if v >= 2)

    # 信息够丰富 + 至少有两个槽位在跨轮中重复出现，才认为需求相对稳定
    if detail_count >= 4 and stable_slot_count >= 2:
        return True

    return False


def _build_stop_message(memory_profile: Dict) -> str:
    summary_text = memory_profile.get("summary") or "你的需求方向"
    return (
        f"我目前已经初步了解你的需求方向：{summary_text}。"
        f"如果你觉得差不多了，可以结束本轮交互并开始答题；"
        f"如果你愿意，我也可以继续帮你细化比较。"
    )


# =========================
# 7. 校准操控：基于合并记忆选商品
# =========================
def _safe_filter_products_by_ids(products: List[Dict], allowed_ids: Set[str]) -> List[Dict]:
    if not allowed_ids:
        return products
    return [p for p in products if p.get("product_id") in allowed_ids]


def _exclude_history_products(products: List[Dict], history_ids: Set[str]) -> List[Dict]:
    filtered = [p for p in products if p.get("product_id") not in history_ids]
    return filtered if filtered else list(products)


def _high_calibration_select(
    base_products: List[Dict],
    user_intent: str,
    intent_details: Dict,
    top_n: int = 5
) -> List[Dict]:
    matched = get_matching_products(user_intent, intent_details, top_n=top_n * 2)

    base_ids = {p.get("product_id") for p in base_products if p.get("product_id")}
    matched = _safe_filter_products_by_ids(matched, base_ids)

    if matched:
        return matched[:top_n]

    fallback = copy.deepcopy(base_products)
    random.shuffle(fallback)
    return fallback[:top_n]


def _low_calibration_select(
    base_products: List[Dict],
    top_n: int = 5
) -> List[Dict]:
    random_products = get_random_products(top_n=top_n * 2)
    base_ids = {p.get("product_id") for p in base_products if p.get("product_id")}
    random_products = _safe_filter_products_by_ids(random_products, base_ids)

    if len(random_products) >= top_n:
        return random_products[:top_n]

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
    user_intent: str,
    merged_profile: Dict,
    calib_level: str,
    history_product_ids: Set[str],
    top_n: int = 5
) -> List[Dict]:
    candidate_products = _exclude_history_products(all_products, history_product_ids)

    intent_details = {}
    if merged_profile.get("max_price") is not None:
        intent_details["max_price"] = merged_profile["max_price"]
    if merged_profile.get("headset_type"):
        intent_details["headset_type"] = merged_profile["headset_type"]
    if merged_profile.get("brand"):
        intent_details["brand"] = merged_profile["brand"]
    if merged_profile.get("core_function"):
        intent_details["core_function"] = merged_profile["core_function"]

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

    return _dedup_products(selected, max_n=top_n)


# =========================
# 8. 主函数
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

    # 1) 读取当前session涉入度，并过滤商品池
    all_products = load_products_from_csv()
    involvement = _get_session_involvement(session_uuid)
    all_products = filter_products_by_involvement(all_products, involvement)

    # 2) 本轮意图 + 历史记忆 + 合并画像
    user_intent = _detect_user_intent(user_msg)
    current_details = _build_intent_details(user_msg)
    history_memory = _build_user_memory_profile(session_uuid)
    merged_profile = _merge_memory_with_current(history_memory, current_details)
    stable_counts = _count_stable_signals(session_uuid)

    # 3) 明确结束指令优先
    if _is_explicit_finish_intent(user_msg):
        ai_text = _build_stop_message(merged_profile)
        return ai_text, adapt_level, calib_level, []

    # 4) HIGH adaptivity：第2-3轮优先做确认式追问，避免过早收口
    if adapt_level == "HIGH" and _need_confirmation_followup(merged_profile, current_turn, stable_counts):
        ai_text = _build_confirmation_followup(merged_profile)
        return ai_text, adapt_level, calib_level, []

    # 5) HIGH adaptivity：只有明显信息不足时才追问缺失项
    if adapt_level == "HIGH" and _need_clarification_from_memory(merged_profile, current_turn):
        ai_text = _build_targeted_clarifying_question(merged_profile)
        return ai_text, adapt_level, calib_level, []

    # 6) 更保守地判断是否可以进入结束阶段
    if _is_need_clear_enough(merged_profile, current_turn, user_msg, stable_counts):
        ai_text = _build_stop_message(merged_profile)
        return ai_text, adapt_level, calib_level, []

    # 7) 做校准型选品
    history_product_ids = _get_history_product_ids(session_uuid)
    selected_products = _select_products_by_calibration(
        all_products=all_products,
        user_intent=user_intent,
        merged_profile=merged_profile,
        calib_level=calib_level,
        history_product_ids=history_product_ids,
        top_n=5
    )

    # comparison 场景允许加入少量最近历史商品做对比
    if user_intent == "comparison":
        recent_history = _get_recent_history_products(session_uuid, max_n=2)
        selected_products = _dedup_products(recent_history + selected_products, max_n=5)

    # 8) 调用模型生成回复
    ai_text = call_deepseek_with_products(
        user_msg=user_msg,
        user_intent=user_intent,
        recommended_products=selected_products,
        adapt_level=adapt_level,
        calib_level=calib_level,
        memory_profile=merged_profile
    )

    core_products = extract_product_core_info(selected_products[:5])

    return ai_text, adapt_level, calib_level, core_products
