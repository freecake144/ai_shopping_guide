import copy
import random
from utils.product_loader import extract_product_core_info, load_products_from_csv
from utils.deepseek_client import call_deepseek_with_products
from typing import Tuple, List, Dict
from models.main import InteractionTurn

# 定义实验条件 (2x2 设计)
def get_experiment_condition(group_id):
    """
    明确 2×2 实验操控
    """
    condition_map = {
        'A': ('LOW', 'LOW'),
        'B': ('LOW', 'HIGH'),
        'C': ('HIGH', 'LOW'),
        'D': ('HIGH', 'HIGH')
    }
    # 防止无效group_id导致报错，默认给D
    return condition_map[group_id]

def assign_group():
    """随机分配用户到一个组"""
    return random.choice(['A', 'B', 'C', 'D'])


def get_ai_response(
        user_msg: str,
        group_id: str,
        current_turn: int,
        assigned_adaptivity: str,
        assigned_calibration: str,
        session_uuid: str,  # 新增参数：从/api/send传session_uuid，用于查询多轮历史
        previous_recommended_products: list = []  # 可保留作为兜底，但主要用db查询
) -> Tuple[str, str, str, List[Dict]]:
    adapt_level = assigned_adaptivity
    calib_level = assigned_calibration

    # 1. 实时加载所有商品作为基底
    all_products = load_products_from_csv()

    # 2. 从数据库查询该session所有历史AI推荐商品
    history_turns = InteractionTurn.query.filter_by(
        session_uuid=session_uuid,
        sender='ai'
    ).order_by(InteractionTurn.turn_index.asc()).all()  # 按轮次排序

    historical_products = []
    for turn in history_turns:
        if turn.recommended_products:  # recommended_products是JSON列表
            historical_products.extend(turn.recommended_products)

    # 3. 构建final_products：历史前置 + 当前所有商品
    final_products = copy.deepcopy(all_products)

    if historical_products:
        history_ids = {p['product_id'] for p in historical_products if 'product_id' in p}
        # 前置历史商品（去重）
        unique_history = []
        seen = set()
        for p in historical_products:
            pid = p.get('product_id')
            if pid and pid not in seen:
                unique_history.append(p)
                seen.add(pid)

        # 移除final_products中已有的历史，重新前置
        final_products = [p for p in final_products if p['product_id'] not in history_ids]
        final_products = unique_history + final_products

    # 4. 去重（保留历史优先顺序）
    seen_ids = set()
    dedup_products = []
    for p in final_products:
        pid = p.get('product_id')
        if pid and pid not in seen_ids:
            dedup_products.append(p)
            seen_ids.add(pid)
    final_products = dedup_products

    # 5. 调用DeepSeek（传全量商品上下文，让AI自主记住历史）
    ai_text = call_deepseek_with_products(
        user_msg=user_msg,
        user_intent="recommendation",
        recommended_products=final_products,  # 全量传给AI
        adapt_level=adapt_level,
        calib_level=calib_level
    )

    # 6. 前端显示：只取前8款（避免卡片过多），优先历史+最新
    display_products = unique_history[:4] + final_products[:8]  # 历史优先显示
    seen_display = set()
    unique_display = []
    for p in display_products:
        pid = p.get('product_id')
        if pid and pid not in seen_display:
            unique_display.append(p)
            seen_display.add(pid)
    core_products = extract_product_core_info(unique_display[:8])

    return ai_text, adapt_level, calib_level, core_products


