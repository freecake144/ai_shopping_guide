import copy
import random
from utils.product_loader import  extract_product_core_info, load_products_from_csv
from utils.deepseek_client import call_deepseek_with_products
from typing import Tuple, List, Dict

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
    previous_recommended_products: list = []
) -> Tuple[str, str, str, List[Dict]]:
    adapt_level = assigned_adaptivity
    calib_level = assigned_calibration

    # 实时加载所有商品
    all_products = load_products_from_csv()
    final_products = copy.deepcopy(all_products)

    # 优先保留并前置历史商品（确保对比/追问时可用）
    if previous_recommended_products:
        history_ids = {p['product_id'] for p in previous_recommended_products}
        # 前置历史
        final_products = [p for p in previous_recommended_products if p['product_id'] in history_ids] + \
                         [p for p in final_products if p['product_id'] not in history_ids]
        # 补充历史（去重）
        for prev_p in previous_recommended_products:
            if prev_p['product_id'] not in {p['product_id'] for p in final_products}:
                final_products.append(prev_p)

    # 去重
    seen_ids = set()
    dedup_products = []
    for p in final_products:
        if p['product_id'] not in seen_ids:
            dedup_products.append(p)
            seen_ids.add(p['product_id'])
    final_products = dedup_products

    # 调用DeepSeek（传完整列表，让AI自主选）
    ai_text = call_deepseek_with_products(
        user_msg=user_msg,
        user_intent="recommendation",  # 可根据 adapt_level 动态调整
        recommended_products=final_products,
        adapt_level=adapt_level,
        calib_level=calib_level
    )

    # 从AI回复文本中提取被推荐的商品
    recommended_objs = []
    for p in final_products:
        p_name = p.get('product_name', '').lower()
        p_id = p.get('product_id', '').lower()
        if p_id and p_id in ai_text.lower():
            recommended_objs.append(p)
        elif p_name and p_name in ai_text.lower():
            recommended_objs.append(p)

    # 返回核心商品（用于前端显示：AI回复中提到的，或默认前几款）
    if recommended_objs:
        core_products = extract_product_core_info(recommended_objs)
    else:
        # 兜底（AI 没点名时）
        core_products = extract_product_core_info(all_products[:6])

    return ai_text, adapt_level, calib_level, core_products


