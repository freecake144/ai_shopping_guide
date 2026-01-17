import copy
import random
from utils.product_loader import  extract_product_core_info, load_products_from_csv
from utils.deepseek_client import call_deepseek_with_products
from typing import Tuple, List, Dict
import re

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

    # 1. 加载商品 (保持不变)
    all_products = load_products_from_csv()
    final_products = copy.deepcopy(all_products)

    # 2. 处理历史商品前置 (保持不变)
    if previous_recommended_products:
        history_ids = {p['product_id'] for p in previous_recommended_products}
        final_products = [p for p in previous_recommended_products if p['product_id'] in history_ids] + \
                         [p for p in final_products if p['product_id'] not in history_ids]
        for prev_p in previous_recommended_products:
            if prev_p['product_id'] not in {p['product_id'] for p in final_products}:
                final_products.append(prev_p)

    # 3. 去重 (保持不变)
    seen_ids = set()
    dedup_products = []
    for p in final_products:
        if p['product_id'] not in seen_ids:
            dedup_products.append(p)
            seen_ids.add(p['product_id'])
    final_products = dedup_products

    # =======================================================
    # 注入“协议指令”
    # =======================================================
    # 随着对话进行，每一轮都会提醒 AI 遵守格式，防止它“忘事”
    protocol_instruction = (
        "\n\n[System Instruction]: "
        "在回复的最后，您必须明确列出产品 ID"
        "您目前推荐的格式如下： "
        "'||REC: [EARxxx], [EARxxx]||'。"
        "请勿包含仅用于比较或历史背景提及的产品 ID。 "
        "现在只列出你希望用户点击的选项。"
    )

    # 将指令拼接到用户输入中发送给 AI
    augmented_user_msg = user_msg + protocol_instruction

    # 4. 调用 AI (传入拼接后的 msg)
    ai_text = call_deepseek_with_products(
        user_msg=augmented_user_msg,
        user_intent="recommendation",
        recommended_products=final_products,
        adapt_level=adapt_level,
        calib_level=calib_level
    )

    # =======================================================
    #  优先匹配“协议格式”
    # =======================================================
    recommended_objs = []
    found_ids = []

    # 策略 1: 尝试提取 ||REC: ... || 中的内容 (最精准，排除历史干扰)
    strict_match = re.search(r"\|\|REC:\s*(.*?)\|\|", ai_text, re.DOTALL | re.IGNORECASE)
    if strict_match:
        # 提取括号里的内容，例如从 "[EAR001], [EAR002]" 中提取
        content = strict_match.group(1)
        found_ids = re.findall(r"(EAR\d{3})", content, re.IGNORECASE)

    # 策略 2: 如果 AI 没遵守协议 (漏写了)，再回退到之前的“全文搜索”作为保底
    if not found_ids:
        found_ids = re.findall(r"(EAR\d{3})", ai_text, re.IGNORECASE)

    # 去重并查找对象
    found_ids = list(set([fid.upper() for fid in found_ids]))

    if found_ids:
        for pid in found_ids:
            match = next((p for p in final_products if p['product_id'] == pid), None)
            if match:
                recommended_objs.append(match)

    # 5. 最终兜底 (保持不变)
    # 此时 recommended_objs 应该非常干净，只包含 AI 明确放在 ||REC: ... || 里的商品
    if recommended_objs:
        core_products = extract_product_core_info(recommended_objs)
    else:
        # 真的没找到，才显示默认6个
        core_products = extract_product_core_info(all_products[:6])

    # 6. 清理 AI 回复文本 (可选)
    # 为了用户体验，你可以把尾巴上的 ||REC: ... || 删掉，不显示给用户
    # 如果你想保留给开发者调试看，可以注释掉下面这行
    ai_text = re.sub(r"\|\|REC:.*?\|\|", "", ai_text, flags=re.DOTALL).strip()

    return ai_text, adapt_level, calib_level, core_products




