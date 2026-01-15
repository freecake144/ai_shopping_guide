import pandas as pd
import os
import random
from typing import List, Dict
import copy

# 配置商品CSV路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
PRODUCT_CSV_PATH = os.path.join(BASE_DIR, "data", "product_list.csv")

# 全局变量：缓存商品数据（避免每次推荐都重新读CSV，提升性能）
GLOBAL_PRODUCTS: List[Dict] = []


def load_products_from_csv() -> List[Dict]:
    """
    从CSV加载商品数据，转为字典列表（全局缓存，只加载一次）
    返回：商品字典列表，每个字典对应一条商品数据
    """
    global GLOBAL_PRODUCTS
    if GLOBAL_PRODUCTS:  # 已缓存，直接返回
        return GLOBAL_PRODUCTS

    # 读取CSV
    try:
        # CSV编码调整
        df = pd.read_csv(PRODUCT_CSV_PATH, encoding="UTF-8")
        # 确保核心字段存在（无则报错，提示用户补充）
        required_fields = [
            "product_id",  # 商品唯一ID（如EAR001）
            "product_name",  # 商品名称
            "price",  # 价格（用于价格敏感匹配）
            "headset_type",  # 耳机类型（头戴式/入耳式/半入耳式，核心匹配字段）
            "core_function"  # 核心功能（降噪/无线蓝牙，核心匹配字段）
        ]
        missing_fields = [f for f in required_fields if f not in df.columns]
        if missing_fields:
            raise ValueError(f"商品CSV缺少核心字段：{', '.join(missing_fields)}")

        # 处理属性字段（转为纯Python列表，避免数组类型）
        df["core_function"] = df["core_function"].fillna("")  # 空值转为空字符串
        df["core_function_list"] = df["core_function"].str.split(",").apply(
            lambda x: [func.strip() for func in x if func.strip()]  # 移除空字符串元素
        ).apply(list)  # 强制转为Python列表（避免numpy类型）

        # 转为字典列表（全局缓存）
        GLOBAL_PRODUCTS = df.to_dict("records")
        print(f"成功加载商品数据，共{len(GLOBAL_PRODUCTS)}款商品")
        return GLOBAL_PRODUCTS
    except FileNotFoundError:
        raise FileNotFoundError(f"商品CSV文件未找到，请检查路径：{PRODUCT_CSV_PATH}")
    except Exception as e:
        raise Exception(f"加载商品CSV失败：{str(e)}")


def get_matching_products(user_intent: str, intent_details: Dict, top_n: int = 3) -> List[Dict]:
    """
    高校准推荐（HIGH校准）：根据耳机用户意图匹配商品
    参数：
        user_intent: 用户意图（如"price_sensitive"、"recommendation"）
        intent_details: 意图详情（如{"max_price": 500, "headset_type": "入耳式"}）
        top_n: 推荐商品数量
    返回：匹配的商品列表
    """
    products = load_products_from_csv()
    matched = copy.deepcopy(products)

    # 1. 价格敏感意图（匹配预算+可选属性）
    if user_intent == "price_sensitive":
        max_price = intent_details.get("max_price", 500)
        # 若用户同时指定了耳机类型/功能，叠加筛选
        matched = [p for p in products if float(p["price"]) <= max_price]
        if "headset_type" in intent_details:
            matched = [p for p in matched if p["headset_type"] == intent_details["headset_type"]]
        if "core_function" in intent_details:
            required_func = intent_details["core_function"]
            matched = [p for p in matched if required_func in p["core_function_list"]]

    # 2. 直接推荐意图（匹配耳机类型/功能/品牌）
    elif user_intent == "recommendation":
        matched = products
        # 筛选耳机类型
        if "headset_type" in intent_details:
            matched = [p for p in matched if p["headset_type"] == intent_details["headset_type"]]
        # 筛选核心功能（如降噪）
        if "core_function" in intent_details:
            required_func = intent_details["core_function"]
            matched = [p for p in matched if required_func in p["core_function_list"]]
        # 筛选品牌
        if "brand" in intent_details:
            matched = [p for p in matched if p["brand"] == intent_details["brand"]]

    # 3. 对比意图（匹配同类型不同品牌）
    elif user_intent == "comparison":
        headset_type = intent_details.get("headset_type", "入耳式")
        matched = [p for p in matched if p["headset_type"] == headset_type]
        # 筛选不同品牌的同类型耳机
        brands = list(set([p["brand"] for p in matched]))[:top_n*2]  # 取前6个品牌
        matched = [p for p in matched if p["brand"] in brands]

    # 4. 探索意图（高销量优先）
    elif user_intent == "exploration":
        # 按销量排序（处理"5000+"这种字符串，提取数字）
        def get_sales_num(sales_str):
            return int(sales_str.replace("+", "")) if isinstance(sales_str, str) else sales_str
        matched = sorted(products, key=lambda x: get_sales_num(x["sales_volume"]), reverse=True)

    # 取前N款，不足则随机兜底
    result = matched[:top_n] if matched else get_matching_products("exploration", {}, top_n)
    return result

# 商品转自然语言
def render_product_text(products):
    lines = []
    for p in products:
        line = (
            f"- {p['product_name']}｜¥{p['price']}｜{p['headset_type']}｜"
            f"{p['core_function']}｜适合场景：{p['scenario']}"
        )
        lines.append(line)
    return "\n".join(lines)

def get_random_products(top_n: int = 3) -> List[Dict]:
    """低校准推荐：随机推荐商品（确保打乱顺序）"""
    products = load_products_from_csv()
    random.shuffle(products)  # 关键：打乱商品列表顺序
    return products[:top_n]

# 提取推荐商品的核心信息（减少数据库存储冗余）
def extract_product_core_info(products: List[Dict]) -> List[Dict]:
    """
    提取商品的核心字段（用于存入数据库，避免存储冗余数据）
    返回：仅包含核心字段的商品字典列表
    """
    core_fields = ["product_id", "product_name", "price", "headset_type","core_function", "brand","battery_life(hours)","sales_volume"]
    return [
        {field: product[field] for field in core_fields if field in product}
        for product in products
    ]


