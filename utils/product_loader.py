import os
import re
import random
import copy
from typing import List, Dict, Any

import pandas as pd


# =========================
# 1. 路径与缓存
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCT_CSV_PATH = os.path.join(BASE_DIR, "data", "product_list.csv")

GLOBAL_PRODUCTS: List[Dict] = []


# =========================
# 2. 基础清洗函数
# =========================
def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _safe_lower(value: Any) -> str:
    return _safe_str(value).lower()


def _normalize_price(value: Any) -> float:
    """
    支持:
    - 299
    - "299"
    - "299元"
    - "¥299"
    - "299.00"
    """
    if value is None:
        return 0.0

    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)

    text = _safe_str(value)
    text = text.replace("¥", "").replace("元", "").replace(",", "").strip()

    m = re.search(r"\d+(\.\d+)?", text)
    if m:
        try:
            return float(m.group())
        except Exception:
            return 0.0
    return 0.0


def _normalize_sales_volume(value: Any) -> int:
    """
    支持:
    - 5000
    - "5000+"
    - "1.2万+"
    - "300"
    """
    if value is None:
        return 0

    if isinstance(value, (int, float)) and not pd.isna(value):
        return int(value)

    text = _safe_str(value).lower().replace(",", "")
    if not text:
        return 0

    # 处理中文“万”
    m_wan = re.search(r"(\d+(\.\d+)?)\s*万", text)
    if m_wan:
        try:
            return int(float(m_wan.group(1)) * 10000)
        except Exception:
            return 0

    m = re.search(r"\d+", text.replace("+", ""))
    if m:
        try:
            return int(m.group())
        except Exception:
            return 0

    return 0


def _normalize_core_function_list(value: Any) -> List[str]:
    """
    兼容:
    1. 字符串: "降噪，蓝牙，续航"
    2. 字符串: "降噪,蓝牙/续航"
    3. 列表: ["降噪", "蓝牙"]
    4. 空值: None / NaN
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    if isinstance(value, list):
        result = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                parts = re.split(r"[，,、/｜|；;]+", item)
                result.extend([x.strip() for x in parts if x and x.strip()])
            else:
                item_text = _safe_str(item)
                if item_text:
                    result.append(item_text)
        return [x for x in result if x]

    if isinstance(value, str):
        return [x.strip() for x in re.split(r"[，,、/｜|；;]+", value) if x and x.strip()]

    value_text = _safe_str(value)
    return [value_text] if value_text else []


def _normalize_scenario_list(value: Any) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    if isinstance(value, list):
        result = []
        for item in value:
            parts = re.split(r"[，,、/｜|；;]+", _safe_str(item))
            result.extend([x.strip() for x in parts if x and x.strip()])
        return result

    text = _safe_str(value)
    if not text:
        return []
    return [x.strip() for x in re.split(r"[，,、/｜|；;]+", text) if x and x.strip()]


def _normalize_record(record: Dict) -> Dict:
    """
    把每个商品统一成更稳的结构
    """
    r = dict(record)

    r["product_id"] = _safe_str(r.get("product_id"))
    r["product_name"] = _safe_str(r.get("product_name"))
    r["brand"] = _safe_str(r.get("brand"))
    r["headset_type"] = _safe_str(r.get("headset_type"))
    r["core_function"] = _safe_str(r.get("core_function"))
    r["scenario"] = _safe_str(r.get("scenario"))
    r["involvement_level"] = _safe_lower(r.get("involvement_level"))

    r["price"] = _normalize_price(r.get("price"))
    r["sales_volume_num"] = _normalize_sales_volume(r.get("sales_volume"))

    r["core_function_list"] = _normalize_core_function_list(r.get("core_function"))
    r["scenario_list"] = _normalize_scenario_list(r.get("scenario"))

    return r


# =========================
# 3. 加载 CSV
# =========================
def load_products_from_csv(force_reload: bool = False) -> List[Dict]:
    """
    从 CSV 加载商品并缓存
    """
    global GLOBAL_PRODUCTS

    if GLOBAL_PRODUCTS and not force_reload:
        return GLOBAL_PRODUCTS

    try:
        df = pd.read_csv(PRODUCT_CSV_PATH, encoding="utf-8")

        required_fields = [
            "product_id",
            "product_name",
            "price",
            "headset_type",
            "core_function",
            "involvement_level",
        ]
        missing_fields = [f for f in required_fields if f not in df.columns]
        if missing_fields:
            raise ValueError(f"商品CSV缺少核心字段：{', '.join(missing_fields)}")

        # 缺失列兜底，避免下游直接 KeyError
        optional_defaults = {
            "brand": "",
            "scenario": "",
            "sales_volume": 0,
            "battery_life(hours)": "",
        }
        for col, default_val in optional_defaults.items():
            if col not in df.columns:
                df[col] = default_val

        # 逐行标准化，避免 .str.split / re.split 因 list / NaN 崩掉
        records = df.to_dict("records")
        normalized_records = [_normalize_record(r) for r in records]

        # 去掉没有 product_id 的脏数据
        normalized_records = [r for r in normalized_records if r.get("product_id")]

        GLOBAL_PRODUCTS = normalized_records
        print(f"成功加载商品数据，共 {len(GLOBAL_PRODUCTS)} 款商品")

        # 调试阶段可临时打开
        # print([p["core_function"] for p in GLOBAL_PRODUCTS[:5]])
        # print([p["core_function_list"] for p in GLOBAL_PRODUCTS[:5]])

        return GLOBAL_PRODUCTS

    except FileNotFoundError:
        raise FileNotFoundError(f"商品CSV文件未找到，请检查路径：{PRODUCT_CSV_PATH}")
    except Exception as e:
        raise Exception(f"加载商品CSV失败：{str(e)}")


# =========================
# 4. 匹配工具
# =========================
def _match_headset_type(product: Dict, headset_type: str) -> bool:
    if not headset_type:
        return True
    return _safe_str(product.get("headset_type")) == _safe_str(headset_type)


def _match_brand(product: Dict, brand: str) -> bool:
    if not brand:
        return True
    return _safe_str(product.get("brand")).lower() == _safe_str(brand).lower()


def _match_core_function(product: Dict, required_func: str) -> bool:
    if not required_func:
        return True

    required = _safe_str(required_func)
    funcs = product.get("core_function_list", [])
    funcs = funcs if isinstance(funcs, list) else []

    # 先做精确匹配
    if required in funcs:
        return True

    # 再做包含匹配，兼容“无线蓝牙”“蓝牙”“降噪麦克风”这类近似表述
    required_lower = required.lower()
    for f in funcs:
        f_lower = _safe_str(f).lower()
        if required_lower in f_lower or f_lower in required_lower:
            return True

    return False


def _match_price(product: Dict, max_price: Any) -> bool:
    if max_price is None:
        return True
    try:
        return float(product.get("price", 0)) <= float(max_price)
    except Exception:
        return False


def _sort_by_price_asc(products: List[Dict]) -> List[Dict]:
    return sorted(products, key=lambda x: float(x.get("price", 0)))


def _sort_by_sales_desc(products: List[Dict]) -> List[Dict]:
    return sorted(products, key=lambda x: int(x.get("sales_volume_num", 0)), reverse=True)


# =========================
# 5. 推荐主逻辑
# =========================
def get_matching_products(user_intent: str, intent_details: Dict, top_n: int = 3) -> List[Dict]:
    """
    HIGH 校准：基于意图和条件做规则筛选
    """
    products = load_products_from_csv()
    matched = copy.deepcopy(products)

    max_price = intent_details.get("max_price")
    headset_type = intent_details.get("headset_type")
    core_function = intent_details.get("core_function")
    brand = intent_details.get("brand")

    # 1) 价格敏感
    if user_intent == "price_sensitive":
        matched = [p for p in matched if _match_price(p, max_price)]
        matched = [p for p in matched if _match_headset_type(p, headset_type)]
        matched = [p for p in matched if _match_core_function(p, core_function)]
        matched = [p for p in matched if _match_brand(p, brand)]

        # 更适合先按价格升序，再看销量
        matched = sorted(
            matched,
            key=lambda x: (float(x.get("price", 0)), -int(x.get("sales_volume_num", 0)))
        )

    # 2) 常规推荐
    elif user_intent == "recommendation":
        matched = [p for p in matched if _match_headset_type(p, headset_type)]
        matched = [p for p in matched if _match_core_function(p, core_function)]
        matched = [p for p in matched if _match_brand(p, brand)]

        # 如果一个条件都没给，就按销量走
        if not any([headset_type, core_function, brand, max_price]):
            matched = _sort_by_sales_desc(matched)
        else:
            matched = sorted(
                matched,
                key=lambda x: (
                    0 if _match_brand(x, brand) else 1,
                    0 if _match_headset_type(x, headset_type) else 1,
                    0 if _match_core_function(x, core_function) else 1,
                    -int(x.get("sales_volume_num", 0))
                )
            )

        # 若给了预算，再做一个软过滤
        if max_price is not None:
            budget_matched = [p for p in matched if _match_price(p, max_price)]
            if budget_matched:
                matched = budget_matched

    # 3) 对比
    elif user_intent == "comparison":
        # 优先同类型；若无则不过滤类型
        if headset_type:
            type_matched = [p for p in matched if _match_headset_type(p, headset_type)]
            if type_matched:
                matched = type_matched

        if core_function:
            func_matched = [p for p in matched if _match_core_function(p, core_function)]
            if func_matched:
                matched = func_matched

        # 如果指定品牌，优先这些品牌；否则保留多品牌
        if brand:
            brand_matched = [p for p in matched if _match_brand(p, brand)]
            if brand_matched:
                matched = brand_matched

        # 选不同品牌更利于对比
        matched = _sort_by_sales_desc(matched)
        dedup_brands = []
        seen_brands = set()
        for p in matched:
            b = _safe_str(p.get("brand"))
            if b not in seen_brands:
                dedup_brands.append(p)
                seen_brands.add(b)
        matched = dedup_brands

    # 4) 探索
    elif user_intent == "exploration":
        matched = _sort_by_sales_desc(matched)

    else:
        matched = _sort_by_sales_desc(matched)

    # 兜底：没匹配到时不要递归自己，直接随机
    if not matched:
        return get_random_products(top_n=top_n)

    return matched[:top_n]


def render_product_text(products: List[Dict]) -> str:
    lines = []
    for p in products:
        line = (
            f"- {p.get('product_name', '未知商品')}｜"
            f"¥{p.get('price', '未知')}｜"
            f"{p.get('headset_type', '无类型')}｜"
            f"{p.get('core_function', '无功能')}｜"
            f"适合场景：{p.get('scenario', '日常')}"
        )
        lines.append(line)
    return "\n".join(lines)


def get_random_products(top_n: int = 3) -> List[Dict]:
    """
    LOW 校准：随机推荐
    """
    products = copy.deepcopy(load_products_from_csv())
    random.shuffle(products)
    return products[:top_n]


def filter_products_by_involvement(products: List[Dict], level: str) -> List[Dict]:
    """
    根据涉入度过滤
    如果 level 无效或过滤后为空，则回退到原列表
    """
    level = _safe_lower(level)
    if not level:
        return products

    filtered = [p for p in products if _safe_lower(p.get("involvement_level")) == level]
    return filtered if filtered else products


def extract_product_core_info(products: List[Dict]) -> List[Dict]:
    """
    精简存库字段
    """
    core_fields = [
        "product_id",
        "product_name",
        "price",
        "headset_type",
        "core_function",
        "brand",
        "battery_life(hours)",
        "sales_volume",
        "involvement_level",
        "scenario",
    ]
    return [
        {field: product.get(field) for field in core_fields if field in product}
        for product in products
    ]
