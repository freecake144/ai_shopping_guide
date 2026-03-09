import os
import logging
import httpx
from typing import Dict, List, Optional

from openai import OpenAI
from openai._exceptions import OpenAIError

try:
    from openai._exceptions import Timeout
except ImportError:
    try:
        from openai._exceptions import timeout as Timeout
    except ImportError:
        Timeout = OpenAIError

# 本地开发时加载 .env，Render 部署依赖环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def init_deepseek_client():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("❌ 未配置 DEEPSEEK_API_KEY 环境变量！")
        raise ValueError("请在 Render 控制台 -> Environment 添加 DEEPSEEK_API_KEY")

    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    http_client = httpx.Client(
        timeout=30.0,
        proxies=None,
        follow_redirects=True
    )

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client
        )
        logger.info("✅ DeepSeek 客户端初始化成功")
        return client
    except Exception as e:
        logger.error(f"❌ 客户端初始化失败：{str(e)}")
        raise


client = init_deepseek_client()


def _format_product_text(recommended_products: List[Dict]) -> str:
    if not recommended_products:
        return "暂无可用商品"

    lines = []
    for i, p in enumerate(recommended_products, start=1):
        lines.append(
            f"{i}. {p.get('product_name', '未知商品')} "
            f"[{p.get('product_id', 'NO_ID')}] | "
            f"¥{p.get('price', '未知')} | "
            f"{p.get('headset_type', '无类型')} | "
            f"功能: {p.get('core_function', '无')} | "
            f"品牌: {p.get('brand', '无')} | "
            f"适合: {p.get('scenario', '日常')}"
        )
    return "当前轮允许推荐的全部候选商品如下（只能从中选择，不允许编造列表外商品）：\n" + "\n".join(lines)


def _format_memory_text(memory_profile: Optional[Dict]) -> str:
    if not memory_profile:
        return (
            "用户历史需求摘要：暂无明确记录。\n"
            "已知槽位：无。"
        )

    budget = memory_profile.get("max_price")
    headset_type = memory_profile.get("headset_type")
    brand = memory_profile.get("brand")
    core_functions = memory_profile.get("core_functions", [])
    scenarios = memory_profile.get("scenarios", [])
    summary = memory_profile.get("summary", "暂无明确需求")
    known_slots = memory_profile.get("known_slots", [])

    return (
        "用户历史需求摘要（这些是已经明确说过的信息，不要重复追问）：\n"
        f"- 汇总摘要：{summary}\n"
        f"- 预算：{budget if budget is not None else '未知'}\n"
        f"- 耳机类型：{headset_type or '未知'}\n"
        f"- 品牌偏好：{brand or '未知'}\n"
        f"- 核心功能：{('、'.join(core_functions) if core_functions else '未知')}\n"
        f"- 使用场景：{('、'.join(scenarios) if scenarios else '未知')}\n"
        f"- 已知槽位：{('、'.join(known_slots) if known_slots else '无')}"
    )


def _build_system_prompt(adapt_level: str, calib_level: str) -> str:
    return f"""你是一个热情、专业、自然口语化的耳机导购 AI，目标是在实验场景中稳定地完成推荐。

关键规则：
1. 你只能从“当前轮允许推荐的全部候选商品”中选择商品，绝对不允许编造列表外商品。
2. 当你推荐具体商品时，必须在商品名称后附带 Product ID，例如：索尼 XM5 [EAR001]。
3. 你必须优先使用“用户历史需求摘要”中的已知信息来回答。凡是摘要中已经明确给出的内容（预算、类型、功能、品牌、场景等），禁止重复追问。
4. 只有当某项信息在摘要中确实缺失时，才允许追问；而且一次最多追问 1-2 个最关键缺失项。
5. 如果用户在当前轮表达的是比较、补充推荐、追问差异、品牌选择等后续问题，你要默认延续之前已知需求，不要把对话当成全新会话。
6. 回复自然口语化，尽量像真人导购，不用 Markdown，不要列很多生硬条目。
7. 推荐 1-3 款最相关商品即可，不要把所有商品都堆出来。
8. 你的回复中尽量包含：商品名称、价格、核心功能，以及与需求的对应关系。

实验操控规则：
- 适应性水平：{adapt_level}
  * HIGH：你是“私人耳机顾问”。如果关键信息缺失，可以追问，但只能问缺失项；如果信息已知，就要延续上下文，直接给建议。
  * LOW：你是“自动售货机回复模块”。不要主动深挖需求，不要额外追问，主要基于当前给定候选商品直接陈述推荐结果。

- 校准水平：{calib_level}
  * HIGH：必须明确解释“为什么这款适合用户当前需求”，突出需求-商品之间的匹配逻辑。
  * LOW：主要陈述商品事实（名称、价格、功能），弱化或不展开解释推荐理由。

特殊处理：
- 用户问价格范围时，优先围绕预算作答。
- 用户问品牌时，在候选商品中总结可选品牌，并推荐最贴近需求的品牌商品。
- 用户要求对比时，客观比较候选商品差异。
- 如果没有合适商品，也要如实说明“当前候选里没有特别吻合的”，不要编造。

长度建议：
- 正常回复控制在 120-220 字。
- 不要空泛寒暄，不要重复用户原话。
"""


def call_deepseek_with_products(
    user_msg: str,
    user_intent: str,
    recommended_products: list,
    adapt_level: str,
    calib_level: str,
    memory_profile: Optional[Dict] = None
) -> str:
    """
    调用 DeepSeek API，结合商品数据、用户意图、实验条件和会话记忆生成回复
    """
    product_text = _format_product_text(recommended_products)
    memory_text = _format_memory_text(memory_profile)
    system_prompt = _build_system_prompt(adapt_level, calib_level)

    user_prompt = f"""当前用户消息：{user_msg}
当前用户意图：{user_intent}

{memory_text}

{product_text}

请基于以上信息生成回复。
注意：如果历史摘要里已经有预算、场景、类型、品牌或核心功能，就不要重复询问这些内容。"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

    except Timeout as e:
        logger.error(f"DeepSeek API 超时：{str(e)}")
        return "抱歉，我这边刚刚响应有点慢。你刚才提到的需求我已经记住了，你可以再发一句，我继续按之前的条件帮你推荐。"

    except Exception as e:
        logger.error(f"DeepSeek API 调用失败：{str(e)}")
        return "抱歉，我暂时无法继续推荐。不过你前面提到的需求我会按原条件理解，你可以稍后再试一次。"
