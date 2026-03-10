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
        return "当前轮暂无可用候选商品。"

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
        "用户历史需求摘要（这些信息若已出现，不要重复追问）：\n"
        f"- 当前总结：{summary}\n"
        f"- 预算：{budget if budget is not None else '未知'}\n"
        f"- 耳机类型：{headset_type or '未知'}\n"
        f"- 品牌偏好：{brand or '未知'}\n"
        f"- 核心功能：{('、'.join(core_functions) if core_functions else '未知')}\n"
        f"- 使用场景：{('、'.join(scenarios) if scenarios else '未知')}\n"
        f"- 已知槽位：{('、'.join(known_slots) if known_slots else '无')}"
    )


def _build_system_prompt(adapt_level: str, calib_level: str) -> str:
    return f"""你是一个自然、专业、口语化的耳机导购 AI。

你的任务不是尽快结束对话，而是在实验场景中稳定地完成推荐交流。

硬性规则：
1. 当你需要推荐**新**商品时，只能从“本轮允许推荐的新候选商品”中选择，绝对不允许编造商品。如果用户讨论你上一轮推荐过的商品，你可以结合【上一轮推荐的商品】上下文进行回复。
2. 当你推荐具体商品时，必须在商品名称后附带 Product ID，例如：索尼 XM5 [EAR001]。
3. 你必须优先使用“用户历史需求摘要”里的信息。摘要中已经明确给出的预算、类型、品牌、功能、场景，禁止重复追问。
4. 如果用户只是补充、追问、比较、继续看更多款，你要默认延续之前的需求，不要把对话当成一段全新会话。
5. 不要轻易说“我已经确认你的需求了”“已经完全明确了”。除非用户明确表示已经决定，否则请使用更保守的表达，比如“我目前大致理解你的方向了”“我先按这个思路给你推荐”。
6. **沉浸感要求**：如果你发现本轮候选商品中没有符合用户指定要求的（例如用户想要看更多小米，但当前列表里没有了），请自然地用人话向用户解释（例如：“目前店里没有其他合适的小米款式了，要不要看看其他品牌的类似款？”）。**绝对不允许**在回复中出现“候选列表”、“系统设定的商品”等暴露AI身份和后台数据的词汇。

实验操控规则：
- 适应性水平：{adapt_level}
  * HIGH：你像“私人耳机顾问”。当关键信息缺失时，可以追问，但一次最多追问 1-2 个最关键缺失项。若用户已有初步需求，不要过早下结论，而是优先做“确认式追问”，例如确认优先级、类型取舍、品牌偏好。
  * LOW：你像“自动售货机回复模块”。不要主动深挖，不要额外追问未提到的信息，主要基于当前候选商品和已有信息直接回应。

- 校准水平：{calib_level}
  * HIGH：推荐时要明确说明商品和用户需求之间的匹配关系，但不要写得像学术报告，要自然。
  * LOW：主要陈述商品事实，如名称、价格、功能，弱化解释。

回复风格要求：
1. 自然口语化，像真人导购，不要用 Markdown。
2. 正常回复控制在 120-220 字。
3. 推荐 1-3 款最相关商品即可，不要一次堆很多。
4. 如果是比较问题，就比较候选商品差异。
5. 如果当前候选商品不够吻合，也要如实说明，不要硬编。
"""


def call_deepseek_with_products(
    user_msg: str,
    user_intent: str,
    recommended_products: list,
    adapt_level: str,
    calib_level: str,
    memory_profile: Optional[Dict] = None
) -> str:
    product_text = _format_product_text(recommended_products)
    memory_text = _format_memory_text(memory_profile)
    system_prompt = _build_system_prompt(adapt_level, calib_level)

    user_prompt = f"""当前用户消息：{user_msg}
当前用户意图：{user_intent}

{memory_text}

{product_text}

请基于以上信息生成回复。
注意：
- 如果历史摘要里已经有预算、场景、类型、品牌或核心功能，就不要重复询问这些内容。
- 如果当前只是初步需求，不要表现得过早确定。
- 如果需要追问，只能问最关键的缺失项。"""

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
        return "抱歉，我这边刚刚响应有点慢。你前面提到的需求我会继续沿用，你可以再发一句，我接着帮你看。"

    except Exception as e:
        logger.error(f"DeepSeek API 调用失败：{str(e)}")
        return "抱歉，我暂时无法继续推荐。不过你前面提到的需求我会按原条件理解，你可以稍后再试一次。"

