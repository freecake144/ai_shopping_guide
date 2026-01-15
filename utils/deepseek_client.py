import os
import logging
from openai import OpenAI
from openai._exceptions import OpenAIError
try:
    from openai._exceptions import Timeout
except ImportError:
    try:
        from openai._exceptions import timeout as Timeout 
    except ImportError:
        Timeout = OpenAIError  # 极端兜底，用通用异常

#本地开发时加载.env文件，Render 部署无需此依赖（用平台环境变量）
try:
    from dotenv import load_dotenv
    load_dotenv()  # 本地开发加载.env，Render 会自动忽略
except ImportError:
    pass

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 初始化DeepSeek客户端（显式配置HTTP客户端，避免proxies冲突）
def init_deepseek_client():
    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        logger.error("❌ 未配置 DEEPSEEK_API_KEY 环境变量！")
        raise ValueError("请在Render控制台->Environment添加DEEPSEEK_API_KEY")
    
    base_url = os.environ.get('DEEPSEEK_BASE_URL', "https://api.deepseek.com")
    
    # 显式创建httpx客户端，禁用proxies，避免底层隐式传递
    http_client = httpx.Client(
        timeout=30.0,  # 超时配置
        proxies=None,  # 明确禁用代理，解决proxies参数冲突
        follow_redirects=True
    )
    
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client  # 显式传递HTTP客户端，覆盖默认逻辑
        )
        logger.info("✅ DeepSeek客户端初始化成功")
        return client
    except Exception as e:
        logger.error(f"❌ 客户端初始化失败：{str(e)}")
        raise

# 全局客户端实例
client = init_deepseek_client()

def call_deepseek_with_products(user_msg: str, user_intent: str, recommended_products: list, adapt_level: str, calib_level: str) -> str:
    """
    调用DeepSeek API，结合商品数据、用户意图、实验条件生成回复
    :param user_msg: 用户原始消息
    :param user_intent: 识别的用户意图（price_sensitive/recommendation等）
    :param recommended_products: 规则化匹配的商品列表（来自CSV）
    :param adapt_level: 实验的适应性水平（HIGH/LOW）
    :param calib_level: 实验的校准水平（HIGH/LOW）
    :return: DeepSeek生成的回复文本
    """
    # 1. 格式化商品数据（让大模型明确要推荐的商品，避免编造数据）
    if not recommended_products:
        product_text = "暂无可用商品"
    else:
        product_text = "可用商品列表（请从中选择推荐）:\n" + "\n".join([
            f"{i + 1}. {p['product_name']} | ¥{p['price']} | {p.get('headset_type', '无')} | 功能: {p.get('core_function', '无')} | 品牌: {p.get('brand', '无')} | 适合: {p.get('scenario', '日常')}"
            for i, p in enumerate(recommended_products)
        ])

    # 2. 构建系统提示词（控制大模型的角色和回复规则，保证实验一致性）
    system_prompt = f"""你是一个热情专业的耳机导购AI，像真人一样自然聊天。
    关键规则：
    1. 商品列表是全部可用耳机（共24款）,从中自主选择最合适的耳机推荐（可以选1-5款），绝对不允许编造列表外商品。
    2. 自主分析用户需求（预算、类型、功能、品牌等），从列表中匹配。
    3. 风格控制：
       - 适应性水平：{adapt_level}（HIGH：详细理解需求、主动追问不清楚的地方、给出个性化理由；LOW：简短回复、不追问）。
       - 校准水平：{calib_level}（HIGH：强调为什么这些商品匹配用户需求；LOW：直接推荐，不多解释匹配理由）。
    4. 特殊处理：
       - 用户问价格范围（如“1000以上”“高端”），优先选匹配的。
       - 用户问品牌，从列表总结可用品牌。
       - 用户要求对比，客观比较列表中商品。
       - 可以引用之前推荐过的商品。
    5. 回复自然口语化（用“你”“我”“呢”“哦”），必须提到商品名称、价格、核心功能。
       - 长度200-300字，自然结束，不用Markdown。
    """

    # 3. 构建用户提示词（融合用户消息、意图、商品列表）
    user_prompt = f"""用户消息：{user_msg}
用户意图：{user_intent}
需要推荐的商品列表：
{product_text}"""

    try:
        # 4. 调用DeepSeek API（遵循官方示例）
        response = client.chat.completions.create(
            model="deepseek-chat",  # 模型名称，固定为deepseek-chat
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,  # 非流式返回，适合聊天场景
            temperature=0.3  # 低温度，保证回复稳定（实验建议0.1-0.5）
        )

        # 5. 提取回复内容
        return response.choices[0].message.content.strip()

    except Exception as e:
        # 异常处理：API调用失败时返回兜底回复，避免实验中断
        print(f"DeepSeek API调用失败：{str(e)}")

        return f"抱歉，我暂时无法为你推荐耳机，请稍后再试。"




