import math
import re
from collections import Counter

class PreferenceAnalyzer:
    def __init__(self):
        # 从CSV提取的关键词
        self.headset_types = ["头戴式", "入耳式", "半入耳式", "颈挂式"]
        self.core_functions = [
            "降噪", "无线蓝牙", "超长续航", "防水", "游戏低延迟", "空间音频",
            "快充", "重低音", "高解析", "RGB灯效", "触控", "降噪麦克风"
        ]
        self.brands = [
            "索尼", "苹果", "小米", "漫步者", "雷柏", "华为", "森海塞尔", "倍思",
            "荣耀", "JBL", "西伯利亚", "OPPO", "vivo", "铁三角", "Skullcandy",
            "先锋", "三星", "微软", "HyperX", "飞利浦", "Bose", "Anker", "Beats"
        ]
        self.scenarios = ["通勤", "日常", "运动", "游戏", "办公", "音乐"]

        # 价格相关关键词（电商常见表达）
        self.price_low_keywords = ['便宜', '性价比', '实惠', '划算', '低价', '学生', '百元', '200以内', '300以下', '预算低']
        self.price_mid_keywords = ['中等', '中端', '千元', '500-1000', '平衡']
        self.price_high_keywords = ['贵', '好一点', '预算充足', '顶级', '旗舰', '高端', '不差钱', '1000以上', '2000以上']

        # 决策阶段关键词（从探索→考虑→决策）
        self.exploration_keywords = ['推荐', '有什么', '有哪些', '介绍', '看看', '了解']
        self.consideration_keywords = ['对比', '区别', '哪个好', '优缺点', '参数', '怎么样']
        self.decision_keywords = ['就买', '下单', '链接', '决定', '就要', '购买', '买这个']

        # 明确度相关（提及具体属性越多，越明确）
        self.specific_keywords = self.headset_types + self.core_functions + self.brands + ['预算', '价格', '续航', '音质', '佩戴']

    def _calculate_price_preference(self, text: str) -> float:
        """价格偏好: -1(强烈低价) ~ 0(中性) ~ 1(强烈高价)"""
        score = 0.0
        lower_text = text.lower()

        low_count = sum(1 for k in self.price_low_keywords if k in lower_text)
        mid_count = sum(1 for k in self.price_mid_keywords if k in lower_text)
        high_count = sum(1 for k in self.price_high_keywords if k in lower_text)

        if low_count > 0:
            score -= 0.6 * low_count
        if high_count > 0:
            score += 0.6 * high_count
        if mid_count > 0:
            score += 0.0  # 中性，不偏移

        return max(min(score, 1.0), -1.0)

    def _calculate_specificity(self, text: str) -> float:
        """明确度: 0(模糊) ~ 1(高度明确，提及多个具体属性)"""
        lower_text = text.lower()
        matches = sum(1 for k in self.specific_keywords if k in lower_text)
        # 归一化：提及3个以上属性视为高度明确
        return min(matches / 3.0, 1.0)

    def _calculate_decision_readiness(self, text: str) -> float:
        """决策准备度: 0(探索) ~ 0.5(考虑) ~ 1(决策)"""
        lower_text = text.lower()
        if any(k in lower_text for k in self.decision_keywords):
            return 1.0
        elif any(k in lower_text for k in self.consideration_keywords):
            return 0.6
        elif any(k in lower_text for k in self.exploration_keywords):
            return 0.3
        return 0.3  # 默认探索阶段

    def _extract_preferred_attributes(self, text: str) -> dict:
        """提取用户偏好的具体属性（用于多维向量）"""
        lower_text = text.lower()
        prefs = {
            "headset_type": [t for t in self.headset_types if t in lower_text],
            "core_function": [f for f in self.core_functions if f in lower_text],
            "brand": [b for b in self.brands if b in lower_text],
            "scenario": [s for s in self.scenarios if s in lower_text]
        }
        # 强度：提及次数（简单Counter）
        func_counter = Counter(prefs["core_function"])
        prefs["core_function_strength"] = {f: func_counter.get(f, 0) for f in self.core_functions}
        return prefs

    def compute_vector(self, text: str) -> dict:
        """生成丰富偏好向量（适合时间序列分析和SEM）"""
        return {
            "price_preference": self._calculate_price_preference(text),  # -1 ~ 1
            "specificity": self._calculate_specificity(text),            # 0 ~ 1
            "decision_readiness": self._calculate_decision_readiness(text),  # 0 ~ 1
            "preferred_attributes": self._extract_preferred_attributes(text)  # 具体偏好字典
        }

    def calculate_drift(self, current_vec: dict, last_vec: dict) -> float:
        """计算偏好漂移（综合欧氏距离 + 属性变化）"""
        if not last_vec:
            return 0.0

        # 数值维度漂移
        num_keys = ["price_preference", "specificity", "decision_readiness"]
        diff = sum(
            (current_vec.get(k, 0) - last_vec.get(k, 0)) ** 2
            for k in num_keys
        )

        # 属性变化（Jaccard距离简化版）
        curr_attrs = current_vec.get("preferred_attributes", {})
        last_attrs = last_vec.get("preferred_attributes", {})
        attr_diff = 0.0
        for key in ["headset_type", "core_function", "brand", "scenario"]:
            curr_set = set(curr_attrs.get(key, []))
            last_set = set(last_attrs.get(key, []))
            if curr_set or last_set:
                attr_diff += 1 - len(curr_set & last_set) / len(curr_set | last_set)

        return math.sqrt(diff + attr_diff)

    def identify_focus(self, text: str) -> str:
        """识别当前主要关注维度（更细粒度）"""
        lower_text = text.lower()
        if any(k in lower_text for k in self.price_low_keywords + self.price_mid_keywords + self.price_high_keywords + ['预算', '价格']):
            return 'price'
        if any(b in lower_text for b in self.brands):
            return 'brand'
        if any(f in lower_text for f in self.core_functions + ['音质', '佩戴', '续航', '参数']):
            return 'function'
        if any(t in lower_text for t in self.headset_types):
            return 'type'
        if any(s in lower_text for s in self.scenarios):
            return 'scenario'
        return 'exploration'