import random
from .conditions import ExperimentConditions

def assign_participant():
    # 简单随机分组，或者使用区组随机化(Block Randomization)保证各组人数平衡
    groups = list(ExperimentConditions.GROUPS.keys())
    return random.choice(groups)