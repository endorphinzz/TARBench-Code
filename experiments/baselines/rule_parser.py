"""TARBench 非 LLM 基线公共解析层。

规则文本格式（TARBench 统一语法）：
    IF <cond1> [AND <cond2> ...] THEN <action1> [AND <action2> ...]
条件格式：
    <device>.<attribute> <op> <value>    op ∈ {==, !=, >, <, >=, <=}
动作格式：
    <device>.<attribute> = <value>
其中 device 为含房间前缀的全名（如 living_room_fan），值可能携带单位
（如 18°C、100500lux、17:00）。
"""

import re
from dataclasses import dataclass

RULE_SPLIT_RE = re.compile(r"^IF\s+(?P<conditions>.+?)\s+THEN\s+(?P<actions>.+)$", re.DOTALL)
CONDITION_RE = re.compile(r"^(?P<target>.+?)\s*(?P<op>==|!=|>=|<=|>|<)\s*(?P<value>.+)$")
ACTION_RE = re.compile(r"^(?P<target>.+?)\s*=\s*(?P<value>.+)$")
NUMBER_PREFIX_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)")
AND_SPLIT_RE = re.compile(r"\s+AND\s+")


@dataclass
class Condition:
    device: str      # 设备全名（含房间前缀）
    attribute: str   # 属性名
    op: str          # ==, !=, >, <, >=, <=
    value: str       # 原始值字符串（可能带单位）

    @property
    def target(self) -> str:
        return f"{self.device}.{self.attribute}"


@dataclass
class Action:
    device: str
    attribute: str
    value: str

    @property
    def target(self) -> str:
        return f"{self.device}.{self.attribute}"


@dataclass
class Rule:
    rule_id: str          # Rule_1
    conditions: list      # list[Condition]
    actions: list         # list[Action]
    raw: str              # 原始规则文本

    def reads(self, target: str) -> bool:
        return any(cond.target == target for cond in self.conditions)

    def writes(self, target: str) -> bool:
        return any(action.target == target for action in self.actions)


def parse_number(value: str):
    """提取值字符串的前缀数字（去除单位），失败返回 None。"""
    match = NUMBER_PREFIX_RE.match(value)
    return float(match.group(1)) if match else None


def split_target(target: str):
    """'living_room_fan.fan_status' -> ('living_room_fan', 'fan_status')。"""
    device, _, attribute = target.rpartition(".")
    if not device:
        return None, None
    return device.strip(), attribute.strip()


def parse_condition(text: str):
    match = CONDITION_RE.match(text.strip())
    if not match:
        return None
    device, attribute = split_target(match.group("target"))
    if device is None:
        return None
    return Condition(device=device, attribute=attribute, op=match.group("op"), value=match.group("value").strip())


def parse_action(text: str):
    # 注意 ACTION_RE 的 '=' 不能误匹配 '=='：动作侧只出现赋值
    match = ACTION_RE.match(text.strip())
    if not match:
        return None
    device, attribute = split_target(match.group("target"))
    if device is None:
        return None
    return Action(device=device, attribute=attribute, value=match.group("value").strip())


def parse_rule(rule_id: str, rule_text: str):
    """解析单条规则文本；解析失败返回 None。"""
    match = RULE_SPLIT_RE.match(rule_text.strip())
    if not match:
        return None
    conditions = [c for c in (parse_condition(part) for part in AND_SPLIT_RE.split(match.group("conditions"))) if c]
    actions = [a for a in (parse_action(part) for part in AND_SPLIT_RE.split(match.group("actions"))) if a]
    if not conditions or not actions:
        return None
    return Rule(rule_id=rule_id, conditions=conditions, actions=actions, raw=rule_text.strip())


def parse_sample_rules(rules_dict: dict):
    """解析样本的 {'Rule_1': text, ...} 字典，按编号排序返回 list[Rule]。"""
    parsed = []
    for rule_id in sorted(rules_dict, key=lambda key: int(key.split("_")[1])):
        rule = parse_rule(rule_id, rules_dict[rule_id])
        if rule is not None:
            parsed.append(rule)
    return parsed


def action_activates_condition(action: Action, cond: Condition) -> bool:
    """判断动作赋值是否满足（激活）触发条件。数值不可比时保守返回 True。"""
    if action.target != cond.target:
        return False
    if cond.op == "==":
        return action.value == cond.value
    if cond.op == "!=":
        return action.value != cond.value
    left = parse_number(action.value)
    right = parse_number(cond.value)
    if left is None or right is None:
        return True  # 数值未知，保守认为可激活
    return {
        ">": left > right,
        "<": left < right,
        ">=": left >= right,
        "<=": left <= right,
    }[cond.op]
