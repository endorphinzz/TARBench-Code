"""B-BE：基于设备本体值域的越界检查（内部对照：B-SMT 的 BE 检查）。

知识来源：resources/device_ontology.json 的 value_range（平台配置，
任何部署都自带），无需求解器：
  - 离散属性：值不在合法值集（如 AC_mode ∈ {cool, heat, auto, fan_only}）
  - 数值属性：值超出区间（如 temperature_sensor_value ∈ [-20, 59]）
  - 时间属性：alarm_clock_time 超出 00:00-23:59
检查范围：触发侧 + 动作侧。

与 B-SMT 的 BE 检查（独立构造的传感器物理范围 SENSOR_RANGES，
仅触发侧、仅数值）的预期差异：
  - B-BE 覆盖动作侧（Numerical Illegal State / 离散非法值），
    B-SMT 覆盖不了；
  - 两者触发侧数值检查的知识来源不同（本体 vs 独立物理常识）。
"""

import json
import re
from functools import lru_cache
from pathlib import Path

from rule_parser import parse_number, parse_sample_rules

ONTOLOGY_PATH = Path(__file__).resolve().parents[2] / "resources" / "device_ontology.json"

NUMERIC_RANGE_RE = re.compile(r"^(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)")
TIME_VALUE_RE = re.compile(r"^(\d{1,2}):(\d{2})$")


@lru_cache(maxsize=1)
def load_attribute_specs():
    """解析本体 value_range，返回 {attribute: ("numeric", lo, hi) |
    ("discrete", {小写值集}) | ("time", None)}；无条目的属性不入表。"""
    ontology = json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    specs = {}
    for attribute, entries in ontology.get("value_range", {}).items():
        first = str(entries[0]).strip() if entries else ""
        if attribute == "alarm_clock_time" or first.upper().startswith("HH:MM"):
            specs[attribute] = ("time", None)
            continue
        range_match = NUMERIC_RANGE_RE.match(first)
        if range_match:
            specs[attribute] = ("numeric", float(range_match.group(1)), float(range_match.group(2)))
        else:
            specs[attribute] = ("discrete", {str(value).strip().lower() for value in entries})
    return specs


def _check_value(attribute: str, value: str):
    """返回违规描述字符串；合规返回空串。"""
    spec = load_attribute_specs().get(attribute)
    if spec is None:
        return ""
    kind = spec[0]
    if kind == "time":
        match = TIME_VALUE_RE.match(value.strip())
        if not match:
            return f"malformed time value '{value}'"
        hour, minute = int(match.group(1)), int(match.group(2))
        if hour > 23 or minute > 59:
            return f"time value '{value}' outside 00:00-23:59"
        return ""
    if kind == "numeric":
        number = parse_number(value)
        if number is None:
            return ""
        _, low, high = spec
        if number < low or number > high:
            return f"value {value} outside legal range [{low:g}, {high:g}]"
        return ""
    # discrete
    if value.strip().lower() not in spec[1]:
        return f"value '{value}' not in legal value set {sorted(spec[1])}"
    return ""


def detect_boundary_errors(rules_dict: dict) -> dict:
    """对单个样本运行 B-BE 检测。

    返回 {"has_anomaly": bool, "reason": str, "involved_rules": [rule_id...]}。
    """
    rules = parse_sample_rules(rules_dict)
    if not rules:
        return {"has_anomaly": False, "reason": "No parseable rules.", "involved_rules": []}

    findings = []
    involved = set()
    for rule in rules:
        for cond in rule.conditions:
            violation = _check_value(cond.attribute, cond.value)
            if violation:
                findings.append(
                    f"{rule.rule_id} trigger on {cond.target} uses {violation} "
                    f"(boundary error)"
                )
                involved.add(rule.rule_id)
        for action in rule.actions:
            violation = _check_value(action.attribute, action.value)
            if violation:
                findings.append(
                    f"{rule.rule_id} assigns {action.target} = {action.value}, {violation} "
                    f"(boundary error)"
                )
                involved.add(rule.rule_id)

    if not findings:
        return {"has_anomaly": False, "reason": "No anomalies identified.", "involved_rules": []}

    ordered = sorted(involved, key=lambda key: int(key.split("_")[1]))
    reason = " ".join(findings) + "."
    return {"has_anomaly": True, "reason": reason, "involved_rules": ordered}
