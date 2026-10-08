"""B-TM：简单启发式触发条件互斥检测器（内部对照：B-SMT 的 TM 检查）。

仅做离散等值矛盾检查（字符串级，无求解器）：
  - 同一属性 == 两个不同值 -> 触发条件永假（死规则）
  - 同一属性 == v 且 != v   -> 触发条件永假

与 B-SMT（完整可满足性，含数值区间）的预期差异：
  B-SMT 额外捕获数值型互斥（如 temp>30 AND temp<20，
  即 Mathematical Trigger Mutex 子类），B-TM 结构上无法覆盖。
预期：B-TM 覆盖 State Trigger Mutex，漏掉 Mathematical Trigger Mutex。
"""

from rule_parser import parse_sample_rules


def _rule_is_dead(rule) -> str:
    """返回死条件原因字符串；非死规则返回空串。"""
    eq_values = {}
    neq_values = {}
    for cond in rule.conditions:
        if cond.op == "==":
            if cond.target in eq_values and eq_values[cond.target] != cond.value:
                return (
                    f"requires {cond.target} to be both {eq_values[cond.target]} "
                    f"and {cond.value}"
                )
            eq_values[cond.target] = cond.value
        elif cond.op == "!=":
            neq_values.setdefault(cond.target, set()).add(cond.value)

    for target, value in eq_values.items():
        if value in neq_values.get(target, set()):
            return f"requires {target} to be both {value} and not {value}"
    return ""


def detect_trigger_mutex_heuristic(rules_dict: dict) -> dict:
    """对单个样本运行 B-TM 检测。

    返回 {"has_anomaly": bool, "reason": str, "involved_rules": [rule_id...]}。
    """
    rules = parse_sample_rules(rules_dict)
    if not rules:
        return {"has_anomaly": False, "reason": "No parseable rules.", "involved_rules": []}

    findings = []
    involved = set()
    for rule in rules:
        cause = _rule_is_dead(rule)
        if cause:
            findings.append(
                f"{rule.rule_id} is logically dead code because its trigger {cause} "
                f"(trigger mutex)"
            )
            involved.add(rule.rule_id)

    if not findings:
        return {"has_anomaly": False, "reason": "No anomalies identified.", "involved_rules": []}

    ordered = sorted(involved, key=lambda key: int(key.split("_")[1]))
    reason = " ".join(findings) + "."
    return {"has_anomaly": True, "reason": reason, "involved_rules": ordered}
