"""B-AC：简单启发式动作冲突检测器（内部对照：B-SMT 的 AC 检查）。

结构化代理判断，无任何求解器推理：
    两条不同规则对同一 device.attribute 赋不同值，且触发条件共享
    至少一个属性（"相同/相交"触发），即报告动作冲突。

与 B-SMT（Z3 联合可满足性）的预期差异：
  - 误报方向：触发共享属性但区间互斥（如 temp>30 与 temp<20）时，
    B-AC 报告而 B-SMT 正确拒绝；
  - 漏报方向：跨属性触发但可同时满足的链式冲突（Rule_6/Rule_7 型），
    B-SMT 捕获而 B-AC 漏报。
与 B-AR（完全不查触发）构成第三档对照：触发推理从无到结构到形式。
"""

from collections import defaultdict

from rule_parser import parse_sample_rules


def _sort_ids(rule_ids):
    return sorted(set(rule_ids), key=lambda key: int(key.split("_")[1]))


def detect_action_conflict_heuristic(rules_dict: dict) -> dict:
    """对单个样本运行 B-AC 检测。

    返回 {"has_anomaly": bool, "reason": str, "involved_rules": [rule_id...]}。
    """
    rules = parse_sample_rules(rules_dict)
    if not rules:
        return {"has_anomaly": False, "reason": "No parseable rules.", "involved_rules": []}

    writes = defaultdict(list)
    for rule in rules:
        for action in rule.actions:
            writes[action.target].append((rule, action.value))

    findings = []
    involved = set()

    for target in sorted(writes):
        entries = writes[target]
        for index in range(len(entries)):
            for jndex in range(index + 1, len(entries)):
                rule_i, value_i = entries[index]
                rule_j, value_j = entries[jndex]
                if value_i == value_j or rule_i.rule_id == rule_j.rule_id:
                    continue  # 同值归 duplication；单规则内部互斥归 action mutex
                trigger_targets_i = {cond.target for cond in rule_i.conditions}
                trigger_targets_j = {cond.target for cond in rule_j.conditions}
                shared = trigger_targets_i & trigger_targets_j
                if not shared:
                    continue  # 结构启发式：触发不相交则不认为可共存
                findings.append(
                    f"{rule_i.rule_id} and {rule_j.rule_id} assign conflicting values "
                    f"({value_i} vs {value_j}) to {target} while their triggers share "
                    f"{', '.join(sorted(shared))} (action conflict)"
                )
                involved.update({rule_i.rule_id, rule_j.rule_id})

    if not findings:
        return {"has_anomaly": False, "reason": "No anomalies identified.", "involved_rules": []}

    ordered = _sort_ids(involved)
    reason = " ".join(findings) + "."
    return {"has_anomaly": True, "reason": reason, "involved_rules": ordered}
