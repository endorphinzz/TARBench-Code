"""B-AR：基于动作属性的规则间冲突检测器（Action Conflict / Duplication /
Revert / Action Mutex）。

对动作写入同一 device.attribute 的规则分组：
  - 组内赋值全相同        -> Action Duplication（冗余/重复动作）
  - 组内赋值不同          -> Action Conflict / Revert / Mutex
    * 同一条规则内部互斥赋值   -> State Action Mutex（自相矛盾）
    * 规则 i 的动作值恰好激活规则 j 的触发条件且 j 改写同一属性 -> Action Revert
    * 其余                     -> Action Conflict
"""

from collections import defaultdict

from rule_parser import action_activates_condition, parse_sample_rules


def _sort_ids(rule_ids):
    return sorted(set(rule_ids), key=lambda key: int(key.split("_")[1]))


def detect_action_conflicts(rules_dict: dict) -> dict:
    """对单个样本运行 B-AR 检测。

    返回 {"has_anomaly": bool, "reason": str, "involved_rules": [rule_id...]}。
    """
    rules = parse_sample_rules(rules_dict)
    if not rules:
        return {"has_anomaly": False, "reason": "No parseable rules.", "involved_rules": []}

    # 1) 按动作目标属性分组：target -> [(rule_id, value)]
    writes = defaultdict(list)
    for rule in rules:
        for action in rule.actions:
            writes[action.target].append((rule.rule_id, action.value))

    findings = []
    involved = set()

    for target in sorted(writes):
        entries = writes[target]
        rule_ids = [rule_id for rule_id, _ in entries]
        values = [value for _, value in entries]
        distinct = sorted(set(values))

        # 单规则内部互斥：State Action Mutex（如 X = on AND X = off）
        if len(entries) >= 2:
            by_rule = defaultdict(set)
            for rule_id, value in entries:
                by_rule[rule_id].add(value)
            for rule_id, value_set in by_rule.items():
                if len(value_set) >= 2:
                    findings.append(
                        f"{rule_id} issues mutually exclusive commands "
                        f"({', '.join(sorted(value_set))}) to {target} simultaneously (action mutex)"
                    )
                    involved.add(rule_id)

        # 组内只有一条规则且无内部互斥：跳过
        if len(set(rule_ids)) < 2:
            continue

        if len(distinct) == 1:
            findings.append(
                f"Rules {', '.join(_sort_ids(rule_ids))} duplicate the same action "
                f"{target} = {distinct[0]} (action duplication)"
            )
            involved.update(rule_ids)
            continue

        # 赋值不同：区分 Revert 与 Conflict
        # Revert：规则 i 写 X = v1 激活规则 j 的触发条件（读 X），j 再把 X 改为 v2
        reverted_pairs = set()
        for writer in rules:
            for action in writer.actions:
                if action.target != target:
                    continue
                for reader in rules:
                    if reader.rule_id == writer.rule_id:
                        continue
                    if any(action_activates_condition(action, cond) for cond in reader.conditions):
                        if reader.writes(target):
                            pair = tuple(_sort_ids([writer.rule_id, reader.rule_id]))
                            if pair not in reverted_pairs:
                                reverted_pairs.add(pair)

        conflict_rules = _sort_ids(rule_ids)
        if reverted_pairs:
            findings.append(
                f"Rules {', '.join(conflict_rules)} assign conflicting values "
                f"({', '.join(distinct)}) to {target} with chained propagation "
                f"(action revert: {', '.join(' & '.join(pair) for pair in sorted(reverted_pairs))})"
            )
        else:
            findings.append(
                f"Rules {', '.join(conflict_rules)} assign conflicting values "
                f"({', '.join(distinct)}) to {target} (action conflict)"
            )
        involved.update(rule_ids)

    if not findings:
        return {"has_anomaly": False, "reason": "No anomalies identified.", "involved_rules": []}

    reason = " ".join(findings) + "."
    return {"has_anomaly": True, "reason": reason, "involved_rules": _sort_ids(involved)}
