"""B-SMT：基于 SMT 求解（Z3）的符号检测器。

复现 iRuler 式 SMT 分析在 TARBench 统一语法上的核心，覆盖：
  - TM（Trigger Mutex）：单规则触发条件集不可满足 -> 死代码
  - BE（Safety Boundary Error）：触发数值超出传感器物理常识范围
  - IS（Illegal State Rule）：规则读写同一属性 -> 自强制稳态
  - AC（Action Conflict）：对同一属性赋不同值的规则对，
    其触发条件经 SMT 求解证明可同时满足 -> 真冲突

数值条件交给 Z3（Real 语义），离散条件（== / !=）用字符串等价手动分析。
"""

import z3

from rule_parser import parse_number, parse_sample_rules

# 传感器物理常识范围（人工知识库，独立于任何评测数据，避免数据泄漏）
SENSOR_RANGES = {
    "temperature_sensor_value": (-50.0, 60.0),    # °C
    "humidity_sensor_value": (0.0, 100.0),        # %
    "light_sensor_value": (0.0, 100000.0),        # lux
    "air_quality_sensor_value": (0.0, 500.0),     # AQI
    "smoke_sensor_value": (0.0, 1000.0),          # ppm
    "gas_sensor_valve": (0.0, 1000.0),            # ppm
    "water_leak_sensor_value": (0.0, 1.0),
    "motion_sensor_value": (0.0, 1.0),
}

_Z3_OPS = {
    ">": lambda var, val: var > val,
    "<": lambda var, val: var < val,
    ">=": lambda var, val: var >= val,
    "<=": lambda var, val: var <= val,
}


def _z3_var(target: str, cache: dict):
    name = target.replace(".", "_").replace(" ", "_")
    if name not in cache:
        cache[name] = z3.Real(name)
    return cache[name]


def conditions_sat(conditions) -> bool:
    """判断条件集合是否可满足（环境变量为自由变量）。"""
    # 离散 ==：同一属性不能等于两个不同值
    discrete_eq = {}
    for cond in conditions:
        if cond.op == "==":
            if cond.target in discrete_eq and discrete_eq[cond.target] != cond.value:
                return False
            discrete_eq[cond.target] = cond.value
    # == 与 != 同值互斥
    for cond in conditions:
        if cond.op == "!=" and cond.target in discrete_eq and discrete_eq[cond.target] == cond.value:
            return False

    solver = z3.Solver()
    cache = {}
    for cond in conditions:
        number = parse_number(cond.value)
        if cond.op in _Z3_OPS and number is not None:
            solver.add(_Z3_OPS[cond.op](_z3_var(cond.target, cache), z3.RealVal(number)))
    return solver.check() != z3.unsat


def detect_smt(rules_dict: dict) -> dict:
    """对单个样本运行 B-SMT 检测。

    返回 {"has_anomaly": bool, "reason": str, "involved_rules": [rule_id...]}。
    """
    rules = parse_sample_rules(rules_dict)
    if not rules:
        return {"has_anomaly": False, "reason": "No parseable rules.", "involved_rules": []}

    findings = []
    involved = set()

    # 1) TM：死条件（触发条件自相矛盾）
    for rule in rules:
        if not conditions_sat(rule.conditions):
            findings.append(
                f"{rule.rule_id} has logically dead trigger conditions that cannot coexist (trigger mutex)"
            )
            involved.add(rule.rule_id)

    # 2) BE：触发数值超出传感器物理范围
    for rule in rules:
        for cond in rule.conditions:
            if cond.attribute not in SENSOR_RANGES:
                continue
            number = parse_number(cond.value)
            if number is None:
                continue
            low, high = SENSOR_RANGES[cond.attribute]
            if number < low or number > high:
                findings.append(
                    f"{rule.rule_id} uses trigger value {cond.value} on {cond.target} "
                    f"beyond the physical sensor range [{low}, {high}] (safety boundary error)"
                )
                involved.add(rule.rule_id)

    # 3) IS：自强制稳态（读写同一属性）
    for rule in rules:
        read_targets = {cond.target for cond in rule.conditions}
        write_targets = {action.target for action in rule.actions}
        shared = read_targets & write_targets
        if shared:
            findings.append(
                f"{rule.rule_id} reads and writes the same attribute {', '.join(sorted(shared))}, "
                f"forcing a forbidden steady state (illegal state rule)"
            )
            involved.add(rule.rule_id)

    # 4) AC：SMT 证实的动作冲突（触发条件联合可满足且赋值不同）
    writes = {}
    for rule in rules:
        for action in rule.actions:
            writes.setdefault(action.target, []).append((rule, action.value))

    for target in sorted(writes):
        entries = writes[target]
        if len(entries) < 2:
            continue
        values = {value for _, value in entries}
        if len(values) < 2:
            continue  # 相同赋值归 B-AR 的 duplication 处理
        for index in range(len(entries)):
            for jndex in range(index + 1, len(entries)):
                rule_i, value_i = entries[index]
                rule_j, value_j = entries[jndex]
                if value_i == value_j:
                    continue
                if conditions_sat(rule_i.conditions + rule_j.conditions):
                    findings.append(
                        f"{rule_i.rule_id} and {rule_j.rule_id} assign conflicting values "
                        f"({value_i} vs {value_j}) to {target} while their triggers can be "
                        f"simultaneously satisfied (action conflict)"
                    )
                    involved.update({rule_i.rule_id, rule_j.rule_id})

    if not findings:
        return {"has_anomaly": False, "reason": "No anomalies identified.", "involved_rules": []}

    ordered = sorted(involved, key=lambda key: int(key.split("_")[1]))
    reason = " ".join(findings) + "."
    return {"has_anomaly": True, "reason": reason, "involved_rules": ordered}
