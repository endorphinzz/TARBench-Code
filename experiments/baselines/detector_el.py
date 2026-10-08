"""B-EL：触发-动作传播图环路检测器。

将规则集建模为有向图：
  - 节点 = 规则
  - 边 rule_i -> rule_j：rule_i 的某个动作写属性 X，且 rule_j 的某个触发
    条件读属性 X，且该赋值能激活该条件（值匹配；数值不可比时保守建边）。
环（SCC 大小 >= 2 或自环）即 endless loop 威胁。检测逻辑复现
iRuler/SOTERIA 式数据流分析在 TARBench 统一语法上的核心。
"""

from collections import defaultdict

from rule_parser import action_activates_condition, parse_sample_rules


def build_propagation_graph(rules):
    """返回邻接表 {rule_id: set(rule_id)}，边含义见模块 docstring。"""
    graph = defaultdict(set)
    for writer in rules:
        for action in writer.actions:
            for reader in rules:
                for cond in reader.conditions:
                    if action_activates_condition(action, cond):
                        graph[writer.rule_id].add(reader.rule_id)
    return graph


def find_sccs(graph, rule_ids):
    """Tarjan 强连通分量（迭代实现，避免深递归）。"""
    index_counter = [0]
    stack, on_stack = [], set()
    index, lowlink = {}, {}
    sccs = []

    for root in rule_ids:
        if root in index:
            continue
        work = [(root, iter(graph.get(root, ())))]
        index[root] = lowlink[root] = index_counter[0]
        index_counter[0] += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, neighbours = work[-1]
            advanced = False
            for nxt in neighbours:
                if nxt not in index:
                    index[nxt] = lowlink[nxt] = index_counter[0]
                    index_counter[0] += 1
                    stack.append(nxt)
                    on_stack.add(nxt)
                    work.append((nxt, iter(graph.get(nxt, ()))))
                    advanced = True
                    break
                if nxt in on_stack:
                    lowlink[node] = min(lowlink[node], index[nxt])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[node])
            if lowlink[node] == index[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                sccs.append(component)
    return sccs


def detect_loop(rules_dict: dict) -> dict:
    """对单个样本运行 B-EL 检测。

    返回 {"has_anomaly": bool, "reason": str, "involved_rules": [rule_id...]}。
    """
    rules = parse_sample_rules(rules_dict)
    if not rules:
        return {"has_anomaly": False, "reason": "No parseable rules.", "involved_rules": []}

    rule_ids = [rule.rule_id for rule in rules]
    graph = build_propagation_graph(rules)
    sccs = find_sccs(graph, rule_ids)

    loops = []
    for component in sccs:
        if len(component) >= 2:
            loops.append(sorted(component, key=lambda key: int(key.split("_")[1])))
        elif component[0] in graph.get(component[0], set()):
            loops.append(component)  # 自环：规则动作直接重新激活自身触发条件

    if not loops:
        return {"has_anomaly": False, "reason": "No anomalies identified.", "involved_rules": []}

    involved = sorted({rule_id for loop in loops for rule_id in loop}, key=lambda key: int(key.split("_")[1]))
    chain_text = " -> ".join(loops[0] + [loops[0][0]]) if len(loops[0]) > 1 else f"{loops[0][0]} -> {loops[0][0]}"
    reason = (
        f"Rules {', '.join(involved)} form an actuation loop (endless loop): "
        f"{chain_text}. Triggering any rule in this cycle re-fires the others indefinitely."
    )
    return {"has_anomaly": True, "reason": reason, "involved_rules": involved}
