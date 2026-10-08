"""B-GRAPH：规则交互图基线（图特征 + 逻辑回归）。

把每个样本的规则集合解析为交互图：
  - 属性节点：device.attribute（触发条件读、动作写）
  - 规则节点：每条规则
  - 边：规则--属性（读/写）、规则->规则传播边（A 写的属性被 B 读）
提取纯图论特征（共享属性、写-写冲突对、传播密度、强连通分量、
最长传播链、触发相似度等），用与 B-LEX 完全相同的 5 折分层 CV 管线
（固定 seed）输出样本外预测，并提供基于冲突结构的规则定位。

设计动机：rule-rule 结构威胁（冲突/环/互斥）在交互图上有明确结构
签名，而语义类威胁（安全边界、常识违背）没有——用于检验
structural vs semantic 叙事（模拟审稿 Q4 图结构基线）。
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from rule_parser import parse_sample_rules, action_activates_condition, parse_number

CV_SEED = 20260513  # 与 B-LEX 对齐的固定 CV seed


# ---------------------------------------------------------------- 图构建

def build_propagation_edges(rules):
    """规则级传播边：A 的动作写的属性被 B 的条件读 -> A->B。"""
    edges = []
    for a in rules:
        for b in rules:
            if a is b:
                continue
            for action in a.actions:
                if any(cond.target == action.target for cond in b.conditions):
                    edges.append((a.rule_id, b.rule_id, action.target))
                    break
    return edges


def graph_stats(rules):
    """从解析后的规则列表提取图统计量。"""
    n = len(rules)
    rule_by_id = {r.rule_id: r for r in rules}

    write_targets = {r.rule_id: {a.target for a in r.actions} for r in rules}
    read_targets = {r.rule_id: {c.target for c in r.conditions} for r in rules}
    all_targets = set().union(*write_targets.values(), *read_targets.values()) if rules else set()

    # 写-写冲突对（同 target：不同值 -> clash；相同值 -> duplication）
    ww_clash_pairs = 0
    ww_same_pairs = 0
    for i, a in enumerate(rules):
        for b in rules[i + 1:]:
            for target in write_targets[a.rule_id] & write_targets[b.rule_id]:
                va = {act.value for act in a.actions if act.target == target}
                vb = {act.value for act in b.actions if act.target == target}
                if va & vb:
                    ww_same_pairs += 1
                else:
                    ww_clash_pairs += 1

    # 读-写对（A 写 X 且 B 读 X）
    rw_pair_count = sum(
        1 for a in rules for b in rules
        if a.rule_id != b.rule_id and write_targets[a.rule_id] & read_targets[b.rule_id]
    )
    prop_edges = build_propagation_edges(rules)

    # 有向图（传播边）上的结构：SCC / 2-cycle / 最长链
    adj = {r.rule_id: [] for r in rules}
    for src, dst, _ in prop_edges:
        adj[src].append(dst)
    sccs = _tarjan_scc(adj)
    scc_gt1 = [s for s in sccs if len(s) > 1]
    two_cycles = sum(
        1 for src, dst, _ in prop_edges
        if src < dst and any(d == src and s == dst for s, d, _ in prop_edges)
    )
    longest_chain = _longest_dag_path(adj)

    # 单规则自读自写（潜在 Action Revert 签名）
    self_rw = sum(1 for r in rules if write_targets[r.rule_id] & read_targets[r.rule_id])

    # 触发条件 Jaccard 相似度均值（规则对）
    jaccards = []
    for i, a in enumerate(rules):
        for b in rules[i + 1:]:
            union = read_targets[a.rule_id] | read_targets[b.rule_id]
            if union:
                jaccards.append(len(read_targets[a.rule_id] & read_targets[b.rule_id]) / len(union))

    # 属性共享度
    write_count = {}
    read_count = {}
    for r in rules:
        for t in write_targets[r.rule_id]:
            write_count[t] = write_count.get(t, 0) + 1
        for t in read_targets[r.rule_id]:
            read_count[t] = read_count.get(t, 0) + 1
    shared_write = sum(1 for c in write_count.values() if c >= 2)
    shared_rw = sum(1 for t in all_targets if write_count.get(t, 0) >= 1 and read_count.get(t, 0) >= 1)

    return {
        "n_rules": n,
        "n_targets": len(all_targets),
        "prop_edges": len(prop_edges),
        "prop_density": len(prop_edges) / (n * (n - 1)) if n > 1 else 0.0,
        "ww_clash_pairs": ww_clash_pairs,
        "ww_same_pairs": ww_same_pairs,
        "rw_pairs": rw_pair_count,
        "shared_write_targets": shared_write,
        "shared_rw_targets": shared_rw,
        "self_rw_rules": self_rw,
        "two_cycles": two_cycles,
        "scc_gt1_count": len(scc_gt1),
        "largest_scc_frac": max((len(s) for s in sccs), default=0) / n if n else 0.0,
        "longest_chain": longest_chain,
        "mean_triggers": np.mean([len(r.conditions) for r in rules]) if rules else 0.0,
        "mean_actions": np.mean([len(r.actions) for r in rules]) if rules else 0.0,
        "trigger_jaccard_mean": float(np.mean(jaccards)) if jaccards else 0.0,
    }


FEATURE_NAMES = [
    "n_rules", "n_targets", "prop_edges", "prop_density",
    "ww_clash_pairs", "ww_same_pairs", "rw_pairs",
    "shared_write_targets", "shared_rw_targets", "self_rw_rules",
    "two_cycles", "scc_gt1_count", "largest_scc_frac", "longest_chain",
    "mean_triggers", "mean_actions", "trigger_jaccard_mean",
]


def sample_features(rules_dict: dict) -> list:
    rules = parse_sample_rules(rules_dict)
    stats = graph_stats(rules)
    return [float(stats[name]) for name in FEATURE_NAMES]


# ---------------------------------------------------------------- 图算法

def _tarjan_scc(adj):
    index_counter = [0]
    stack, lowlink, index, on_stack = [], {}, {}, {}
    result = []

    def strongconnect(node):
        index[node] = lowlink[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)
        on_stack[node] = True
        for nxt in adj.get(node, []):
            if nxt not in index:
                strongconnect(nxt)
                lowlink[node] = min(lowlink[node], lowlink[nxt])
            elif on_stack.get(nxt):
                lowlink[node] = min(lowlink[node], index[nxt])
        if lowlink[node] == index[node]:
            component = []
            while True:
                nxt = stack.pop()
                on_stack[nxt] = False
                component.append(nxt)
                if nxt == node:
                    break
            result.append(component)

    for node in adj:
        if node not in index:
            strongconnect(node)
    return result


def _longest_dag_path(adj):
    """SCC 压缩后 DAG 上的最长路径（边数）。"""
    sccs = _tarjan_scc(adj)
    comp_of = {}
    for ci, comp in enumerate(sccs):
        for node in comp:
            comp_of[node] = ci
    comp_adj = {ci: set() for ci in range(len(sccs))}
    for src, dsts in adj.items():
        for dst in dsts:
            if comp_of[src] != comp_of[dst]:
                comp_adj[comp_of[src]].add(comp_of[dst])
    indeg = {ci: 0 for ci in comp_adj}
    for src, dsts in comp_adj.items():
        for dst in dsts:
            indeg[dst] += 1
    queue = [ci for ci, d in indeg.items() if d == 0]
    dist = {ci: 0 for ci in comp_adj}
    order = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for nxt in comp_adj[node]:
            dist[nxt] = max(dist[nxt], dist[node] + 1)
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    return max(dist.values(), default=0)


# ---------------------------------------------------------------- 定位

def _mutex_conditions(rule) -> bool:
    """规则内互斥触发：同 target 的条件值域不相交。"""
    by_target = {}
    for cond in rule.conditions:
        by_target.setdefault(cond.target, []).append(cond)
    for conds in by_target.values():
        for i, c1 in enumerate(conds):
            for c2 in conds[i + 1:]:
                if c1.op == "==" and c2.op == "==" and c1.value != c2.value:
                    return True
                n1, n2 = parse_number(c1.value), parse_number(c2.value)
                if n1 is not None and n2 is not None:
                    if (c1.op in (">", ">=") and c2.op in ("<", "<=") and n1 > n2) or (
                        c1.op in ("<", "<=") and c2.op in (">", ">=") and n1 < n2
                    ):
                        return True
    return False


def localize_anomalous_rules(rules_dict: dict) -> list:
    """基于冲突结构的规则定位：互斥触发 / 写-写冲突 / 传播环 / 自读自写。"""
    rules = parse_sample_rules(rules_dict)
    involved = set()

    for rule in rules:
        if _mutex_conditions(rule):
            involved.add(rule.rule_id)

    write_targets = {r.rule_id: {a.target for a in r.actions} for r in rules}
    for i, a in enumerate(rules):
        for b in rules[i + 1:]:
            if write_targets[a.rule_id] & write_targets[b.rule_id]:
                involved.update({a.rule_id, b.rule_id})

    adj = {r.rule_id: [] for r in rules}
    for src, dst, _ in build_propagation_edges(rules):
        adj[src].append(dst)
    for comp in _tarjan_scc(adj):
        if len(comp) > 1:
            involved.update(comp)

    # 自读自写且动作使自身条件失效（单规则 Action Revert 签名）
    for rule in rules:
        for action in rule.actions:
            for cond in rule.conditions:
                if cond.target == action.target and not action_activates_condition(action, cond):
                    involved.add(rule.rule_id)

    return sorted(involved, key=lambda key: int(key.split("_")[1]))


# ---------------------------------------------------------------- CV 管线

def run_graph_baseline(samples: list, n_folds: int = 5, seed: int = CV_SEED) -> dict:
    """对整个评测集运行 5 折 CV 图特征基线（与 B-LEX 平行）。

    返回 predictions / cv_auc_mean / cv_auc_std / fold_aucs / feature_weights。
    """
    features = np.array([sample_features(sample["rules"]) for sample in samples])
    labels = np.array([1 if sample.get("label") == "abnormal" else 0 for sample in samples])
    sample_ids = [sample.get("sample_id") for sample in samples]

    probabilities = np.zeros(len(samples))
    fold_aucs = []
    fold_coefs = []

    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_index, test_index in splitter.split(features, labels):
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=2000, C=1.0)),
            ]
        )
        model.fit(features[train_index], labels[train_index])
        probabilities[test_index] = model.predict_proba(features[test_index])[:, 1]
        if len(np.unique(labels[test_index])) > 1:
            fold_aucs.append(float(roc_auc_score(labels[test_index], probabilities[test_index])))
        fold_coefs.append(model.named_steps["lr"].coef_[0])

    predictions = {
        sample_id: {
            "has_anomaly": bool(probabilities[position] >= 0.5),
            "probability": float(probabilities[position]),
        }
        for position, sample_id in enumerate(sample_ids)
    }

    auc_array = np.array(fold_aucs) if fold_aucs else np.array([0.5])
    coef_mean = np.mean(fold_coefs, axis=0) if fold_coefs else np.zeros(len(FEATURE_NAMES))
    feature_weights = sorted(
        ({"feature": name, "weight": round(float(coef_mean[i]), 4)} for i, name in enumerate(FEATURE_NAMES)),
        key=lambda item: abs(item["weight"]),
        reverse=True,
    )
    return {
        "predictions": predictions,
        "cv_auc_mean": float(auc_array.mean()),
        "cv_auc_std": float(auc_array.std()),
        "fold_aucs": fold_aucs,
        "feature_weights": feature_weights,
    }
