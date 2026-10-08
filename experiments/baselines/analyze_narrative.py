"""分组叙事分析：rule-rule vs rule-environment 两类威胁在符号基线/LLM 上的表现对比。

rule-rule（结构性，符号可判定）：
  Action Conflict / Action Revert / Action Duplication / Loop / Trigger Mutex / Action Mutex
  / Condition Bypass / Condition Block / Illegal State Rule
rule-environment（常识/物理，需 LLM 常识）：
  Common Sense Violation / Privilege Escalation / Physical Cross-rule Interaction / Safety Boundary Error

输出每个主类在各基线 + LLM(without_rag) 上的 recall / FPR / loc-F1。
"""
import json
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent / "outputs"
RAG_OUTS = Path(__file__).resolve().parent.parent / "rag" / "outputs"

RULE_RULE = {
    "Action Conflict", "Action Revert", "Action Duplication", "Loop",
    "Trigger Mutex", "Action Mutex", "Condition Bypass", "Condition Block",
    "Illegal State Rule",
}
RULE_ENV = {
    "Common Sense Violation", "Privilege Escalation", "Physical Cross-rule Interaction",
    "Safety Boundary Error",
}


def load_metrics(name, stem="rag_dataset"):
    path = OUT / f"baseline_{name}_{stem}_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def llm_metrics(model):
    # without_rag 变体最接近 0-shot 主表设置
    path = RAG_OUTS / f"{model}_500_250" / f"rag_{model}_without_rag_metrics.json"
    if not path.exists():
        # 尝试扁平命名
        path = RAG_OUTS / f"rag_{model}_without_rag_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def group_recall(metrics, group):
    """按 anomaly_type 分组聚合 recall：组内异常总数 / 组内 TP 数。"""
    if metrics is None:
        return None
    groups = metrics["groups"]["anomaly_type"]
    tp = fn = 0
    for type_name, counts in groups.items():
        if type_name in group:
            tp += counts["tp"]
            fn += counts["fn"]
    return tp / (tp + fn) if (tp + fn) else None


def group_fpr(metrics):
    """整体 FPR（按 benign 计算）。"""
    if metrics is None:
        return None
    o = metrics["overall"]
    return o["false_positive_rate"]


def group_loc_f1(metrics, group):
    """组内定位 F1（对已检测的异常样本）。"""
    if metrics is None:
        return None
    # metrics 只聚合了子类级 localization；这里用 sample 级重算太贵，
    # 退而用整体 localization_all_abnormal 作为参考
    return metrics["abnormal"]["localization_all_abnormal"]["overlap_f1"]


names = {
    "el": "B-EL", "ar": "B-AR", "smt": "B-SMT", "lex": "B-LEX",
}
models = ["claude-sonnet-4-5", "deepseek-v4-flash"]

print("=== 分组 recall（anomaly_type 聚合） ===\n")
header = f"{'方法':28s} {'rule-rule R':>12s} {'rule-env R':>12s} {'整体FPR':>10s}"
print(header)
for key, label in names.items():
    m = load_metrics(key)
    if m is None:
        continue
    rr = group_recall(m, RULE_RULE)
    re_ = group_recall(m, RULE_ENV)
    print(f"{label:28s} {rr:12.3f} {re_:12.3f} {group_fpr(m):10.3f}")
print("-" * len(header))
for model in models:
    m = llm_metrics(model)
    if m is None:
        print(f"{model:28s} (metrics 未找到)")
        continue
    rr = group_recall(m, RULE_RULE)
    re_ = group_recall(m, RULE_ENV)
    print(f"{model:28s} {rr:12.3f} {re_:12.3f} {group_fpr(m):10.3f}")

print("\n=== 符号基线在 rule-rule 各 Type 的 recall（细看） ===")
type_order = [
    "Action Conflict", "Action Revert", "Action Duplication", "Loop",
    "Trigger Mutex", "Action Mutex", "Condition Bypass", "Condition Block",
    "Illegal State Rule", "Common Sense Violation", "Privilege Escalation",
    "Physical Cross-rule Interaction", "Safety Boundary Error",
]
header = f"{'Type':28s}" + "".join(f"{names[k]:>10s}" for k in ["el", "ar", "smt"])
print(header)
for type_name in type_order:
    cells = []
    for k in ["el", "ar", "smt"]:
        m = load_metrics(k)
        g = m["groups"]["anomaly_type"].get(type_name) if m else None
        cells.append(f"{g['recall']:10.3f}" if g and (g['tp'] + g['fn']) else "      ---")
    print(f"{type_name:28s}" + "".join(cells))
