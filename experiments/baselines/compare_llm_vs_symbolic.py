"""生成论文用 LaTeX 表格：LLM vs 非 LLM 基线对比。

威胁分组严格遵循论文 Section 4 官方分类：
  rule-rule (7)：AC, AD, AR, EL, CB, AM, IS
  rule-environment (7)：BE, TM, AB, CBk, CV, PE, PD
Dual/Triple Anomaly 样本按 sub_category 括号内的组件类型计入对应组
（可同时计入两组）。

产出（experiments/baselines/paper_tables/）：
  table_main.tex     正文主表：4 个代表基线 + 符号集成 + 8 个 LLM（全量 15,183）
  table_groups.tex   威胁分组叙事表：rule-rule vs rule-environment recall
                     （ICL 评测集 750：7 基线 + 集成 + 3 个 LLM 0-shot）
  table_appendix.tex 附录全表：7 基线 + 集成 × 14 威胁类型 recall（全量）

LLM 主表数字取自论文 Table 2（tab:mainresults，全量基准运行，
原始输出不在本仓库）；基线数字由本仓库计算，评测集与主表一致。
"""

import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / "outputs"
ICL_METRICS = BASE_DIR.parent / "icl" / "outputs"
PAPER_DIR = BASE_DIR / "paper_tables"
FULL = "full_dataset"
ICL = "icl_dataset"

RULE_RE = re.compile(r"Rule_\d+")
BRACKET_RE = re.compile(r"\[(.+)\]")

# 论文 Section 4 官方分类（数据侧类型名）
RR_TYPES = {
    "Action Conflict", "Action Duplication", "Action Revert", "Loop",
    "Condition Bypass", "Action Mutex", "Illegal State Rule",
}
RE_TYPES = {
    "Safety Boundary Error", "Trigger Mutex", "Action Block", "Condition Block",
    "Common Sense Violation", "Privilege Escalation", "Physical Cross-rule Interaction",
}

# (论文显示名, 数据类型名)，顺序：rule-rule 7 个 + rule-environment 7 个
TYPE_ORDER = [
    ("Action Conflict (AC)", "Action Conflict"),
    ("Action Duplication (AD)", "Action Duplication"),
    ("Action Revert (AR)", "Action Revert"),
    ("Endless Loop (EL)", "Loop"),
    ("Condition Bypass (CB)", "Condition Bypass"),
    ("Action Mutex (AM)", "Action Mutex"),
    ("Illegal State (IS)", "Illegal State Rule"),
    ("Boundary Error (BE)", "Safety Boundary Error"),
    ("Trigger Mutex (TM)", "Trigger Mutex"),
    ("Action Block (AB)", "Action Block"),
    ("Condition Block (CBk)", "Condition Block"),
    ("Common-sense Violation (CV)", "Common Sense Violation"),
    ("Privilege Escalation (PE)", "Privilege Escalation"),
    ("Physical Danger (PD)", "Physical Cross-rule Interaction"),
]

# 论文 Table 2 常量（Model: Acc, P, R, F1, FPR, ARL-IoU, TESS）
PAPER_MAIN = [
    ("gpt-5.4", 0.83, 0.65, 0.97, 0.78, 0.24, 0.76, 3.33),
    ("gpt-5-mini", 0.33, 0.32, 1.00, 0.49, 0.98, 0.77, 3.47),
    ("ds-v3", 0.56, 0.42, 1.00, 0.59, 0.64, 0.71, 3.30),
    ("ds-r1", 0.75, 0.57, 0.84, 0.68, 0.29, 0.72, 3.55),
    ("claude-sonnet", 0.71, 0.53, 0.99, 0.69, 0.42, 0.84, 3.64),
    ("claude-haiku", 0.34, 0.33, 1.00, 0.49, 0.97, 0.65, 3.61),
    ("qwen-8b", 0.89, 0.83, 0.83, 0.83, 0.08, 0.62, 3.07),
    ("qwen-32b", 0.86, 0.95, 0.59, 0.73, 0.01, 0.48, 3.38),
]

ICL_LLMS = ["gpt-5.4", "deepseek-v4-flash", "gemini-3.1-flash-lite-preview"]
ICL_LLM_DISPLAY = {
    "gpt-5.4": "gpt-5.4",
    "deepseek-v4-flash": "ds-v4",
    "gemini-3.1-flash-lite-preview": "gemini-3.1",
}

RULE_LEVEL = ["el", "ar", "ac", "tm", "be", "smt"]
DISPLAY = {
    "el": "Heuristic (loop)",
    "ar": "Heuristic (action)",
    "ac": "Heuristic (trig.-overlap)",
    "tm": "Heuristic (mutex)",
    "be": "Heuristic (ontology)",
    "smt": "SMT (iRuler-style)",
    "lex": "Lexical (TF-IDF+LR)",
    "graph": "Graph (features+LR)",
    "ens": "Symbolic ensemble",
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_predictions(name, stem):
    return load_json(OUT / f"baseline_{name}_{stem}.json")


def load_metrics(name, stem):
    return load_json(OUT / f"baseline_{name}_{stem}_metrics.json")


# ------------------------------------------------ 组件解析（论文分类口径）

def components_of(anomaly_type, sub_category):
    """解析异常样本的组件威胁类型列表。

    Dual/Triple Anomaly 的 sub_category 形如 "Dual Anomaly [X & Y]"，
    括号内即类型名；纯样本直接用 anomaly_type。
    """
    if anomaly_type in ("Dual Anomaly", "Triple Anomaly"):
        match = BRACKET_RE.search(sub_category or "")
        if match:
            return [part.strip() for part in match.group(1).split("&")]
        return []
    return [anomaly_type] if anomaly_type else []


def groups_of(components):
    tags = set()
    for component in components:
        if component in RR_TYPES:
            tags.add("rr")
        if component in RE_TYPES:
            tags.add("re")
    return tags


def build_sub_to_type():
    """从全量预测文件构建 {sub_category: anomaly_type}（供 LLM metrics 聚合用）。"""
    mapping = {}
    for item in load_predictions("ar", FULL):
        if item.get("Label") != "abnormal":
            continue
        gt = item["Ground_Truth"]
        mapping[gt.get("sub_category")] = gt.get("anomaly_type")
    return mapping


# ------------------------------------------------ 预测文件侧指标

def rule_ids_from_prediction(value):
    if isinstance(value, list):
        text = " ".join(str(item) for item in value)
    else:
        text = str(value or "")
    return set(RULE_RE.findall(text))


def rule_range(start_index, length):
    if start_index is None or length is None:
        return set()
    start = int(start_index)
    return {f"Rule_{index}" for index in range(start, start + int(length))}


def ground_truth_involved_rules(item):
    truth = item.get("Ground_Truth", {})
    if truth.get("has_anomaly") is not True:
        return set()
    explicit = rule_ids_from_prediction(truth.get("involved_rules", []))
    if explicit:
        return explicit
    return rule_range(truth.get("abnormal_fragment_start_index"), truth.get("abnormal_fragment_length"))


def mean_arl_iou(items):
    """论文口径：仅在异常样本上求 E_pred/E_true 的 IoU 均值（预测为正常 → 0）。"""
    ious = []
    for item in items:
        if item.get("Label") != "abnormal":
            continue
        pred = rule_ids_from_prediction(item["LLM_Prediction"].get("involved_rules", []))
        truth = ground_truth_involved_rules(item)
        union = pred | truth
        ious.append(len(pred & truth) / len(union) if union else 0.0)
    return sum(ious) / len(ious) if ious else 0.0


def overall_from_items(items):
    """从预测 items 重算混淆矩阵指标（用于 ensemble）。"""
    tp = fp = tn = fn = 0
    for item in items:
        truth = item.get("Label") == "abnormal"
        pred = item["LLM_Prediction"]["has_anomaly"] is True
        if truth and pred:
            tp += 1
        elif truth and not pred:
            fn += 1
        elif not truth and pred:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": (tp + tn) / len(items),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
    }


def group_recall_from_items(items, group_tag):
    """分组 recall（论文分类口径，Dual/Triple 按组件计入）。"""
    tp = fn = 0
    for item in items:
        if item.get("Label") != "abnormal":
            continue
        gt = item["Ground_Truth"]
        if group_tag not in groups_of(components_of(gt.get("anomaly_type"), gt.get("sub_category"))):
            continue
        if item["LLM_Prediction"]["has_anomaly"] is True:
            tp += 1
        else:
            fn += 1
    return tp / (tp + fn) if tp + fn else None


def type_recall_from_items(items, type_name):
    """单类型 recall（Dual/Triple 按组件计入该类型）。"""
    tp = fn = 0
    for item in items:
        if item.get("Label") != "abnormal":
            continue
        gt = item["Ground_Truth"]
        if type_name not in components_of(gt.get("anomaly_type"), gt.get("sub_category")):
            continue
        if item["LLM_Prediction"]["has_anomaly"] is True:
            tp += 1
        else:
            fn += 1
    return (tp / (tp + fn) if tp + fn else None), tp + fn


def build_ensemble(items_by_detector):
    """6 个规则级检测器的并集：任一触发即异常，定位取触发检测器的并。"""
    detector_names = [name for name in RULE_LEVEL if name in items_by_detector]
    reference = items_by_detector[detector_names[0]]
    merged = []
    for index in range(len(reference)):
        base = reference[index]
        firing = [
            items_by_detector[name][index]
            for name in detector_names
            if items_by_detector[name][index]["LLM_Prediction"]["has_anomaly"] is True
        ]
        involved = set()
        for item in firing:
            involved |= rule_ids_from_prediction(item["LLM_Prediction"].get("involved_rules", []))
        reasons = [item["LLM_Prediction"]["reason"] for item in firing]
        merged.append(
            {
                **base,
                "LLM_Prediction": {
                    "has_anomaly": bool(firing),
                    "reason": " ".join(reasons) if reasons else "No anomalies identified.",
                    "involved_rules": sorted(involved, key=lambda key: int(key.split("_")[1])),
                },
            }
        )
    return merged


# ------------------------------------------------ LLM metrics 侧聚合

def llm_group_recall(metrics, group_tag, sub_to_type):
    """从 metrics 的 sub_category 分组聚合出分组 recall（论文分类口径）。"""
    tp = fn = 0
    for sub_category, counts in metrics["groups"]["sub_category"].items():
        if sub_category == "__benign__":
            continue
        anomaly_type = sub_to_type.get(sub_category)
        if group_tag not in groups_of(components_of(anomaly_type, sub_category)):
            continue
        tp += counts["tp"]
        fn += counts["fn"]
    return tp / (tp + fn) if tp + fn else None


# ------------------------------------------------ LaTeX 输出工具

def fmt(value, bold=False, underline=False):
    text = f"{value:.2f}"
    if bold:
        return r"\textbf{" + text + "}"
    if underline:
        return r"\underline{" + text + "}"
    return text


def mark_best(rows, col_count, lower_better_cols=()):
    """返回 {row_index: {col: 'bold'/'underline'}}，按列取最优/次优。

    并列最优全部加粗；并列次优（严格次于最优的值）全部下划线。
    lower_better_cols：越小越好的列索引（如 FPR）。
    """
    marks = {}
    for col in range(col_count):
        values = []
        for row_index, row in enumerate(rows):
            value = row[col]
            if value is None or isinstance(value, str):
                continue
            values.append((round(value, 2), row_index))  # 按显示精度判定并列
        if not values:
            continue
        distinct = sorted({value for value, _ in values}, reverse=col not in lower_better_cols)
        best_value = distinct[0]
        for value, row_index in values:
            if value == best_value:
                marks.setdefault(row_index, {})[col] = "bold"
        if len(distinct) > 1:
            second_value = distinct[1]
            for value, row_index in values:
                if value == second_value:
                    marks.setdefault(row_index, {})[col] = "underline"
    return marks


def baseline_row(name, stem, items=None):
    """返回 (display, acc, P, R, F1, FPR, arl_iou, None)；TESS 恒为 None（无解释）。"""
    if items is None:
        items = load_predictions(name, stem)
        o = load_metrics(name, stem)["overall"]
        tp, fp, fn = o["tp"], o["fp"], o["fn"]
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    else:
        o = overall_from_items(items)
        f1 = o["f1"]
    fpr = o.get("false_positive_rate", o.get("fpr", 0.0))
    return (
        DISPLAY[name], o["accuracy"], o["precision"], o["recall"],
        f1, fpr, mean_arl_iou(items), None,
    )


# ---------------------------------------------------------------- 主表

def emit_main_table():
    rows = []
    for name in ["smt", "ar", "lex"]:
        rows.append(baseline_row(name, FULL))
    ens_items = build_ensemble({name: load_predictions(name, FULL) for name in RULE_LEVEL})
    rows.append(baseline_row("ens", FULL, items=ens_items))
    for entry in PAPER_MAIN:
        rows.append(entry)

    # 7 列数值（acc,P,R,F1,FPR,IoU；FPR 越小越好）+ TESS 第 8 列（仅 LLM 行有值）
    marks = mark_best([row[:7] for row in rows], 7, lower_better_cols=(5,))
    tess_candidates = [(row[7], index) for index, row in enumerate(rows) if row[7] is not None]
    tess_candidates.sort(reverse=True)
    tess_map = {tess_candidates[0][1]: "bold"}
    if len(tess_candidates) > 1 and tess_candidates[1][0] != tess_candidates[0][0]:
        tess_map[tess_candidates[1][1]] = "underline"

    lines = [
        r"\begin{table*}[h]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Overall metrics of non-LLM baselines and LLMs on TARBench.}",
        r"\label{tab:mainresults}",
        r"\begin{tabular}{lccccccc}",
        r"\hline",
        r"Model & Accuracy & Precision & Recall & F1 score & FPR & ARL-IoU & TESS \\",
        r"\hline",
    ]
    baseline_count = 4
    for row_index, row in enumerate(rows):
        cells = [row[0]]
        for col in range(1, 7):
            style = marks.get(row_index, {}).get(col)
            cells.append(fmt(row[col], bold=style == "bold", underline=style == "underline"))
        tess = row[7]
        if tess is None:
            cells.append("--")
        else:
            style = tess_map.get(row_index)
            cells.append(fmt(tess, bold=style == "bold", underline=style == "underline"))
        lines.append(" & ".join(cells) + r" \\")
        if row_index == baseline_count - 1:
            lines.append(r"\hline")
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- 分组叙事表

def emit_groups_table():
    stem = ICL
    sub_to_type = build_sub_to_type()
    rows = []
    for name in RULE_LEVEL + ["lex", "graph"]:
        m = load_metrics(name, stem)
        items = load_predictions(name, stem)
        rows.append(
            (DISPLAY[name],
             group_recall_from_items(items, "rr"), group_recall_from_items(items, "re"),
             m["overall"]["recall"], m["overall"]["false_positive_rate"])
        )
    ens_items = build_ensemble({name: load_predictions(name, stem) for name in RULE_LEVEL})
    ens_overall = overall_from_items(ens_items)
    rows.append(
        (DISPLAY["ens"],
         group_recall_from_items(ens_items, "rr"), group_recall_from_items(ens_items, "re"),
         ens_overall["recall"], ens_overall["fpr"])
    )
    for model in ICL_LLMS:
        m = load_json(ICL_METRICS / f"{model}_500_250" / f"icl_{model}_0shot_metrics.json")
        rows.append(
            (ICL_LLM_DISPLAY[model],
             llm_group_recall(m, "rr", sub_to_type), llm_group_recall(m, "re", sub_to_type),
             m["overall"]["recall"], m["overall"]["false_positive_rate"])
        )

    marks = mark_best([row[1:] for row in rows], 4, lower_better_cols=(3,))
    # 单栏显示：方法名用短标签（与附录表 SHORT_DISPLAY 一致）
    SINGLE_COL_DISPLAY = {
        "el": "Loop",
        "ar": "Action",
        "ac": "Trig.",
        "tm": "Mutex",
        "be": "Ont.",
        "smt": "SMT",
        "lex": "Lex",
        "graph": "Graph",
        "ens": "Ensemble",
    }
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Recall on rule-rule ($R_{rr}$) vs.\ rule-environment ($R_{re}$) "
        r"threats (750-sample ICL set; LLMs zero-shot). Heuristic baselines are named "
        r"after the structural property they check "
        r"(Appendix~\ref{app:baselines}); Dual/Triple-anomaly samples count toward "
        r"each component group.}",
        r"\label{tab:groups}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Method & $R_{rr}$ & $R_{re}$ & $R$ & FPR \\",
        r"\hline",
    ]
    for row_index, row in enumerate(rows):
        cells = [SINGLE_COL_DISPLAY.get(row[0], row[0])]
        for col in range(1, 5):
            style = marks.get(row_index, {}).get(col - 1)
            cells.append(fmt(row[col], bold=style == "bold", underline=style == "underline"))
        lines.append(" & ".join(cells) + r" \\")
        if row_index == len(rows) - 4:  # 基线块与 LLM 块之间加分隔线
            lines.append(r"\hline")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------- 附录全表

# 附录表用短列头（ACL table* 全宽 455pt 放不下完整名称），图注中给出对应关系
SHORT_DISPLAY = {
    "el": "Loop",
    "ar": "Action",
    "ac": "Trig.",
    "tm": "Mutex",
    "be": "Ont.",
    "smt": "SMT",
    "lex": "Lex",
    "graph": "Graph",
    "ens": "Ens.",
}


def emit_appendix_table():
    stem = FULL
    columns = RULE_LEVEL + ["lex", "graph"]
    items_by_name = {name: load_predictions(name, stem) for name in columns}
    ens_items = build_ensemble({name: load_predictions(name, stem) for name in RULE_LEVEL})

    lines = [
        r"\begin{table*}[p]",
        r"\centering",
        r"\small",
        r"\caption{Per-threat-type recall of non-LLM baselines on the full benchmark "
        r"(15,183 samples). A sample counts toward each of its component threat types "
        r"(Dual/Triple anomalies included). Heuristic baselines are named after the "
        r"property they check: loop structure (Loop), action-device matching (Action), "
        r"trigger overlap (Trig.), trigger mutex (Mutex), and ontology bounds (Ont.). "
        r"SMT is the Z3-based symbolic baseline (iRuler-style); Lex is the TF-IDF+LR "
        r"classifier; Graph is a rule-interaction-graph feature classifier (logistic "
        r"regression, 5-fold cross-validation); Ens.\ is the union of the six rule-level detectors.}",
        r"\label{tab:appendixbaselines}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "c" * (len(columns) + 2) + "}",
        r"\hline",
        r"Threat type & $n$ & " + " & ".join(SHORT_DISPLAY[name] for name in columns)
        + r" & " + SHORT_DISPLAY["ens"] + r" \\",
        r"\hline",
    ]
    for display_name, type_name in TYPE_ORDER:
        recall_by_name = {}
        n = None
        for name in columns:
            recall, count = type_recall_from_items(items_by_name[name], type_name)
            recall_by_name[name] = recall
            n = count
        ens_recall, _ = type_recall_from_items(ens_items, type_name)
        if not n:
            continue
        cells = [display_name, str(n)]
        for name in columns:
            recall = recall_by_name[name]
            cells.append(f"{recall:.2f}" if recall is not None else "--")
        cells.append(f"{ens_recall:.2f}" if ens_recall is not None else "--")
        lines.append(" & ".join(cells) + r" \\")

    # 分组汇总行
    lines.append(r"\hline")
    for label, tag in (("Rule-rule (all)", "rr"), ("Rule-environment (all)", "re")):
        cells = [label, ""]
        for name in columns:
            recall = group_recall_from_items(items_by_name[name], tag)
            cells.append(f"{recall:.2f}" if recall is not None else "--")
        ens_recall = group_recall_from_items(ens_items, tag)
        cells.append(f"{ens_recall:.2f}" if ens_recall is not None else "--")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def main():
    PAPER_DIR.mkdir(exist_ok=True)
    tables = {
        "table_main.tex": emit_main_table,
        "table_groups.tex": emit_groups_table,
        "table_appendix.tex": emit_appendix_table,
    }
    for filename, emitter in tables.items():
        content = emitter()
        (PAPER_DIR / filename).write_text(content + "\n", encoding="utf-8")
        print(f"wrote {PAPER_DIR / filename}")


if __name__ == "__main__":
    main()
