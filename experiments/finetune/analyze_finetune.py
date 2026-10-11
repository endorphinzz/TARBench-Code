"""E9 微调实验分析脚本。

输入  experiments/finetune/outputs/pred_{split}_{tag}.json
      (由云端 infer_all.sh 产出, 命名见 infer_all.sh)
输出  experiments/finetune/analysis/
      metrics/pred_{split}_{tag}_metrics.json   与 evaluate_icl 同 schema
      e9_summary.json                           全部聚合数据 (供写论文引用)
      e9_report.md                              人读报告

功能:
  1. 逐文件调 evaluate_icl.evaluate() 评测 (契约与 ICL/符号基线一致)
  2. M1 三 seeds mean±std 聚合; M2 数据规模 / M3 输出结构消融
  3. base 0-shot vs 各 FT 的 McNemar 精确二项检验 (不依赖 scipy)
  4. 配对 bootstrap 95% CI: dF1 (全样本) 与 dARL-F1 (异常样本)
  5. 分组 recall: rule-rule vs rule-environment / structural vs semantic
     / 14 威胁类型 (Dual/Triple 按 sub_category 方括号组件拆分, 可计入多组)
  6. 对比表参考行: ICL 网关模型 (T2 同集) / 符号基线 / 论文 Table 2 qwen-8b

用法:
  python analyze_finetune.py
  python analyze_finetune.py --bootstrap-b 5000 --bootstrap-tags e9_m1_s1
"""

import argparse
import json
import math
import random
import re
import sys
from datetime import datetime
from pathlib import Path

FT_DIR = Path(__file__).resolve().parent
ICL_DIR = FT_DIR.parent / "icl"
BASELINES_DIR = FT_DIR.parent / "baselines"
sys.path.insert(0, str(ICL_DIR))
import evaluate_icl  # noqa: E402

SPLITS = ["test_t1", "test_t2_icl750", "test_t3_variants"]
SPLIT_DISPLAY = {
    "test_t1": "T1 main",
    "test_t2_icl750": "T2 ICL-750",
    "test_t3_variants": "T3 contextualized (OOD)",
}
FT_TAGS = ["e9_m1_s1", "e9_m1_s2", "e9_m1_s3", "e9_m3_cls", "e9_m2_500", "e9_m2_2000"]
M1_SEEDS = ["e9_m1_s1", "e9_m1_s2", "e9_m1_s3"]
BASE_TAG = "base0shot"
TAG_DISPLAY = {
    "base0shot": "Qwen3-8B base (0-shot)",
    "e9_m1_s1": "E9-FT seed-1",
    "e9_m1_s2": "E9-FT seed-2",
    "e9_m1_s3": "E9-FT seed-3",
    "e9_m3_cls": "E9-FT cls-only (M3)",
    "e9_m2_500": "E9-FT 500 samples (M2)",
    "e9_m2_2000": "E9-FT 2,000 samples (M2)",
    "e9_m1_mean": "E9-FT (3-seed mean)",
}

# ---- 与 compare_llm_vs_symbolic.py 一致的官方分类 ----
RR_TYPES = {
    "Action Conflict", "Action Duplication", "Action Revert", "Loop",
    "Condition Bypass", "Action Mutex", "Illegal State Rule",
}
RE_TYPES = {
    "Safety Boundary Error", "Trigger Mutex", "Action Block", "Condition Block",
    "Common Sense Violation", "Privilege Escalation", "Physical Cross-rule Interaction",
}
SEMANTIC_TYPES = {
    "Common Sense Violation", "Privilege Escalation",
    "Physical Cross-rule Interaction", "Condition Block",
}
BRACKET_RE = re.compile(r"\[(.+)\]")
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

# ---- 参考行: 论文 Table 2 qwen-8b 常量 (Acc,P,R,F1,FPR,ARL-IoU,TESS) ----
PAPER_QWEN8B = {"acc": 0.89, "p": 0.83, "r": 0.83, "f1": 0.83, "fpr": 0.08, "arl": 0.62, "tess": 3.07}
ICL_LLMS = ["gpt-5.4", "deepseek-v4-flash", "gemini-3.1-flash-lite-preview"]
ICL_DISPLAY = {
    "gpt-5.4": "GPT-5.4 (ICL)",
    "deepseek-v4-flash": "DS-v4-flash (ICL)",
    "gemini-3.1-flash-lite-preview": "Gemini-3.1-flash (ICL)",
}
SYMBOLIC_BASELINES = [
    ("el", "Heuristic (loop)"),
    ("ar", "Heuristic (action)"),
    ("ac", "Heuristic (trig.-overlap)"),
    ("tm", "Heuristic (mutex)"),
    ("be", "Heuristic (ontology)"),
    ("smt", "SMT (iRuler-style)"),
    ("lex", "Lexical (TF-IDF+LR)"),
    ("graph", "Graph (features+LR)"),
]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def f1_of(metrics):
    p = metrics["overall"]["precision"]
    r = metrics["overall"]["recall"]
    return round(2 * p * r / (p + r), 6) if (p + r) else 0.0


def std_of(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def mean_pm_std(values, digits=3):
    if not values:
        return "-"
    mean = sum(values) / len(values)
    if len(values) < 2:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f}±{std_of(values):.{digits}f}"


# ============ 统计检验 ============

def correctness(item):
    """二分类正确性: unknown 判错。"""
    truth = item["Ground_Truth"]["has_anomaly"] is True
    pred = evaluate_icl.normalize_bool(item["LLM_Prediction"]["has_anomaly"])
    if pred is None:
        return False
    return pred == truth


def mcnemar_exact(b, c):
    """双侧精确二项检验: b=仅A对, c=仅B对。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def paired_bootstrap(values_a, values_b, b_iters, rng):
    """配对 bootstrap: values_a - values_b 的 95% CI (list of float, 等长对齐)。"""
    n = len(values_a)
    deltas = []
    for _ in range(b_iters):
        idx = [rng.randrange(n) for _ in range(n)]
        da = sum(values_a[i] for i in idx) / n
        db = sum(values_b[i] for i in idx) / n
        deltas.append(da - db)
    deltas.sort()
    mean = sum(deltas) / len(deltas)
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[min(len(deltas) - 1, int(0.975 * len(deltas)))]
    return {"mean": round(mean, 6), "ci95": [round(lo, 6), round(hi, 6)]}


def f1_from_binary(truths, preds01):
    tp = sum(1 for t, p in zip(truths, preds01) if t and p)
    fp = sum(1 for t, p in zip(truths, preds01) if not t and p)
    fn = sum(1 for t, p in zip(truths, preds01) if t and not p)
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 0.0


def bootstrap_detection_f1(pairs, b_iters, rng):
    """pairs: [(truth01_a, pred01_a, truth01_b, pred01_b), ...] 逐样本对齐。"""
    n = len(pairs)
    deltas = []
    for _ in range(b_iters):
        idx = [rng.randrange(n) for _ in range(n)]
        ta = [pairs[i][0] for i in idx]
        pa = [pairs[i][1] for i in idx]
        tb = [pairs[i][2] for i in idx]
        pb = [pairs[i][3] for i in idx]
        deltas.append(f1_from_binary(ta, pa) - f1_from_binary(tb, pb))
    deltas.sort()
    mean = sum(deltas) / len(deltas)
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[min(len(deltas) - 1, int(0.975 * len(deltas)))]
    return {"mean": round(mean, 6), "ci95": [round(lo, 6), round(hi, 6)]}


# ============ 分组 recall (组件级) ============

def truth_components(gt):
    """Ground_Truth -> 组件类型列表。Dual/Triple 从 sub_category 方括号拆分。"""
    atype = gt.get("anomaly_type")
    if not gt.get("has_anomaly"):
        return []
    sub = str(gt.get("sub_category") or "")
    m = BRACKET_RE.search(sub)
    if atype in ("Dual Anomaly", "Triple Anomaly") and m:
        return [c.strip() for c in m.group(1).split("&") if c.strip()]
    if atype:
        return [atype]
    return []


def component_groups(comp):
    groups = []
    if comp in RR_TYPES:
        groups.append("rule_rule")
    if comp in RE_TYPES:
        groups.append("rule_env")
        if comp in SEMANTIC_TYPES:
            groups.append("semantic")
        else:
            groups.append("structural")
    elif comp in RR_TYPES:
        groups.append("structural")
    return groups


def group_recall(items):
    """返回 {rule_rule/rule_env/structural/semantic: {total, tp}} (按组件计数, Dual/Triple 计入多组)。"""
    stats = {k: {"total": 0, "tp": 0} for k in ("rule_rule", "rule_env", "structural", "semantic")}
    type_stats = {}
    for item in items:
        gt = item["Ground_Truth"]
        if gt.get("has_anomaly") is not True:
            continue
        detected = evaluate_icl.normalize_bool(item["LLM_Prediction"]["has_anomaly"]) is True
        atype = gt.get("anomaly_type") or "__unknown__"
        bucket = type_stats.setdefault(atype, {"total": 0, "tp": 0})
        bucket["total"] += 1
        if detected:
            bucket["tp"] += 1
        for comp in truth_components(gt):
            for g in component_groups(comp):
                stats[g]["total"] += 1
                if detected:
                    stats[g]["tp"] += 1
    return stats, type_stats


def recall_of(bucket):
    return round(bucket["tp"] / bucket["total"], 4) if bucket["total"] else None


# ============ 主流程 ============

def evaluate_all(outputs_dir, metrics_dir):
    """返回 {split: {tag: {"items":..., "metrics":..., "enriched":...}}}。"""
    results = {}
    for split in SPLITS:
        found = sorted(outputs_dir.glob(f"pred_{split}_*.json"))
        results[split] = {}
        for path in found:
            tag = path.stem[len(f"pred_{split}_"):]
            items = load_json(path)
            metrics, enriched = evaluate_icl.evaluate(items)
            metrics["f1"] = f1_of(metrics)
            write_json(metrics_dir / f"pred_{split}_{tag}_metrics.json", metrics)
            results[split][tag] = {"items": items, "metrics": metrics, "enriched": enriched}
            print(f"[eval] {split} {tag}: acc={metrics['overall']['accuracy']:.3f} "
                  f"f1={metrics['f1']:.3f} fpr={metrics['overall']['false_positive_rate']:.3f} "
                  f"arl_f1={metrics['abnormal']['localization_all_abnormal']['overlap_f1']:.3f} "
                  f"unknown={metrics['overall']['unknown']}")
    return results


def row_from_metrics(tag, metrics):
    loc = metrics["abnormal"]["localization_all_abnormal"]
    return {
        "tag": tag,
        "n": metrics["overall"]["total"],
        "acc": metrics["overall"]["accuracy"],
        "p": metrics["overall"]["precision"],
        "r": metrics["overall"]["recall"],
        "f1": metrics.get("f1", f1_of(metrics)),
        "fpr": metrics["overall"]["false_positive_rate"],
        "arl_f1": loc["overlap_f1"],
        "loc_em": loc["exact_match_rate"],
        "type_mentioned": metrics["abnormal"]["type_mentioned_rate"],
        "overthinking": metrics["benign"]["overthinking_rate"],
    }


def seeds_aggregate(split_results):
    """M1 三 seeds 聚合: 各指标 mean±std。"""
    rows = [row_from_metrics(tag, split_results[tag]["metrics"]) for tag in M1_SEEDS if tag in split_results]
    if not rows:
        return None
    agg = {}
    for key in ("acc", "p", "r", "f1", "fpr", "arl_f1", "loc_em", "type_mentioned", "overthinking"):
        agg[key] = {
            "mean": round(sum(r[key] for r in rows) / len(rows), 6),
            "std": round(std_of([r[key] for r in rows]), 6),
        }
    agg["n_seeds"] = len(rows)
    return agg


def mcnemar_against_base(split_results, split):
    out = []
    if BASE_TAG not in split_results:
        return out
    base_items = split_results[BASE_TAG]["items"]
    base_correct = {it["Sample_Index"]: correctness(it) for it in base_items}
    for tag in FT_TAGS:
        if tag not in split_results:
            continue
        ft_items = split_results[tag]["items"]
        b = c = 0  # b: base 对 FT 错; c: base 错 FT 对
        for it in ft_items:
            ft_ok = correctness(it)
            base_ok = base_correct.get(it["Sample_Index"], False)
            if base_ok and not ft_ok:
                b += 1
            elif not base_ok and ft_ok:
                c += 1
        p = mcnemar_exact(b, c)
        out.append({"tag": tag, "split": split, "base_better": b, "ft_better": c, "p_value": round(p, 6)})
        sig = "*" if p < 0.05 else ""
        print(f"[mcnemar] {split} base vs {tag}: b={b} c={c} p={p:.4g}{sig}")
    return out


def bootstrap_against_base(split_results, split, tags, b_iters, seed):
    """dF1 (全样本) 与 dARL-F1 (异常样本) 的配对 bootstrap CI。"""
    out = []
    if BASE_TAG not in split_results:
        return out
    base = split_results[BASE_TAG]
    base_map = {it["Sample_Index"]: it for it in base["items"]}
    base_loc = {e["Sample_Index"]: e["Evaluation"].get("localization", {}).get("overlap_f1", 0.0)
                for e in base["enriched"] if e["Evaluation"]["truth_has_anomaly"]}
    for tag in tags:
        if tag not in split_results:
            continue
        ft = split_results[tag]
        pairs = []
        for it in ft["items"]:
            base_it = base_map.get(it["Sample_Index"])
            if base_it is None:
                continue
            pairs.append((
                1 if it["Ground_Truth"]["has_anomaly"] else 0,
                1 if evaluate_icl.normalize_bool(it["LLM_Prediction"]["has_anomaly"]) is True else 0,
                1 if base_it["Ground_Truth"]["has_anomaly"] else 0,
                1 if evaluate_icl.normalize_bool(base_it["LLM_Prediction"]["has_anomaly"]) is True else 0,
            ))
        rng = random.Random(seed)
        d_f1 = bootstrap_detection_f1(pairs, b_iters, rng)
        # dARL: 仅异常样本 (以 FT enriched 的 truth 对齐)
        ft_vals, base_vals = [], []
        for e in ft["enriched"]:
            if not e["Evaluation"]["truth_has_anomaly"]:
                continue
            ft_vals.append(e["Evaluation"].get("localization", {}).get("overlap_f1", 0.0))
            base_vals.append(base_loc.get(e["Sample_Index"], 0.0))
        rng2 = random.Random(seed + 1)
        d_arl = paired_bootstrap(ft_vals, base_vals, min(b_iters, 2000), rng2)
        out.append({"tag": tag, "split": split, "d_f1": d_f1, "d_arl_f1": d_arl})
        print(f"[boot] {split} {tag}: dF1={d_f1['mean']:.4f} [{d_f1['ci95'][0]:.4f},{d_f1['ci95'][1]:.4f}] "
              f"dARL={d_arl['mean']:.4f} [{d_arl['ci95'][0]:.4f},{d_arl['ci95'][1]:.4f}]")
    return out


# ============ 参考行 ============

def icl_reference_rows(shot_keys=("0shot", "5shot")):
    rows = []
    for model in ICL_LLMS:
        for shot in shot_keys:
            path = ICL_DIR / "outputs" / f"{model}_500_250" / f"icl_{model}_{shot}_metrics.json"
            if not path.exists():
                continue
            m = load_json(path)
            row = row_from_metrics(f"icl_{model}_{shot}", m)
            row["display"] = f"{ICL_DISPLAY[model]} {shot}"
            rows.append(row)
    return rows


def symbolic_reference_rows():
    rows = []
    for name, display in SYMBOLIC_BASELINES:
        path = BASELINES_DIR / "outputs" / f"baseline_{name}_full_dataset_metrics.json"
        if not path.exists():
            continue
        m = load_json(path)
        row = row_from_metrics(f"symbolic_{name}", m)
        row["display"] = display
        rows.append(row)
    return rows


# ============ 报告渲染 ============

def fmt_row(row, with_std=None):
    def cell(key):
        if with_std and key in with_std:
            pair = with_std[key]  # [mean, std] 已是聚合结果, 直接格式化
            return f"{pair[0]:.3f}±{pair[1]:.3f}"
        return f"{row[key]:.3f}"
    return (f"| {row.get('display', TAG_DISPLAY.get(row['tag'], row['tag']))} "
            f"| {cell('acc')} | {cell('p')} | {cell('r')} | {cell('f1')} "
            f"| {cell('fpr')} | {cell('arl_f1')} | {cell('loc_em')} "
            f"| {cell('type_mentioned')} | {cell('overthinking')} |")


TABLE_HEADER = ("| System | Acc | P | R | F1 | FPR | ARL-F1 | Loc-EM | TypeM | OTR |\n"
                "|---|---|---|---|---|---|---|---|---|---|")


def render_main_table(split, split_results, summary, symbolic_rows):
    lines = [f"### {SPLIT_DISPLAY[split]} (n={split_results[BASE_TAG]['metrics']['overall']['total'] if BASE_TAG in split_results else '?'})",
             "", TABLE_HEADER, ""]
    if BASE_TAG in split_results:
        row = row_from_metrics(BASE_TAG, split_results[BASE_TAG]["metrics"])
        row["display"] = TAG_DISPLAY[BASE_TAG]
        lines.append(fmt_row(row))
    agg = summary["seeds"].get(split)
    if agg:
        std_map = {k: [agg[k]["mean"], agg[k]["std"]] for k in agg if isinstance(agg[k], dict)}
        mean_row = {k: (agg[k]["mean"] if isinstance(agg[k], dict) else agg[k]) for k in agg if isinstance(agg[k], dict)}
        mean_row["tag"] = "e9_m1_mean"
        mean_row["display"] = "E9-FT (3 seeds, mean±std)"
        lines.append(fmt_row(mean_row, with_std=std_map))
    for tag in ("e9_m3_cls", "e9_m2_500", "e9_m2_2000"):
        if tag in split_results:
            row = row_from_metrics(tag, split_results[tag]["metrics"])
            row["display"] = TAG_DISPLAY[tag]
            lines.append(fmt_row(row))
    lines.append("")
    lines.append(f"| Qwen3-8B (ICL, paper Tab.2) | {PAPER_QWEN8B['acc']:.2f} | {PAPER_QWEN8B['p']:.2f} "
                 f"| {PAPER_QWEN8B['r']:.2f} | {PAPER_QWEN8B['f1']:.2f} | {PAPER_QWEN8B['fpr']:.2f} "
                 f"| {PAPER_QWEN8B['arl']:.2f} | - | - | - |")
    if symbolic_rows and split == "test_t1":
        lines.append("")
        lines.append("符号基线参考 (full-dataset 口径, 非同集, 仅对照):")
        lines.append("")
        for row in symbolic_rows:
            row = {**row, "arl_f1": row.get("arl_f1", 0.0)}
            lines.append(fmt_row(row))
    return lines


def render_group_table(split, split_results, summary):
    lines = [f"### 分组 recall — {SPLIT_DISPLAY[split]}", "",
             "| System | rule-rule | rule-env | structural | semantic |", "|---|---|---|---|---|"]
    tags = [BASE_TAG] + FT_TAGS
    for tag in tags:
        g = summary["groups"].get(split, {}).get(tag)
        if not g:
            continue

        def cell(key):
            rec = g[key]["recall"]
            return f"{rec:.3f} ({g[key]['tp']}/{g[key]['total']})" if rec is not None else "-"

        lines.append(f"| {TAG_DISPLAY.get(tag, tag)} | {cell('rule_rule')} | {cell('rule_env')} "
                     f"| {cell('structural')} | {cell('semantic')} |")
    lines.append("")
    lines.append("注: Dual/Triple Anomaly 按组件计入多组; semantic = CV/PE/PD/CBk, structural = 其余。")
    return lines


def render_type_table(split, split_results, summary):
    lines = [f"### 14 威胁类型 recall — {SPLIT_DISPLAY[split]}", ""]
    header = "| Type | " + " | ".join(TAG_DISPLAY.get(t, t) for t in [BASE_TAG, "e9_m1_mean"] + FT_TAGS[3:]) + " |"
    lines.append(header)
    lines.append("|---|" + "---|" * 5)
    type_data = summary["type_recall"].get(split, {})
    for display, data_name in TYPE_ORDER:
        cells = []
        for tag in [BASE_TAG, "e9_m1_mean", "e9_m3_cls", "e9_m2_500", "e9_m2_2000"]:
            bucket = type_data.get(tag, {}).get(data_name)
            cells.append(f"{bucket['recall']:.2f}" if bucket and bucket["recall"] is not None else "-")
        lines.append(f"| {display} | " + " | ".join(cells) + " |")
    for dual in ("Dual Anomaly", "Triple Anomaly"):
        cells = []
        for tag in [BASE_TAG, "e9_m1_mean", "e9_m3_cls", "e9_m2_500", "e9_m2_2000"]:
            bucket = type_data.get(tag, {}).get(dual)
            if bucket and bucket["recall"] is not None:
                tot = bucket.get("total")
                cells.append(f"{bucket['recall']:.2f} ({tot})" if tot is not None else f"{bucket['recall']:.2f}")
            else:
                cells.append("-")
        lines.append(f"| {dual} | " + " | ".join(cells) + " |")
    return lines


def main():
    parser = argparse.ArgumentParser(description="E9 微调实验分析")
    parser.add_argument("--outputs", type=Path, default=FT_DIR / "outputs")
    parser.add_argument("--analysis-dir", type=Path, default=FT_DIR / "analysis")
    parser.add_argument("--bootstrap-b", type=int, default=2000)
    parser.add_argument("--bootstrap-tags", default="e9_m1_s1,e9_m1_s2,e9_m1_s3")
    parser.add_argument("--bootstrap-split", default="test_t1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metrics_dir = args.analysis_dir / "metrics"
    if not args.outputs.exists():
        print(f"[warn] 预测目录不存在: {args.outputs} (等云端 infer_all.sh 产出后重跑)")
        return

    print("=== 1. 评测全部预测文件 ===")
    results = evaluate_all(args.outputs, metrics_dir)
    total = sum(len(v) for v in results.values())
    if total == 0:
        print("[warn] 无 pred_*.json, 退出")
        return

    print("=== 2. 聚合与统计检验 ===")
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seeds": {}, "rows": {}, "mcnemar": [], "bootstrap": [], "groups": {}, "type_recall": {},
    }
    for split, split_results in results.items():
        if not split_results:
            continue
        agg = seeds_aggregate(split_results)
        if agg:
            summary["seeds"][split] = agg
        summary["rows"][split] = {tag: row_from_metrics(tag, r["metrics"]) for tag, r in split_results.items()}
        summary["mcnemar"].extend(mcnemar_against_base(split_results, split))
        boot_tags = [t.strip() for t in args.bootstrap_tags.split(",") if t.strip()]
        if split == args.bootstrap_split:
            summary["bootstrap"].extend(
                bootstrap_against_base(split_results, split, boot_tags, args.bootstrap_b, args.seed))
        # 分组 recall (逐 tag)
        summary["groups"][split] = {}
        summary["type_recall"][split] = {}
        for tag, r in split_results.items():
            stats, type_stats = group_recall(r["items"])
            summary["groups"][split][tag] = {
                k: {"recall": recall_of(v), **v} for k, v in stats.items()
            }
            summary["type_recall"][split][tag] = {
                name: {"recall": recall_of(b), **b} for name, b in type_stats.items()
            }
        # M1 seeds 的分组 recall 聚合 (mean)
        m1_group = {}
        for comp_key in ("rule_rule", "rule_env", "structural", "semantic"):
            vals = [summary["groups"][split][s][comp_key]["recall"] for s in M1_SEEDS
                    if s in summary["groups"][split] and summary["groups"][split][s][comp_key]["recall"] is not None]
            if vals:
                m1_group[comp_key] = round(sum(vals) / len(vals), 4)
        if m1_group:
            summary["groups"][split]["e9_m1_mean"] = {**m1_group, "recall": m1_group}
        m1_type = {}
        for name in [d for _, d in TYPE_ORDER] + ["Dual Anomaly", "Triple Anomaly"]:
            vals = [summary["type_recall"][split][s].get(name, {}).get("recall") for s in M1_SEEDS
                    if s in summary["type_recall"][split] and summary["type_recall"][split][s].get(name)]
            vals = [v for v in vals if v is not None]
            if vals:
                # total 分母三个 seed 相同 (同测试集), 取任一 seed 的值
                ref = next(summary["type_recall"][split][s][name] for s in M1_SEEDS
                           if s in summary["type_recall"][split] and summary["type_recall"][split][s].get(name))
                m1_type[name] = {"recall": round(sum(vals) / len(vals), 4), "total": ref.get("total")}
        if m1_type:
            summary["type_recall"][split]["e9_m1_mean"] = m1_type

    summary["references"] = {
        "paper_qwen8b_table2": PAPER_QWEN8B,
        "icl_gateway": icl_reference_rows(),
        "symbolic": symbolic_reference_rows(),
    }

    print("=== 3. 渲染报告 ===")
    sym_rows = summary["references"]["symbolic"]
    md = ["# E9 微调实验分析报告", "",
          f"生成时间: {summary['generated_at']}  |  bootstrap B={args.bootstrap_b} (seed={args.seed})",
          "",
          "口径: ARL-F1 = involved-rules 集合 overlap-F1 (异常样本均值); Loc-EM = 精确集合匹配率; "
          "TypeM = reason 提及异常类型比例; OTR = benign 过度思考率。TESS 由 E3 管线另计。", ""]
    for split in SPLITS:
        if split not in results or not results[split]:
            continue
        md.append(f"## {SPLIT_DISPLAY[split]}")
        md.append("")
        md.extend(render_main_table(split, results[split], summary, sym_rows if split == "test_t1" else []))
        md.append("")
        if split == "test_t2_icl750" and summary["references"]["icl_gateway"]:
            md.append("ICL 网关模型参考 (T2 与论文 Table 3 同集, 可同集对比):")
            md.append("")
            md.append(TABLE_HEADER)
            for row in summary["references"]["icl_gateway"]:
                md.append(fmt_row(row))
            md.append("")
        md.extend(render_group_table(split, results[split], summary))
        md.append("")
        md.extend(render_type_table(split, results[split], summary))
        md.append("")
        mcn = [m for m in summary["mcnemar"] if m["split"] == split]
        if mcn:
            md.append(f"### McNemar 精确检验 (base vs FT) — {SPLIT_DISPLAY[split]}")
            md.append("")
            md.append("| FT | base对FT错 (b) | base错FT对 (c) | p | 显著(α=.05) |")
            md.append("|---|---|---|---|---|")
            for m in mcn:
                md.append(f"| {TAG_DISPLAY.get(m['tag'], m['tag'])} | {m['base_better']} | {m['ft_better']} "
                          f"| {m['p_value']:.4g} | {'yes' if m['p_value'] < 0.05 else 'no'} |")
            md.append("")
    boot = summary["bootstrap"]
    if boot:
        md.append(f"## 配对 bootstrap 95% CI — {SPLIT_DISPLAY[args.bootstrap_split]}")
        md.append("")
        md.append("| FT | ΔF1 [95% CI] | ΔARL-F1 [95% CI] |")
        md.append("|---|---|---|")
        for b in boot:
            md.append(f"| {TAG_DISPLAY.get(b['tag'], b['tag'])} "
                      f"| {b['d_f1']['mean']:.3f} [{b['d_f1']['ci95'][0]:.3f}, {b['d_f1']['ci95'][1]:.3f}] "
                      f"| {b['d_arl_f1']['mean']:.3f} [{b['d_arl_f1']['ci95'][0]:.3f}, {b['d_arl_f1']['ci95'][1]:.3f}] |")
        md.append("")

    write_json(args.analysis_dir / "e9_summary.json", summary)
    report = args.analysis_dir / "e9_report.md"
    report.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[done] {report}")
    print(f"[done] {args.analysis_dir / 'e9_summary.json'}")


if __name__ == "__main__":
    main()
