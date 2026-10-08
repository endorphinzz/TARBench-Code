"""非 LLM 基线统一入口。

用法：
    python run_baselines.py --detector all --dataset ../datasets/icl_dataset.json
    python run_baselines.py --detector smt,el --dataset ../datasets/rag_dataset.json

输出与 ICL 基准完全兼容的预测 JSON（outputs/baseline_{det}_{stem}.json），
随后自动调用 ../icl/evaluate_icl.py 生成 _metrics.json，供论文与 LLM 主表
同表比较（二分类混淆矩阵、FPR、分子类 recall、ARL-IoU 定位指标）。
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from detector_ac import detect_action_conflict_heuristic
from detector_ar import detect_action_conflicts
from detector_be import detect_boundary_errors
from detector_el import detect_loop
from detector_graph import run_graph_baseline, localize_anomalous_rules
from detector_lex import run_lexical_baseline
from detector_smt import detect_smt
from detector_tm import detect_trigger_mutex_heuristic

BASE_DIR = Path(__file__).resolve().parent
EVALUATE_SCRIPT = BASE_DIR.parent / "icl" / "evaluate_icl.py"

DETECTORS = {
    "el": "baseline-el",
    "ar": "baseline-ar",
    "ac": "baseline-ac",
    "tm": "baseline-tm",
    "be": "baseline-be",
    "smt": "baseline-smt",
    "lex": "baseline-lex",
    "graph": "baseline-graph",
}

RULE_DETECTORS = {
    "el": detect_loop,
    "ar": detect_action_conflicts,
    "ac": detect_action_conflict_heuristic,
    "tm": detect_trigger_mutex_heuristic,
    "be": detect_boundary_errors,
    "smt": detect_smt,
}


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def format_rules_for_output(sample):
    return [f"{rule_id}: {rule_text}" for rule_id, rule_text in sample["rules"].items()]


def make_item(detector_name, sample, index, prediction):
    """构造与 ICL 输出对齐的评测 item。"""
    return {
        "Model": DETECTORS[detector_name],
        "Shot_Count": 0,
        "Sample_Index": index,
        "Case_ID": sample.get("chain_id"),
        "Label": sample.get("label"),
        "Scenario_Group": sample.get("scenario_group"),
        "Chain_Length": sample.get("chain_length"),
        "Rules": format_rules_for_output(sample),
        "Ground_Truth": sample.get("ground_truth", {}),
        "LLM_Prediction": {
            "has_anomaly": prediction["has_anomaly"],
            "reason": prediction["reason"],
            "involved_rules": prediction["involved_rules"],
        },
    }


def run_detector_on_dataset(detector_name, samples):
    """逐样本运行规则级检测器（el/ar/ac/tm/be/smt）。"""
    detect = RULE_DETECTORS[detector_name]
    items = []
    for index, sample in enumerate(samples, start=1):
        prediction = detect(sample["rules"])
        items.append(make_item(detector_name, sample, index, prediction))
    return items


def run_lex_on_dataset(samples):
    """词面基线：5 折 CV 样本外预测。"""
    result = run_lexical_baseline(samples)
    items = []
    for index, sample in enumerate(samples, start=1):
        prediction = result["predictions"][sample["sample_id"]]
        items.append(
            make_item(
                "lex",
                sample,
                index,
                {
                    "has_anomaly": prediction["has_anomaly"],
                    "reason": (
                        f"Lexical classifier (TF-IDF + logistic regression, 5-fold CV) predicts "
                        f"anomaly with probability {prediction['probability']:.3f}. "
                        f"CV AUC = {result['cv_auc_mean']:.3f} +/- {result['cv_auc_std']:.3f}."
                    ),
                    "involved_rules": [],  # 词面基线不提供规则定位
                },
            )
        )
    print(f"  [lex] 5-fold CV AUC: {result['cv_auc_mean']:.4f} +/- {result['cv_auc_std']:.4f}")
    return items


def run_graph_on_dataset(samples):
    """图特征基线：5 折 CV 样本外预测 + 冲突结构定位。"""
    result = run_graph_baseline(samples)
    items = []
    for index, sample in enumerate(samples, start=1):
        prediction = result["predictions"][sample["sample_id"]]
        involved = localize_anomalous_rules(sample["rules"]) if prediction["has_anomaly"] else []
        items.append(
            make_item(
                "graph",
                sample,
                index,
                {
                    "has_anomaly": prediction["has_anomaly"],
                    "reason": (
                        f"Graph classifier (rule interaction graph features + logistic regression, "
                        f"5-fold CV) predicts anomaly with probability {prediction['probability']:.3f}. "
                        f"CV AUC = {result['cv_auc_mean']:.3f} +/- {result['cv_auc_std']:.3f}."
                    ),
                    "involved_rules": involved,
                },
            )
        )
    print(f"  [graph] 5-fold CV AUC: {result['cv_auc_mean']:.4f} +/- {result['cv_auc_std']:.4f}")
    print(f"  [graph] top features: {result['feature_weights'][:5]}")
    return items


def evaluate_output(output_path):
    """调用 ICL 评测脚本生成指标。"""
    command = [sys.executable, str(EVALUATE_SCRIPT), str(output_path)]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise RuntimeError(f"Evaluation failed for {output_path}")
    print(f"  metrics -> {output_path.with_name(output_path.stem + '_metrics.json')}")


def main():
    parser = argparse.ArgumentParser(description="Run non-LLM baselines on TARBench.")
    parser.add_argument(
        "--detector",
        default="all",
        help="逗号分隔的检测器列表：el, ar, smt, lex 或 all",
    )
    parser.add_argument(
        "--dataset",
        default=str(BASE_DIR.parent / "datasets" / "icl_dataset.json"),
        help="评测数据集 JSON 路径（与 LLM 实验对齐）",
    )
    parser.add_argument("--output-dir", default=str(BASE_DIR / "outputs"))
    parser.add_argument("--skip-eval", action="store_true", help="只生成预测，不调用评测脚本")
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    dataset = load_json(dataset_path)
    samples = dataset["samples"]
    stem = dataset_path.stem
    print(f"Dataset: {dataset_path} ({len(samples)} samples)")

    names = list(DETECTORS) if args.detector == "all" else [n.strip() for n in args.detector.split(",")]
    unknown = [n for n in names if n not in DETECTORS]
    if unknown:
        raise SystemExit(f"Unknown detectors: {unknown}. Available: {', '.join(DETECTORS)}")

    output_dir = Path(args.output_dir)
    for name in names:
        print(f"Running baseline: {DETECTORS[name]}")
        if name == "lex":
            items = run_lex_on_dataset(samples)
        elif name == "graph":
            items = run_graph_on_dataset(samples)
        else:
            items = run_detector_on_dataset(name, samples)
        output_path = output_dir / f"baseline_{name}_{stem}.json"
        write_json(output_path, items)
        print(f"  predictions -> {output_path} ({len(items)} items)")
        if not args.skip_eval:
            evaluate_output(output_path)

    print("Done.")


if __name__ == "__main__":
    main()
