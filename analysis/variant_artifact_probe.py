"""E5-6 情境化变体伪影复测：设备共享插入后词面捷径是否仍成立。

协议（全部使用与 B-LEX / artifact_probe 完全一致的 TF-IDF(1-2gram, min_df=2,
sublinear) + LR 管线，seed=20260513）：
  1) original_cv  —— 原全量数据集 5 折 CV（复核 B-LEX AUC 0.9956）；
  2) transfer     —— 在原全量数据上拟合单一模型，对设备共享变体打分。
     AUC = 变体(异常) vs 原良性池 的排序能力。若 AUC 大幅下降，说明设备域
     隔离是伪影主要来源；若仍接近 1.0，说明伪影来自写作风格分布偏移。
  3) transfer_inset —— 同一 transfer 模型在原测试折上的样本外 AUC 作为对照。
  4) invariant_cv —— 变体 + 原良性池内 5 折 CV（设备共享场景下词面模型
     是否仍可学出分离面）。
  5) matched_host_cv —— 变体 + 其宿主链（逐链配对的良性样本）内 5 折 CV。

输出: analysis/outputs/variant_artifact_probe.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "baselines"))
sys.path.insert(0, str(PROJECT_ROOT / "analysis"))

from detector_lex import sample_to_text  # noqa: E402
from artifact_probe import cv_auc  # noqa: E402
from build_full_dataset import build_sample  # noqa: E402

FULL_DATASET = PROJECT_ROOT / "experiments" / "datasets" / "full_dataset.json"
VARIANT_DATASET = (
    PROJECT_ROOT / "data" / "abnormal" / "abnormal_chains_gpt54_contextual_variants.json"
)
OUT_PATH = PROJECT_ROOT / "analysis" / "outputs" / "variant_artifact_probe.json"
SEED = 20260513
N_FOLDS = 5


def make_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("lr", LogisticRegression(max_iter=2000, C=1.0)),
    ])


def chains_to_samples(chains: list[dict]) -> list[dict]:
    return [build_sample(chain, "abnormal", i + 1) for i, chain in enumerate(chains)]


def main() -> None:
    full = json.loads(FULL_DATASET.read_text(encoding="utf-8"))
    samples = full["samples"]
    variant_data = json.loads(VARIANT_DATASET.read_text(encoding="utf-8"))
    variant_samples = chains_to_samples(variant_data["chains"])
    print(f"original dataset: {len(samples)} samples; variants: {len(variant_samples)}")

    orig_texts = np.asarray([sample_to_text(s) for s in samples], dtype=object)
    orig_labels = np.array([1 if s["label"] == "abnormal" else 0 for s in samples])
    benign_mask = orig_labels == 0

    var_texts = np.asarray([sample_to_text(s) for s in variant_samples], dtype=object)
    var_types = [s["ground_truth"]["anomaly_type"] for s in variant_samples]
    shared_sizes = [len(c.get("device_overlap", [])) for c in variant_data["chains"]]

    results: dict = {
        "config": {
            "n_original": len(samples),
            "n_benign_original": int(benign_mask.sum()),
            "n_variants": len(variant_samples),
            "device_overlap_distribution": {
                str(k): v for k, v in
                sorted(dict(defaultdict(int, {s: shared_sizes.count(s) for s in set(shared_sizes)})).items())
            },
            "n_folds": N_FOLDS,
            "seed": SEED,
        },
    }

    # 1) original in-set CV（复核 B-LEX）
    _, orig_cv = cv_auc(orig_texts, orig_labels)
    results["original_cv"] = orig_cv
    print(f"[original_cv] AUC = {orig_cv['auc_mean']:.4f} ± {orig_cv['auc_std']:.4f}")

    # 2) transfer：原全量拟合 -> 变体打分
    transfer_model = make_model().fit(orig_texts, orig_labels)
    var_scores = transfer_model.predict_proba(var_texts)[:, 1]
    benign_scores = transfer_model.predict_proba(orig_texts[benign_mask])[:, 1]

    pooled_labels = np.array([0] * int(benign_mask.sum()) + [1] * len(variant_samples))
    pooled_scores = np.concatenate([benign_scores, var_scores])
    transfer_auc = float(roc_auc_score(pooled_labels, pooled_scores))
    flagged = float((var_scores > 0.5).mean())
    results["transfer"] = {
        "auc_variants_vs_original_benign": transfer_auc,
        "variant_score_mean": float(var_scores.mean()),
        "variant_score_std": float(var_scores.std()),
        "variant_flag_rate_gt_0.5": flagged,
        "original_benign_score_mean": float(benign_scores.mean()),
    }
    print(
        f"[transfer] AUC = {transfer_auc:.4f} | variant score "
        f"{var_scores.mean():.3f}±{var_scores.std():.3f} | flag rate {flagged:.3f}"
    )

    # 3) transfer 对照：原数据 5 折样本外 AUC（同一管线的 in-distribution 参照）
    oof = np.zeros(len(orig_labels))
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for train_idx, test_idx in splitter.split(orig_texts, orig_labels):
        model = make_model().fit(orig_texts[train_idx], orig_labels[train_idx])
        oof[test_idx] = model.predict_proba(orig_texts[test_idx])[:, 1]
    results["transfer_reference_oof"] = {
        "auc_original_oof": float(roc_auc_score(orig_labels, oof)),
    }
    print(f"[transfer_reference_oof] AUC = {results['transfer_reference_oof']['auc_original_oof']:.4f}")

    # per-type transfer AUC（one-vs-original-benign）
    per_type = defaultdict(list)
    for i, t in enumerate(var_types):
        per_type[t].append(i)
    transfer_per_type = {}
    for atype, idx_list in sorted(per_type.items()):
        labels = np.array([0] * int(benign_mask.sum()) + [1] * len(idx_list))
        scores = np.concatenate([benign_scores, var_scores[idx_list]])
        transfer_per_type[atype] = {
            "n_variants": len(idx_list),
            "auc": float(roc_auc_score(labels, scores)),
            "score_mean": float(var_scores[idx_list].mean()),
        }
        print(
            f"[transfer/type] {atype:35s} n={len(idx_list):3d}  "
            f"AUC = {transfer_per_type[atype]['auc']:.4f}"
        )
    results["transfer_per_type"] = transfer_per_type

    # 4) in-variant CV：变体 + 原良性池
    inv_texts = np.concatenate([var_texts, orig_texts[benign_mask]])
    inv_labels = np.array([1] * len(variant_samples) + [0] * int(benign_mask.sum()))
    _, inv_stats = cv_auc(inv_texts, inv_labels)
    results["invariant_cv"] = inv_stats
    print(f"[invariant_cv] AUC = {inv_stats['auc_mean']:.4f} ± {inv_stats['auc_std']:.4f}")

    # 5) matched-host CV：变体 + 其宿主链
    host_ids = {c["source_benign_chain_id"] for c in variant_data["chains"]}
    host_samples = [
        s for s in samples
        if s["label"] == "benign" and s.get("chain_id") in host_ids
    ]
    matched_texts = np.concatenate([var_texts, [sample_to_text(s) for s in host_samples]])
    matched_labels = np.array([1] * len(variant_samples) + [0] * len(host_samples))
    _, matched_stats = cv_auc(matched_texts, matched_labels)
    results["matched_host_cv"] = {
        "n_host_chains": len(host_samples),
        **matched_stats,
    }
    print(
        f"[matched_host_cv] hosts={len(host_samples)}  "
        f"AUC = {matched_stats['auc_mean']:.4f} ± {matched_stats['auc_std']:.4f}"
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
