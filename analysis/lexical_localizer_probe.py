"""规则级词面定位器 + 片段-宿主设备重叠统计。

1) 异常片段与宿主规则是否共享设备（全量数据集）；
2) 一个只看规则文本表面的词面定位器（TF-IDF + 逻辑回归，规则级 5 折交叉验证）
   在论文 ARL-IoU 口径（仅异常样本，|pred ∩ truth| / |pred ∪ truth|，无预测记 0）下的得分，
   供与主表中的模型对照。

输出: analysis/outputs/lexical_localizer_probe.json"""

import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "experiments" / "datasets" / "full_dataset.json"
OUTPUT = ROOT / "analysis" / "outputs" / "lexical_localizer_probe.json"

SEED = 20260513


def devices(rule_text):
    return set(re.findall(r"([A-Za-z0-9_]+)\.", rule_text))


def fragment_overlap_stats(samples):
    """W2：片段 vs 宿主的设备级交互情况。"""
    n = zero = 0
    for sample in samples:
        if sample.get("label") != "abnormal":
            continue
        gt = sample["ground_truth"]
        st, ln = gt.get("abnormal_fragment_start_index"), gt.get("abnormal_fragment_length")
        if st is None or ln is None:
            continue
        frag_ids = {f"Rule_{i}" for i in range(int(st), int(st) + int(ln))}
        frag_dev, host_dev = set(), set()
        for rid, rtext in sample["rules"].items():
            (frag_dev if rid in frag_ids else host_dev).update(devices(rtext))
        n += 1
        if not (frag_dev & host_dev):
            zero += 1
    return {"abnormal_samples": n, "zero_device_overlap": zero, "zero_overlap_rate": zero / n if n else None}


def run_localizer(samples):
    """W1：规则级词面定位器，论文 ARL-IoU 口径。"""
    rows, meta = [], []
    for si, sample in enumerate(samples):
        gt = sample.get("ground_truth") or {}
        st, ln = gt.get("abnormal_fragment_start_index"), gt.get("abnormal_fragment_length")
        frag_ids = (
            {f"Rule_{i}" for i in range(int(st), int(st) + int(ln))}
            if st is not None and ln is not None
            else set()
        )
        for rid in sorted(sample["rules"], key=lambda k: int(k.split("_")[1])):
            rows.append(sample["rules"][rid])
            meta.append((si, rid in frag_ids))

    y = np.array([m[1] for m in meta])
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = vec.fit_transform(np.array(rows))
    proba = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[train_idx], y[train_idx])
        proba[test_idx] = clf.predict_proba(X[test_idx])[:, 1]

    by_sample = {}
    for (si, in_frag), p in zip(meta, proba):
        by_sample.setdefault(si, []).append((in_frag, p))

    ious = []
    for si, sample in enumerate(samples):
        if sample.get("label") != "abnormal":
            continue
        pairs = by_sample[si]
        chosen = [p >= 0.5 for _, p in pairs]
        if not any(chosen):
            best = int(np.argmax([p for _, p in pairs]))
            chosen = [i == best for i in range(len(pairs))]
        pred = {i for i, c in enumerate(chosen) if c}
        truth = {i for i, (f, _) in enumerate(pairs) if f}
        union = pred | truth
        ious.append(len(pred & truth) / len(union) if union else 0.0)

    ious = np.array(ious)
    return {
        "method": "TF-IDF(1-2gram) + LogisticRegression, rule-level labels, 5-fold CV, paper ARL-IoU protocol",
        "rule_rows": int(len(rows)),
        "in_fragment_rules": int(y.sum()),
        "abnormal_samples_scored": int(len(ious)),
        "arl_iou_mean": float(ious.mean()),
        "exact_iou_rate": float((ious == 1).mean()),
        "zero_iou_rate": float((ious == 0).mean()),
        "reference": {
            "best_prompted_llm_arl_iou": 0.84,
            "qwen_8b_arl_iou": 0.62,
            "qwen_8b_ft_arl_iou": 0.986,
        },
    }


def main():
    data = json.load(open(DATASET, encoding="utf-8-sig"))
    samples = data["samples"] if isinstance(data, dict) else data
    result = {
        "w2_device_overlap": fragment_overlap_stats(samples),
        "w1_lexical_localizer": run_localizer(samples),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
