"""模板不相交（template-held-out）条件下的词面探针。

主表的词面基线用 5 折交叉验证，训练折里见过全部片段模板的实例。这里改用微调实验的
划分（experiments/finetune/datasets/splits/）：训练集与测试集在片段模板层面不相交，
测试集中出现的片段模板在训练集中从未出现。用同一套 TF-IDF(1-2gram)+逻辑回归在
训练集上训练，在测试集上评测：

  检测  样本级二分类，报告 AUC / F1 / FPR
  定位  规则级二分类（规则是否位于异常片段内），按论文 ARL-IoU 口径
        （仅异常样本，|pred ∩ truth| / |pred ∪ truth|，无预测记 0）

输出: analysis/outputs/template_disjoint_probe.json
"""

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "experiments/finetune/datasets/splits/train_full.json"
TEST = ROOT / "experiments/finetune/datasets/splits/test_t1.json"
OUT = ROOT / "analysis/outputs/template_disjoint_probe.json"


def load(path):
    d = json.load(open(path, encoding="utf-8-sig"))
    return d["samples"] if isinstance(d, dict) else d


def sample_text(s):
    parts = [str(s.get("scenario") or "")]
    for rid in sorted(s["rules"], key=lambda k: int(k.split("_")[1])):
        parts.append(f"{rid}: {s['rules'][rid]}")
    return "\n".join(parts)


def fragment_ids(s):
    gt = s.get("ground_truth") or {}
    st, ln = gt.get("abnormal_fragment_start_index"), gt.get("abnormal_fragment_length")
    if st is None or ln is None:
        return set()
    return {f"Rule_{i}" for i in range(int(st), int(st) + int(ln))}


def ordered_rules(sample):
    return [(rid, sample["rules"][rid]) for rid in sorted(sample["rules"], key=lambda k: int(k.split("_")[1]))]


def detection(train, test):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train = vec.fit_transform([sample_text(s) for s in train])
    y_train = np.array([s.get("label") == "abnormal" for s in train])
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(X_train, y_train)
    proba = clf.predict_proba(vec.transform([sample_text(s) for s in test]))[:, 1]
    y = np.array([s.get("label") == "abnormal" for s in test])
    pred = proba >= 0.5
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"auc": round(float(roc_auc_score(y, proba)), 4),
            "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
            "fpr": round(fp / (fp + tn), 4) if fp + tn else 0.0}


def localization(train, test):
    train_rows, train_y = [], []
    for s in train:
        if s.get("label") != "abnormal":
            continue
        frag = fragment_ids(s)
        for rid, rule in ordered_rules(s):
            train_rows.append(rule)
            train_y.append(rid in frag)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(vec.fit_transform(train_rows), np.array(train_y))

    ious = []
    for s in test:
        if s.get("label") != "abnormal":
            continue
        frag = fragment_ids(s)
        rules = ordered_rules(s)
        proba = clf.predict_proba(vec.transform([r for _, r in rules]))[:, 1]
        chosen = proba >= 0.5
        if not chosen.any():  # 没有规则过阈值时取最高分的一条
            chosen = np.zeros(len(rules), dtype=bool)
            chosen[int(np.argmax(proba))] = True
        pred = {rid for (rid, _), c in zip(rules, chosen) if c}
        union = pred | frag
        ious.append(len(pred & frag) / len(union) if union else 0.0)
    ious = np.array(ious)
    return {"arl_iou": round(float(ious.mean()), 4), "exact_rate": round(float((ious == 1).mean()), 4),
            "n_abnormal_scored": int(len(ious))}


def main():
    train, test = load(TRAIN), load(TEST)
    result = {
        "protocol": "TF-IDF(1-2gram) + LogisticRegression trained on train_full.json, evaluated on test_t1.json; "
                    "fragment templates of test chains are absent from training",
        "n_train": len(train), "n_test": len(test),
        "detection": detection(train, test),
        "localization": localization(train, test),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
