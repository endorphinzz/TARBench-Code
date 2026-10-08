"""B-LEX：词面基线（TF-IDF + 逻辑回归）。

仅凭样本文本的表面特征预测异常标签，5 折分层交叉验证输出样本外
（out-of-fold）预测，确保无训练/测试泄漏。该基线同时服务两用：
  1) 论文中的 simple heuristic baseline（Qjg5 W2）
  2) 构造伪影探测（COLING 计划 E5）：若词面特征即可高准确率预测
     异常，说明数据集存在浅层伪影；AUC 接近 0.5 则是"无伪影"证据。
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


def sample_to_text(sample: dict) -> str:
    """把样本渲染为纯文本（scenario + 全部规则原文）。"""
    parts = [str(sample.get("scenario") or "")]
    rules = sample.get("rules", {})
    for rule_id in sorted(rules, key=lambda key: int(key.split("_")[1])):
        parts.append(f"{rule_id}: {rules[rule_id]}")
    return "\n".join(parts)


def run_lexical_baseline(samples: list, n_folds: int = 5, seed: int = 20260513) -> dict:
    """对整个评测集运行 5 折 CV 词面基线。

    返回 {
        "predictions": {sample_id: {"has_anomaly": bool, "probability": float}},
        "cv_auc_mean": float, "cv_auc_std": float, "fold_aucs": [...],
    }。
    """
    texts = np.array([sample_to_text(sample) for sample in samples])
    labels = np.array([1 if sample.get("label") == "abnormal" else 0 for sample in samples])
    sample_ids = [sample.get("sample_id") for sample in samples]

    probabilities = np.zeros(len(samples))
    fold_aucs = []

    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_index, test_index in splitter.split(texts, labels):
        model = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                ("lr", LogisticRegression(max_iter=2000, C=1.0)),
            ]
        )
        model.fit(texts[train_index], labels[train_index])
        probabilities[test_index] = model.predict_proba(texts[test_index])[:, 1]
        if len(np.unique(labels[test_index])) > 1:
            fold_aucs.append(float(roc_auc_score(labels[test_index], probabilities[test_index])))

    predictions = {
        sample_id: {
            "has_anomaly": bool(probabilities[position] >= 0.5),
            "probability": float(probabilities[position]),
        }
        for position, sample_id in enumerate(sample_ids)
    }

    auc_array = np.array(fold_aucs) if fold_aucs else np.array([0.5])
    return {
        "predictions": predictions,
        "cv_auc_mean": float(auc_array.mean()),
        "cv_auc_std": float(auc_array.std()),
        "fold_aucs": fold_aucs,
    }
