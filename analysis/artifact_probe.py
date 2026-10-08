"""E5 构造伪影探测：词面特征的分类能力从哪来。

三个分析（全部样本外 5 折分层 CV，seed 与 B-LEX 一致）：
  1) overall  —— 复核全量 TF-IDF+LR AUC（= E1 B-LEX 同管线）；
  2) per-type —— 每个异常子类 vs 全部良性的 one-vs-benign AUC（附录表）；
  3) ablation —— 特征视图消融：scenario / rules / trigger / action /
     skeleton(仅 device.attribute 结构骨架) / values(仅操作符+值 token)，
     定位伪影主要由哪类词面信息承载；
  4) top feats —— full 视图 LR 系数绝对值 top n-gram（正/负各 20）。

输出: analysis/outputs/artifact_probe.json
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "baselines"))
from detector_lex import sample_to_text  # noqa: E402

FULL_DATASET = PROJECT_ROOT / "experiments" / "datasets" / "full_dataset.json"
OUT_PATH = PROJECT_ROOT / "analysis" / "outputs" / "artifact_probe.json"
SEED = 20260513
N_FOLDS = 5

DEV_ATTR = re.compile(r"\b[a-z_]+\.[a-z_]+\b")


def split_rule(rule_text: str):
    """把 'IF <cond> THEN <actions>' 拆成 (trigger_part, action_part)。"""
    if " THEN " in rule_text:
        cond, action = rule_text.split(" THEN ", 1)
        return cond.removeprefix("IF ").strip(), action.strip()
    return rule_text.strip(), ""


def view_scenario(sample):
    return str(sample.get("scenario") or "")


def view_rules(sample):
    rules = sample.get("rules", {})
    return " ".join(rules[key] for key in sorted(rules, key=lambda k: int(k.split("_")[1])))


def _per_rule_texts(sample, part):
    texts = []
    for key in sorted(sample.get("rules", {}), key=lambda k: int(k.split("_")[1])):
        trigger, action = split_rule(sample["rules"][key])
        texts.append(trigger if part == "trigger" else action)
    return " ".join(t for t in texts if t)


def view_trigger(sample):
    return _per_rule_texts(sample, "trigger")


def view_action(sample):
    return _per_rule_texts(sample, "action")


def view_skeleton(sample):
    """仅 device.attribute 结构骨架（去掉操作符与值）。"""
    return " ".join(DEV_ATTR.findall(view_rules(sample)))


def view_values(sample):
    """仅操作符与值 token（去掉 device.attribute）。"""
    text = DEV_ATTR.sub(" ", view_rules(sample))
    return " ".join(re.findall(r"[!=<>]+|(?<!\w)-?\d+(?!\w)|[A-Za-z_]\w*", text))


VIEWS = {
    "full": sample_to_text,
    "scenario_only": view_scenario,
    "rules_only": view_rules,
    "trigger_only": view_trigger,
    "action_only": view_action,
    "skeleton_only": view_skeleton,
    "values_only": view_values,
}


def cv_auc(texts, labels, seed=SEED, n_folds=N_FOLDS):
    """样本外概率 + fold AUC 列表。"""
    texts = np.asarray(texts, dtype=object)
    labels = np.asarray(labels)
    prob = np.zeros(len(labels))
    fold_aucs = []
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(texts, labels):
        model = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("lr", LogisticRegression(max_iter=2000, C=1.0)),
        ])
        model.fit(texts[train_idx], labels[train_idx])
        prob[test_idx] = model.predict_proba(texts[test_idx])[:, 1]
        if len(np.unique(labels[test_idx])) > 1:
            fold_aucs.append(float(roc_auc_score(labels[test_idx], prob[test_idx])))
    arr = np.array(fold_aucs) if fold_aucs else np.array([0.5])
    return prob, {"auc_mean": float(arr.mean()), "auc_std": float(arr.std()), "fold_aucs": fold_aucs}


def top_features(texts, labels, k=20):
    """全量拟合一次，取 LR 系数 top n-gram（排除纯数字/单元 token 噪声）。"""
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = vec.fit_transform(texts)
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(X, labels)
    names = np.array(vec.get_feature_names_out())
    coef = lr.coef_[0]
    order = np.argsort(np.abs(coef))[::-1][: 2 * k]
    pos, neg = [], []
    for i in order:
        entry = {"feature": names[i], "coef": round(float(coef[i]), 3)}
        (pos if coef[i] > 0 else neg).append(entry)
    return {"top_anomaly_features": pos[:k], "top_benign_features": neg[:k]}


def main():
    data = json.loads(FULL_DATASET.read_text(encoding="utf-8"))
    samples = data["samples"]
    print(f"dataset: {len(samples)} samples")

    labels = np.array([1 if s["label"] == "abnormal" else 0 for s in samples])
    benign_mask = labels == 0

    # 1) overall
    full_texts = [sample_to_text(s) for s in samples]
    prob, overall = cv_auc(full_texts, labels)
    print(f"[overall] AUC = {overall['auc_mean']:.4f} ± {overall['auc_std']:.4f}")

    # 2) per-type one-vs-benign
    per_type = {}
    type_lists = defaultdict(list)
    for i, s in enumerate(samples):
        if s["label"] == "abnormal":
            type_lists[s["ground_truth"].get("anomaly_type", "Unknown")].append(i)
    for atype, idx_list in sorted(type_lists.items()):
        sub_idx = np.array(idx_list + list(np.where(benign_mask)[0]))
        sub_texts = [full_texts[i] for i in sub_idx]
        sub_labels = np.array([1] * len(idx_list) + [0] * int(benign_mask.sum()))
        _, stats = cv_auc(sub_texts, sub_labels)
        per_type[atype] = {"n_anomalous": len(idx_list), **stats}
        print(f"[per-type] {atype:35s} n={len(idx_list):4d}  AUC = {stats['auc_mean']:.4f} ± {stats['auc_std']:.4f}")

    # 3) feature-view ablation
    ablation = {}
    for name, fn in VIEWS.items():
        if name == "full":
            ablation[name] = overall
            continue
        texts = [fn(s) for s in samples]
        _, stats = cv_auc(texts, labels)
        ablation[name] = stats
        print(f"[ablation] {name:15s} AUC = {stats['auc_mean']:.4f} ± {stats['auc_std']:.4f}")

    # 4) top features on full view
    feats = top_features(full_texts, labels)
    print(f"[top feats] anomaly-side: {[f['feature'] for f in feats['top_anomaly_features'][:8]]}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "config": {"n_samples": len(samples), "n_folds": N_FOLDS, "seed": SEED,
                            "vectorizer": "TF-IDF (1-2 gram, min_df=2, sublinear) + LogisticRegression"},
                "overall": overall,
                "per_type_one_vs_benign": per_type,
                "feature_view_ablation": ablation,
                "top_features": feats,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
