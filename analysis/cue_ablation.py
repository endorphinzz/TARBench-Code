"""线索消融：片段-良性可分性由哪些词面线索承载。

三个候选线索（规则级出现率，片段 vs 宿主）：
  AND 合取 13.1% vs 0.0% | THEN ... = off 42.5% vs 3.5% | 锁/警报类安全词汇 35.3% vs 7.5%

三个测试（全量 15,183，论文探针同款 TF-IDF(1-2)+LR，5 折 CV）：
  (a) 只用三线索的规则分类器能到多少
  (b) 从词表中删掉线索相关特征后，TF-IDF 探针掉到多少
  (c) 片段完全不含三线索的异常样本上，探针 AUC 是多少
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "outputs" / "cue_ablation.json"
CUES = {"AND": r"\bAND\b", "OFF": r"THEN [^=]+ = off\b", "SEC": r"(secure_mode|smart_lock|window_lock|siren)"}


def sample_text(s):
    parts = [str(s.get("scenario") or "")]
    for rid in sorted(s["rules"], key=lambda k: int(k.split("_")[1])):
        parts.append(f"{rid}: {s['rules'][rid]}")
    return "\n".join(parts)


def cv_probe(texts, y, drop_tokens=frozenset()):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = vec.fit_transform(texts)
    if drop_tokens:
        names = vec.get_feature_names_out()
        keep = np.array([not any(t in drop_tokens for t in n.split(" ")) for n in names])
        X = X[:, keep]
    proba = np.zeros(len(y))
    for tr, te in StratifiedKFold(n_splits=5, shuffle=True, random_state=20260513).split(X, y):
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], y[tr])
        proba[te] = clf.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, proba), proba


def main():
    d = json.load(open(ROOT / "experiments/datasets/full_dataset.json", encoding="utf-8-sig"))
    samples = d["samples"] if isinstance(d, dict) else d
    texts = [sample_text(s) for s in samples]
    y = np.array([s["label"] == "abnormal" for s in samples])

    def frag(s):
        gt = s.get("ground_truth") or {}
        st, ln = gt.get("abnormal_fragment_start_index"), gt.get("abnormal_fragment_length")
        if st is None or ln is None:
            return []
        return [s["rules"][f"Rule_{i}"] for i in range(int(st), int(st) + int(ln)) if f"Rule_{i}" in s["rules"]]

    # (a) 三线索规则分类器（样本级：片段规则不可知，故对全文任一规则命中即判异常）
    cue_hit = {k: np.array([bool(re.search(p, t)) for t in texts]) for k, p in CUES.items()}
    rule_any = cue_hit["AND"] | cue_hit["OFF"] | cue_hit["SEC"]
    tp = int((rule_any & y).sum()); fp = int((rule_any & ~y).sum()); fn = int((~rule_any & y).sum()); tn = int((~rule_any & ~y).sum())
    prec = tp / (tp + fp); rec = tp / (tp + fn)
    a = {"rule": "flag if ANY rule in the sample contains AND / '= off' / security vocab",
         "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(2 * prec * rec / (prec + rec), 3),
         "fpr": round(fp / (fp + tn), 3), "per_cue_recall": {k: round(float(cue_hit[k][y].mean()), 3) for k, c in cue_hit.items()},
         "per_cue_benign_rate": {k: round(float(cue_hit[k][~y].mean()), 3) for k, c in cue_hit.items()}}

    # (b) 特征删除消融
    sec_devices = sorted({m for s in samples if s["label"] == "abnormal" for r in frag(s)
                          for m in re.findall(r"([a-z0-9_]+)\.", r) if re.search(r"(lock|siren|secure|camera)", m)})
    auc_full, proba = cv_probe(texts, y)
    abl = {"full": round(float(auc_full), 4)}
    for name, toks in [("drop_AND", {"and"}), ("drop_OFF", {"off"}), ("drop_SEC", set(sec_devices)),
                       ("drop_all_three", {"and", "off"} | set(sec_devices))]:
        abl[name] = round(float(cv_probe(texts, y, frozenset(toks))[0]), 4)

    # (c) 无线索异常子集
    cue_free = np.array([bool(fr) and not any(re.search(p, r) for r in fr for p in CUES.values()) if s["label"] == "abnormal" else False
                         for s, fr in ((s, frag(s)) for s in samples)])
    abn = y.astype(bool)
    n_cf = int(cue_free.sum())
    c = {"abnormal_samples_without_any_cue": n_cf, "of": int(abn.sum())}
    if n_cf >= 10:
        idx = np.where(cue_free | ~abn)[0]
        c["probe_auc_on_cuefree_abnormal_vs_benign"] = round(float(roc_auc_score(abn[idx], proba[idx])), 4)
        c["probe_flag_rate_on_cuefree_abnormal"] = round(float((proba[cue_free] >= 0.5).mean()), 3)
        c["probe_mean_score_cuefree_abnormal_vs_benign"] = [round(float(proba[cue_free].mean()), 3), round(float(proba[~abn].mean()), 3)]

    res = {"n": len(samples), "a_three_cue_rule": a, "b_feature_drop_ablation_auc": abl,
           "security_devices_dropped": len(sec_devices), "c_cue_free_subset": c}
    print(json.dumps(res, ensure_ascii=False, indent=2))
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
