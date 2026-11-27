"""风格匹配评测集，第 4 步（本地）：组装评测集 + 词面迁移探针。

1. 用通过验证的改写片段原位替换 ICL-750 的异常片段 -> data/stylematched_500_250.json
   （与 run_icl_benchmark.py 的数据格式兼容；ground_truth 的 involved 区间不变）
2. 词面迁移探针（与附录的设备共享变体探针同一口径）：在全量原始数据上训练
   TF-IDF+LR，对改写后的异常样本与良性样本打分，与原始片段对照。

输出: data/stylematched_500_250.json, data/stylematch_probe.json"""

import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parent
CODE = BASE_DIR.parents[1]
ICL = CODE / "experiments" / "icl" / "datasets" / "icl_eval_500_250.json"
FULL = CODE / "experiments" / "datasets" / "full_dataset.json"
OUT_DS = BASE_DIR / "data" / "stylematched_500_250.json"
OUT_PROBE = BASE_DIR / "data" / "stylematch_probe.json"


def sample_text(s):
    parts = [str(s.get("scenario") or "")]
    for rid in sorted(s["rules"], key=lambda k: int(k.split("_")[1])):
        parts.append(f"{rid}: {s['rules'][rid]}")
    return "\n".join(parts)


def main():
    # ---- 1. 组装诊断集 ----
    paras = {json.loads(l)["sample_id"]: json.loads(l) for l in (BASE_DIR / "data" / "paraphrase_results.jsonl").open(encoding="utf-8")}
    verdicts = {json.loads(l)["sample_id"]: json.loads(l) for l in (BASE_DIR / "data" / "verify_results.jsonl").open(encoding="utf-8")}
    icl = json.load(open(ICL, encoding="utf-8-sig"))
    samples = icl["samples"] if isinstance(icl, dict) else icl

    kept, dropped = [], []
    new_samples = []
    for s in samples:
        s2 = json.loads(json.dumps(s))
        if s.get("label") == "abnormal":
            sid = s["sample_id"]
            v = verdicts.get(sid, {})
            p = paras.get(sid, {})
            if v.get("verdict") == "pass" and p.get("ok"):
                for rid, new_rule in zip(p["fragment_rule_ids"], p["rewritten_rules"]):
                    s2["rules"][rid] = new_rule
                s2["style_rewrite"] = {"mappings": p.get("mappings", {}), "notes": p.get("notes", "")}
                kept.append(sid)
            else:
                dropped.append((sid, v.get("verdict", "missing")))
        new_samples.append(s2)
    print(f"kept {len(kept)} rewritten abnormal; dropped {len(dropped)}: {[d for d in dropped][:10]}")

    out = {"metadata": {"source": "icl_eval_500_250.json", "style_rewritten_abnormal": len(kept),
                        "dropped": {k: v for k, v in dropped}},
           "samples": new_samples}
    OUT_DS.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 2. 词面迁移探针（App C transfer 口径）----
    full = json.load(open(FULL, encoding="utf-8-sig"))
    fs = full["samples"] if isinstance(full, dict) else full
    X_full = [sample_text(s) for s in fs]
    y_full = np.array([s.get("label") == "abnormal" for s in fs])
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xf = vec.fit_transform(X_full)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xf, y_full)

    orig_abn = [s for s in samples if s.get("label") == "abnormal" and s["sample_id"] in kept]
    new_abn = [s for s in new_samples if s.get("label") == "abnormal" and s["sample_id"] in kept]
    ben = [s for s in samples if s.get("label") == "benign"]

    po = clf.predict_proba(vec.transform([sample_text(s) for s in orig_abn]))[:, 1]
    pn = clf.predict_proba(vec.transform([sample_text(s) for s in new_abn]))[:, 1]
    pb = clf.predict_proba(vec.transform([sample_text(s) for s in ben]))[:, 1]

    res = {
        "probe": "TF-IDF(1-2)+LR trained on full ORIGINAL dataset, applied to ICL-750 subsets (App C transfer protocol)",
        "n": {"orig_abnormal": len(po), "rewritten_abnormal": len(pn), "benign": len(pb)},
        "original_fragment": {
            "flag_rate@0.5": float((po >= 0.5).mean()),
            "score_mean": float(po.mean()),
            "auc_vs_benign": float(roc_auc_score(np.r_[np.ones(len(po)), np.zeros(len(pb))], np.r_[po, pb])),
        },
        "rewritten_fragment": {
            "flag_rate@0.5": float((pn >= 0.5).mean()),
            "score_mean": float(pn.mean()),
            "auc_vs_benign": float(roc_auc_score(np.r_[np.ones(len(pn)), np.zeros(len(pb))], np.r_[pn, pb])),
        },
        "benign": {"flag_rate@0.5": float((pb >= 0.5).mean()), "score_mean": float(pb.mean())},
        "reference": {
            "device_sharing_variant_flag_rate_90.7pct_auc_0.994": "variant_artifact_probe.json (App C)",
            "cv_auc_on_original": 0.9956,
        },
    }
    OUT_PROBE.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
