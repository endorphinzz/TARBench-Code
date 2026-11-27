"""结构化线索定位：把规则解析成 (条件 device.attr 对, 触发器 device.attr, 动作 device.attr 对)，
看片段与良性池在这些结构量上的差异，以及单特征能解释多少可分性。

背景：三线索(AND/off/security)特征删除后 AUC 仍 0.99；骨架探针 0.985 → 线索在设备-属性结构。
"池外"(out-of-pool) 指该 device.attr 对从不出现在任何良性规则的对应位置。
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "outputs" / "structure_probe.json"
RULE_RE = re.compile(r"^IF\s+(.+?)\s+THEN\s+(.+)$", re.S)
PAIR_RE = re.compile(r"([a-z0-9_]+)\.([a-z0-9_]+)")
FEATS = ["n_cond_oop", "n_trig_oop", "n_state_trig", "n_act_oop", "n_and"]


def parse(rule):
    m = RULE_RE.match(rule.strip())
    if not m:
        return None
    conds = [PAIR_RE.findall(c) for c in re.split(r"\s+AND\s+", m.group(1))]
    return conds, PAIR_RE.findall(m.group(2))


def main():
    d = json.load(open(ROOT / "experiments/datasets/full_dataset.json", encoding="utf-8-sig"))
    samples = d["samples"] if isinstance(d, dict) else d

    # ---- 良性池：条件对 / 首触发器对 / 动作对，以及设备的读取/被执行角色 ----
    ben_cond, ben_trig, ben_act = Counter(), Counter(), Counter()
    cond_dev, act_dev = Counter(), Counter()
    for s in samples:
        if s["label"] != "benign":
            continue
        for r in s["rules"].values():
            p = parse(r)
            if not p:
                continue
            conds, acts = p
            for c in conds:
                for pair in c:
                    ben_cond[pair] += 1
            for pair in conds[0]:
                ben_trig[pair] += 1
            for pair in acts:
                ben_act[pair] += 1
            for dev, _ in (pr for c in conds for pr in c):
                cond_dev[dev] += 1
            for dev, _ in acts:
                act_dev[dev] += 1
    # 设备角色 = 该设备在良性规则中出现在条件里的占比（高=传感器，低=被执行设备）
    dev_role = {dev: cond_dev[dev] / (cond_dev[dev] + act_dev[dev]) for dev in set(cond_dev) | set(act_dev)}

    def feats(rules):
        f = dict.fromkeys(FEATS, 0)
        for r in rules:
            p = parse(r)
            if not p:
                continue
            conds, acts = p
            f["n_and"] += len(conds) - 1
            for c in conds:
                f["n_cond_oop"] += sum(pair not in ben_cond for pair in c)
            for dev, _ in conds[0]:
                f["n_trig_oop"] += (dev, _) not in ben_trig
                f["n_state_trig"] += dev_role.get(dev, 0.5) < 0.5
            f["n_act_oop"] += sum(pair not in ben_act for pair in acts)
        return f

    # ---- 规则级对比：片段 / 宿主 / 良性 ----
    agg = {k: Counter() for k in ("frag", "host", "benign")}
    n = Counter()
    for s in samples:
        gt = s.get("ground_truth") or {}
        st, ln = gt.get("abnormal_fragment_start_index"), gt.get("abnormal_fragment_length")
        fids = {f"Rule_{i}" for i in range(int(st), int(st) + int(ln))} if st is not None and ln is not None else set()
        for rid, r in s["rules"].items():
            key = "frag" if rid in fids else ("host" if s["label"] == "abnormal" else "benign")
            f = feats([r])
            n[key] += 1
            for k in FEATS:
                agg[key][k] += f[k]
    rule_level = {k: {f: round(agg[k][f] / n[k], 4) for f in FEATS} for k in agg}
    rule_level["n_rules"] = dict(n)

    # ---- 片段模板数：模板身份 = 插入区间规则文本的规范化元组（与 finetune/build_sft_data.py 一致） ----
    normalize = lambda r: re.sub(r"\s+", " ", r.strip().lower())
    templates, n_abnormal = set(), 0
    for s in samples:
        gt = s.get("ground_truth") or {}
        st, ln = gt.get("abnormal_fragment_start_index"), gt.get("abnormal_fragment_length")
        if s["label"] != "abnormal" or st is None or ln is None:
            continue
        n_abnormal += 1
        templates.add(tuple(normalize(s["rules"][f"Rule_{i}"]) for i in range(int(st), int(st) + int(ln))
                            if f"Rule_{i}" in s["rules"]))
    templates.discard(())
    templates_info = {"abnormal_samples": n_abnormal, "distinct_fragment_templates": len(templates),
                      "chains_per_template": round(n_abnormal / len(templates), 1)}
    print("rule-level rates      frag     host   benign")
    for f in FEATS:
        print(f"  {f:20s} {rule_level['frag'][f]:7.1%} {rule_level['host'][f]:7.1%} {rule_level['benign'][f]:7.1%}")

    # ---- 样本级 ----
    y = np.array([s["label"] == "abnormal" for s in samples])
    X = np.array([[feats(list(s["rules"].values()))[f] for f in FEATS] for s in samples], dtype=float)
    single = {f: round(float(roc_auc_score(y, X[:, i])), 4) for i, f in enumerate(FEATS)}
    print("\nsample-level single-feature AUC:", single)

    def cv(Z):
        p = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=20260513).split(Z, y):
            p[te] = LogisticRegression(max_iter=2000).fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
        return round(float(roc_auc_score(y, p)), 4)

    nr = np.maximum(np.array([len(s["rules"]) for s in samples]), 1)[:, None]
    combined = {"counts_only": cv(StandardScaler().fit_transform(X)),
                "counts_plus_per_rule_rates": cv(StandardScaler().fit_transform(np.column_stack([X, X / nr])))}
    print("5 structural features combined (LR, 5-fold CV) AUC:", combined)

    res = {"fragment_templates": templates_info, "rule_level": rule_level, "sample_single_feature_auc": single, "combined_cv_auc": combined}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
