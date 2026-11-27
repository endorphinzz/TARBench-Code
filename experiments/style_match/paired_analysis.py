"""风格匹配评测集上的配对比较（同一批 750 样本、同一顺序、同一时段跑出的原始对照）。

三个量：
  full_750          : 原始集与风格匹配集的整体混淆矩阵指标
  paired_229        : 仅在改写并通过独立验证的异常样本上做配对检测/定位比较
  unchanged-control : 未改写的异常样本与全部良性样本上，两次运行的判定是否一致
                      （温度 0 下的运行间差异，用于解读 paired_229）

输入: ../icl/outputs/stylematch_ctrl/ 与 ../icl/outputs/stylematch/ 下的逐样本预测
输出: data/paired_report.json"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
OUT = CODE / "experiments" / "icl" / "outputs"
sys.path.insert(0, str(OUT.parent))
from evaluate_icl import rule_ids_from_prediction  # noqa: E402  论文口径的规则 ID 解析
MODELS = ["gpt-5.4", "gemini-3.1-flash-lite"]


def load(path):
    return {it["Case_ID"]: it for it in json.load(open(path, encoding="utf-8"))}


def conf(items):
    tp = fp = fn = tn = 0
    for it in items:
        ab, p = it["Label"] == "abnormal", it["LLM_Prediction"].get("has_anomaly") is True
        if ab and p: tp += 1
        elif ab: fn += 1
        elif p: fp += 1
        else: tn += 1
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return {"acc": round((tp + tn) / max(len(items), 1), 3), "P": round(P, 3), "R": round(R, 3),
            "F1": round(2 * P * R / (P + R), 3) if P + R else 0.0,
            "FPR": round(fp / (fp + tn), 3) if fp + tn else 0.0}


def iou(it):
    """论文口径：仅异常样本有定位得分；预测为正常或无规则 ID 记 0。"""
    gt = it["Ground_Truth"]
    if it["Label"] != "abnormal" or gt.get("abnormal_fragment_start_index") is None:
        return None
    st, ln = int(gt["abnormal_fragment_start_index"]), int(gt["abnormal_fragment_length"])
    truth = {f"Rule_{i}" for i in range(st, st + ln)}
    pred = rule_ids_from_prediction(it["LLM_Prediction"].get("involved_rules"))
    if not it["LLM_Prediction"].get("has_anomaly") or not pred:
        return 0.0
    return len(pred & truth) / len(pred | truth)


def mean_iou(items):
    vals = [v for v in (iou(x) for x in items) if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def main():
    verd = {}
    for line in (HERE / "data" / "verify_results.jsonl").open(encoding="utf-8"):
        v = json.loads(line)
        verd[v["sample_id"]] = v["verdict"]
    sm = json.load(open(HERE / "data" / "stylematched_500_250.json", encoding="utf-8"))["samples"]
    chain_of = {s["chain_id"]: s["sample_id"] for s in sm}
    rewritten = {c for c, sid in chain_of.items() if verd.get(sid) == "pass"}
    unrewritten_abn = {c for c, sid in chain_of.items()
                       if next(s for s in sm if s["chain_id"] == c)["label"] == "abnormal" and verd.get(sid) != "pass"}

    report = {"n_rewritten_abnormal": len(rewritten), "n_unrewritten_abnormal": len(unrewritten_abn), "models": {}}
    for m in MODELS:
        cp, sp = OUT / "stylematch_ctrl" / f"icl_{m}_0shot.json", OUT / "stylematch" / f"icl_{m}_0shot.json"
        if not (cp.exists() and sp.exists()):
            print(f"[skip] {m}: missing control or style-matched file")
            continue
        ctrl, new = load(cp), load(sp)
        if len(ctrl) != 750 or len(new) != 750:
            print(f"[skip] {m}: incomplete (ctrl {len(ctrl)}, new {len(new)})")
            continue
        ids = [c for c in ctrl if c in new]
        c_all, n_all = [ctrl[c] for c in ids], [new[c] for c in ids]
        c_rw, n_rw = [ctrl[c] for c in ids if c in rewritten], [new[c] for c in ids if c in rewritten]

        det = lambda it: it["LLM_Prediction"].get("has_anomaly") is True
        b = sum(1 for c in rewritten if det(ctrl[c]) and not det(new[c]))
        d = sum(1 for c in rewritten if not det(ctrl[c]) and det(new[c]))
        both_yes = sum(1 for c in rewritten if det(ctrl[c]) and det(new[c]))
        iou_c = [iou(x) for x in c_rw]
        iou_n = [iou(x) for x in n_rw]
        iou_drop = sum(1 for a, b_ in zip(iou_c, iou_n) if b_ < a - 1e-9)
        iou_gain = sum(1 for a, b_ in zip(iou_c, iou_n) if b_ > a + 1e-9)

        # 未改写样本上的运行间一致性（噪声底线）
        unchanged = [c for c in ids if c in unrewritten_abn or ctrl[c]["Label"] == "benign"]
        flip_unchanged = sum(1 for c in unchanged if det(ctrl[c]) != det(new[c]))
        iou_changed_unchanged = sum(1 for c in unchanged if c in unrewritten_abn
                                    and abs(iou(ctrl[c]) - iou(new[c])) > 1e-9)

        report["models"][m] = {
            "full_750": {"control": conf(c_all), "style_matched": conf(n_all),
                          "arl_iou_control": mean_iou(c_all),
                          "arl_iou_style_matched": mean_iou(n_all)},
            "paired_229_rewritten": {
                "recall_control": round(sum(map(det, c_rw)) / len(c_rw), 3),
                "recall_style_matched": round(sum(map(det, n_rw)) / len(n_rw), 3),
                "detected_both": both_yes, "lost_after_rewrite": b, "gained_after_rewrite": d,
                "arl_iou_control": round(sum(iou_c) / len(iou_c), 3),
                "arl_iou_style_matched": round(sum(iou_n) / len(iou_n), 3),
                "iou_up": iou_gain, "iou_down": iou_drop,
                "exact_control": sum(1 for v in iou_c if v == 1.0), "exact_style_matched": sum(1 for v in iou_n if v == 1.0)},
            "run_to_run_noise_on_unchanged_samples": {
                "n": len(unchanged), "detection_flips": flip_unchanged,
                "iou_changes_on_unrewritten_abnormal": iou_changed_unchanged,
                "n_unrewritten_abnormal": len(unrewritten_abn)},
        }
    (HERE / "data" / "paired_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
