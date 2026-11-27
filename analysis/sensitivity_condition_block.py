"""敏感性分析：剔除 Condition Block 后，各模型指标是否变化。

只用已有的逐样本预测，不产生新的运行。Condition Block 是依赖上下文的子类型
（IAA 中 11/11 被判良性）；这里报告"全集"与"不含 Condition Block"两种口径。

输出: analysis/outputs/sensitivity_condition_block.json"""

import json
from pathlib import Path

EXP = Path(__file__).resolve().parents[1] / "experiments"
OUT = Path(__file__).resolve().parent / "outputs" / "sensitivity_condition_block.json"
RUNS = {
    "gpt-5.4 0-shot (rerun 11-26)": EXP / "icl/outputs/stylematch_ctrl/icl_gpt-5.4_0shot.json",
    "gemini-3.1 0-shot (rerun 11-26)": EXP / "icl/outputs/stylematch_ctrl/icl_gemini-3.1-flash-lite_0shot.json",
    "gpt-5.4 0-shot (submitted)": EXP / "icl/outputs/gpt-5.4_500_250/icl_gpt-5.4_0shot.json",
    "ds-v4 0-shot (submitted)": EXP / "icl/outputs/deepseek-v4-flash_500_250/icl_deepseek-v4-flash_0shot.json",
    "gemini-3.1 0-shot (submitted)": EXP / "icl/outputs/gemini-3.1-flash-lite-preview_500_250/icl_gemini-3.1-flash-lite-preview_0shot.json",
    "gpt-5.4 RAG (submitted)": EXP / "rag/outputs/gpt-5.4_500_250/rag_gpt-5.4.json",
    "claude-sonnet RAG (submitted)": EXP / "rag/outputs/claude-sonnet-4-5_500_250/rag_claude-sonnet-4-5.json",
    "ds-v4 RAG (submitted)": EXP / "rag/outputs/deepseek-v4-flash_500_250/rag_deepseek-v4-flash.json",
    "gpt-5.4 no-retr. (rerun 11-26)": EXP / "rag/outputs/same_session/rag_gpt-5.4_without_rag.json",
    "claude-sonnet no-retr. (rerun 11-26)": EXP / "rag/outputs/same_session/rag_claude-sonnet-4-5_without_rag.json",
    "ds-v4 no-retr. (rerun 11-26)": EXP / "rag/outputs/same_session/rag_deepseek-v4-flash_without_rag.json",
    "qwen-8b base 0-shot (t1)": EXP / "finetune/outputs/pred_test_t1_base0shot.json",
    "qwen-8b-FT s1 (t1)": EXP / "finetune/outputs/pred_test_t1_e9_m1_s1.json",
    "qwen-8b-FT s2 (t1)": EXP / "finetune/outputs/pred_test_t1_e9_m1_s2.json",
    "qwen-8b-FT s3 (t1)": EXP / "finetune/outputs/pred_test_t1_e9_m1_s3.json",
}


def norm(items):
    rows = []
    for it in items:
        gt = it.get("Ground_Truth") or {}
        if "Label" in it:
            ab = it["Label"] == "abnormal"
        else:
            ab = gt.get("has_anomaly") is True
        rows.append((ab, it["LLM_Prediction"].get("has_anomaly") is True, str(gt.get("anomaly_type"))))
    return rows


def metrics(rows):
    tp = sum(1 for a, p, _ in rows if a and p); fn = sum(1 for a, p, _ in rows if a and not p)
    fp = sum(1 for a, p, _ in rows if not a and p); tn = sum(1 for a, p, _ in rows if not a and not p)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return {"F1": round(2 * P * R / (P + R), 3) if P + R else 0.0, "R": round(R, 3),
            "FPR": round(fp / (fp + tn), 3) if fp + tn else 0.0}


def main():
    report = {}
    print(f"{'run':34s} {'n':>5s} {'CBk':>4s} {'R(CBk)':>7s} {'F1 all':>7s} {'F1 -CBk':>8s} {'R all':>6s} {'R -CBk':>7s}")
    for name, path in RUNS.items():
        if not path.exists():
            continue
        rows = norm(json.load(open(path, encoding="utf-8-sig")))
        cbk = [r for r in rows if r[2] == "Condition Block"]
        rest = [r for r in rows if r[2] != "Condition Block"]
        a, b = metrics(rows), metrics(rest)
        r_cbk = round(sum(1 for x, p, _ in cbk if p) / len(cbk), 3) if cbk else None
        report[name] = {"n": len(rows), "n_condition_block": len(cbk), "recall_condition_block": r_cbk,
                        "all": a, "without_condition_block": b}
        print(f"{name:34s} {len(rows):5d} {len(cbk):4d} {str(r_cbk):>7s} {a['F1']:7.3f} {b['F1']:8.3f} {a['R']:6.3f} {b['R']:7.3f}")
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
