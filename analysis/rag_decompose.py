"""RAG 增益分解：同一批 750 样本上，按条件逐样本配对比较。

条件（均为当天运行，避免跨时间的模型漂移）：
  no_retrieval  : rag_{m}_without_rag.json
  nocase        : 知识库仅 22 文档（类型定义 + 对比指南），检索不到任何存储案例
  fullkb        : 发布版全量知识库（12,205 文档）
gpt-5.4 / claude 的 nocase 在 outputs/nocase/，deepseek 的三个条件都在 outputs/same_session/。

输出每个条件的混淆矩阵指标，以及相邻条件间的逐样本转移
（异常样本：检出 -> 漏检 / 漏检 -> 检出；良性样本：误报 -> 正确 / 正确 -> 误报）。
"""

import json
import sys
from pathlib import Path

RAG = Path(__file__).resolve().parents[1] / "experiments" / "rag" / "outputs"
OUT = Path(__file__).resolve().parent / "outputs" / "rag_decomposition.json"


def load(path):
    return {it["Case_ID"]: it for it in json.load(open(path, encoding="utf-8"))}


def det(it):
    return it["LLM_Prediction"].get("has_anomaly") is True


def conf(runs):
    items = list(runs.values())
    tp = sum(1 for it in items if it["Label"] == "abnormal" and det(it))
    fn = sum(1 for it in items if it["Label"] == "abnormal" and not det(it))
    fp = sum(1 for it in items if it["Label"] == "benign" and det(it))
    tn = sum(1 for it in items if it["Label"] == "benign" and not det(it))
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return {"n": len(items), "acc": round((tp + tn) / max(len(items), 1), 3), "P": round(P, 3), "R": round(R, 3),
            "F1": round(2 * P * R / (P + R), 3) if P + R else 0.0, "FPR": round(fp / (fp + tn), 3) if fp + tn else 0.0}


def transition(a, b):
    """a -> b 的逐样本转移。"""
    ids = [c for c in a if c in b]
    ab = [c for c in ids if a[c]["Label"] == "abnormal"]
    be = [c for c in ids if a[c]["Label"] == "benign"]
    return {"n_paired": len(ids),
            "abnormal_gained": sum(1 for c in ab if not det(a[c]) and det(b[c])),
            "abnormal_lost": sum(1 for c in ab if det(a[c]) and not det(b[c])),
            "benign_fixed": sum(1 for c in be if det(a[c]) and not det(b[c])),
            "benign_newly_flagged": sum(1 for c in be if not det(a[c]) and det(b[c]))}


def main():
    report = {"note": "all conditions run on 2026-11-26 on the same 750 samples; published submission-time numbers shown for reference only", "models": {}}
    plan = {
        "gpt-5.4": {"no_retrieval": RAG / "same_session/rag_gpt-5.4_without_rag.json",
                    "nocase": RAG / "nocase/rag_gpt-5.4__nocase.json",
                    "fullkb": RAG / "gpt-5.4_500_250/rag_gpt-5.4.json"},
        "claude-sonnet-4-5": {"no_retrieval": RAG / "same_session/rag_claude-sonnet-4-5_without_rag.json",
                              "nocase": RAG / "nocase/rag_claude-sonnet-4-5__nocase.json"},
        "deepseek-v4-flash": {"no_retrieval": RAG / "same_session/rag_deepseek-v4-flash_without_rag.json",
                              "nocase": RAG / "same_session/rag_deepseek-v4-flash__nocase.json",
                              "fullkb": RAG / "same_session/rag_deepseek-v4-flash__fullkb.json"},
    }
    for model, conds in plan.items():
        runs = {}
        for name, path in conds.items():
            if not path.exists():
                continue
            try:
                runs[name] = load(path)
            except Exception:
                continue
            if len(runs[name]) != 750:
                print(f"[{model}] {name}: incomplete ({len(runs[name])}/750) - skipped")
                del runs[name]
        if "no_retrieval" not in runs:
            print(f"[{model}] no complete same-day baseline yet")
            continue
        row = {name: conf(r) for name, r in runs.items()}
        pair = {}
        if "nocase" in runs:
            pair["no_retrieval->nocase"] = transition(runs["no_retrieval"], runs["nocase"])
        if "fullkb" in runs and "nocase" in runs:
            pair["nocase->fullkb"] = transition(runs["nocase"], runs["fullkb"])
        if "fullkb" in runs:
            pair["no_retrieval->fullkb"] = transition(runs["no_retrieval"], runs["fullkb"])
        # 保留份额：以 F1 与 FPR 计，nocase 相对 fullkb 的比例（仅当三者同日才有意义）
        shares = {}
        if all(k in runs for k in ("no_retrieval", "nocase", "fullkb")):
            a, c, f = row["no_retrieval"], row["nocase"], row["fullkb"]
            shares["f1_gain_retained_without_cases"] = round((c["F1"] - a["F1"]) / (f["F1"] - a["F1"]), 3) if f["F1"] != a["F1"] else None
            shares["fpr_reduction_retained_without_cases"] = round((a["FPR"] - c["FPR"]) / (a["FPR"] - f["FPR"]), 3) if a["FPR"] != f["FPR"] else None
        report["models"][model] = {"conditions": row, "paired_transitions": pair, "same_day_shares": shares}
        print(f"=== {model} ===")
        for name, r in row.items():
            print(f"  {name:13s} {r}")
        for name, p in pair.items():
            print(f"  {name:22s} {p}")
        if shares:
            print(f"  shares (same-day): {shares}")
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
