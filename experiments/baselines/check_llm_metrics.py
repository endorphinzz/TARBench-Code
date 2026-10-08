"""核查 LLM 指标：RAG without_rag / ICL 0shot 的样本规模与总体指标，
与论文 Table 2 (tab:mainresults) 对照，确定主表数据来源。"""
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
RAG = _ROOT / "rag" / "outputs"
ICL = _ROOT / "icl" / "outputs"

PAPER = {  # 论文 Table 2 数字（Model: acc, P, R, F1, FPR）
    "gpt-5.4": (0.83, 0.65, 0.97, 0.78, 0.24),
    "ds-v4": (None,) * 5,
    "claude-sonnet": (0.71, 0.53, 0.99, 0.69, 0.42),
    "gemini-3.1": (None,) * 5,
}

print("=== RAG without_rag ===")
for model in ["gpt-5.4", "claude-sonnet-4-5", "deepseek-v4-flash", "gemini-3.1-flash-lite-preview"]:
    p = RAG / f"{model}_500_250" / f"rag_{model}_without_rag_metrics.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    o = m["overall"]
    n_abn, n_ben = o["tp"] + o["fn"], o["tn"] + o["fp"]
    f1 = 2 * o["precision"] * o["recall"] / (o["precision"] + o["recall"]) if o["precision"] + o["recall"] else 0
    print(
        f"{model:35s} n=({n_ben}ben,{n_abn}abn) acc={o['accuracy']:.3f} P={o['precision']:.3f} "
        f"R={o['recall']:.3f} F1={f1:.3f} FPR={o['false_positive_rate']:.3f}"
    )

print("\n=== ICL 0shot ===")
for model in ["gpt-5.4", "deepseek-v4-flash", "gemini-3.1-flash-lite-preview"]:
    p = ICL / f"{model}_500_250" / f"icl_{model}_0shot_metrics.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    o = m["overall"]
    n_abn, n_ben = o["tp"] + o["fn"], o["tn"] + o["fp"]
    f1 = 2 * o["precision"] * o["recall"] / (o["precision"] + o["recall"]) if o["precision"] + o["recall"] else 0
    print(
        f"{model:35s} n=({n_ben}ben,{n_abn}abn) acc={o['accuracy']:.3f} P={o['precision']:.3f} "
        f"R={o['recall']:.3f} F1={f1:.3f} FPR={o['false_positive_rate']:.3f}"
    )

print("\n=== 论文 Table 2 参照 ===")
for k, v in PAPER.items():
    if v[0] is not None:
        print(f"{k:35s} acc={v[0]:.2f} P={v[1]:.2f} R={v[2]:.2f} F1={v[3]:.2f} FPR={v[4]:.2f}")
