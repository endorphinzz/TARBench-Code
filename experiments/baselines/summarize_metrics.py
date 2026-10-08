"""汇总基线指标：overall + 分子类 recall（可指定数据集，默认 icl）。"""
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "outputs"
stem = sys.argv[1] if len(sys.argv) > 1 else "icl_dataset"

RULE_LEVEL = ["el", "ar", "ac", "tm", "be", "smt"]

rows = []
for name in RULE_LEVEL + ["lex"]:
    path = OUT / f"baseline_{name}_{stem}_metrics.json"
    m = json.loads(path.read_text(encoding="utf-8"))
    o, a = m["overall"], m["abnormal"]
    rows.append(
        f"baseline-{name:3s}  acc={o['accuracy']:.3f}  P={o['precision']:.3f}  R={o['recall']:.3f}"
        f"  FPR={o['false_positive_rate']:.3f}  tp={o['tp']} fp={o['fp']} tn={o['tn']} fn={o['fn']}"
        f"  loc_f1={a['localization_all_abnormal']['overlap_f1']:.3f}"
    )
print(f"=== {stem} ===")
print("\n".join(rows))

print("\n--- 分子类 recall（异常子类） ---")
subs = set()
metrics_cache = {}
for name in RULE_LEVEL:
    m = json.loads((OUT / f"baseline_{name}_{stem}_metrics.json").read_text(encoding="utf-8"))
    metrics_cache[name] = m
    subs.update(k for k in m["groups"]["sub_category"] if k != "__benign__")
header = f"{'sub_category':50s}" + "".join(f"{n:>7s}" for n in RULE_LEVEL)
print(header)
for sub in sorted(subs):
    cells = []
    for name in RULE_LEVEL:
        g = metrics_cache[name]["groups"]["sub_category"].get(sub)
        cells.append(f"{g['recall']:7.2f}" if g and (g["tp"] + g["fn"]) else "    ---")
    print(f"{sub:50s}" + "".join(cells))
