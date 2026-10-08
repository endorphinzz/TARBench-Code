"""核实全量 metrics 中的 anomaly_type 清单与论文 14 子类对齐。"""
import json
from pathlib import Path

m = json.loads((Path(__file__).resolve().parent / "outputs" / "baseline_ar_full_dataset_metrics.json").read_text(encoding="utf-8"))
types = m["groups"]["anomaly_type"]
print(f"共 {len(types)} 个 anomaly_type：")
for name in sorted(types):
    c = types[name]
    print(f"  {name:35s} n={c['tp'] + c['fn']:5d}  recall={c['recall']:.2f}")
