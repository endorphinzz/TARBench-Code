"""启发式 vs 符号方法内部对照分析（路线 2 的核心产出）。

三组对照（B-SMT 按 reason 标记拆分为 AC/TM/BE/IS 四个检查）：
  B-TM（离散等值互斥）  vs SMT-TM（完整可满足性，含数值区间）
  B-AC（触发结构相交）  vs SMT-AC（Z3 联合可满足）
  B-BE（本体值域）      vs SMT-BE（独立物理范围）
另附 AC 三档对照：B-AR（不查触发）→ B-AC（结构）→ SMT-AC（形式）。

同时导出启发式独有 flagged 的良性样本（潜在标注噪声候选，供 E2 审计）。
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent / "outputs"
stem = sys.argv[1] if len(sys.argv) > 1 else "rag_dataset"

SMT_MARKERS = {
    "smt_ac": "(action conflict)",
    "smt_tm": "(trigger mutex)",
    "smt_be": "(safety boundary error)",
    "smt_is": "(illegal state rule)",
}


def load_predictions(name):
    data = json.loads((OUT / f"baseline_{name}_{stem}.json").read_text(encoding="utf-8"))
    return {item["Sample_Index"]: item for item in data}


def flagged(item):
    return item["LLM_Prediction"]["has_anomaly"] is True


def smt_check_flags(item):
    reason = item["LLM_Prediction"]["reason"]
    return {check: marker in reason for check, marker in SMT_MARKERS.items()}


def compare(heur_name, heur_flag_fn, smt_check, preds, dump_examples=0):
    heur, smt = preds[heur_name], preds["smt"]
    both = heur_only = smt_only = 0
    heur_only_benign, heur_only_abnormal = [], []
    smt_only_subcats = Counter()
    heur_only_subcats = Counter()

    for index, heur_item in heur.items():
        smt_item = smt[index]
        h = heur_flag_fn(heur_item)
        s = smt_check_flags(smt_item)[smt_check]
        if h and s:
            both += 1
        elif h and not s:
            heur_only += 1
            sub = smt_item["Ground_Truth"].get("sub_category") or "__benign__"
            heur_only_subcats[sub] += 1
            if smt_item["Label"] == "benign":
                heur_only_benign.append((index, heur_item, smt_item))
            else:
                heur_only_abnormal.append(smt_item)
        elif s and not h:
            smt_only += 1
            smt_only_subcats[smt_item["Ground_Truth"].get("sub_category") or "__benign__"] += 1

    print(f"\n=== {heur_name.upper()} vs SMT-{smt_check.split('_')[1].upper()} ===")
    print(f"  both: {both}   {heur_name}-only: {heur_only}   smt-only: {smt_only}")
    if heur_only_subcats:
        print(f"  {heur_name}-only 按子类: {dict(heur_only_subcats)}")
    if smt_only_subcats:
        print(f"  smt-only 按子类: {dict(smt_only_subcats)}")

    if dump_examples and heur_only_benign:
        print(f"  -- {heur_name}-only 良性样本（潜在标注噪声候选，前 {dump_examples} 例）--")
        for index, heur_item, smt_item in heur_only_benign[:dump_examples]:
            print(f"  [Sample {index}] {smt_item.get('Scenario_Group')} | "
                  f"chain_len={smt_item.get('Chain_Length')}")
            print(f"    {heur_name} reason: {heur_item['LLM_Prediction']['reason'][:260]}")
            involved = set(heur_item["LLM_Prediction"]["involved_rules"])
            for rule_text in smt_item["Rules"]:
                rule_id = rule_text.split(":", 1)[0]
                if rule_id in involved:
                    print(f"    {rule_text}")
    return heur_only_benign


def main():
    preds = {name: load_predictions(name) for name in ["ac", "tm", "be", "ar", "smt"]}
    print(f"内部对照分析（{stem}，{len(preds['smt'])} 样本）")

    # B-TM vs SMT-TM
    compare("tm", flagged, "smt_tm", preds)

    # B-AC vs SMT-AC（导出良性误报样例供 E2 审计）
    compare("ac", flagged, "smt_ac", preds, dump_examples=3)

    # B-BE vs SMT-BE
    compare("be", flagged, "smt_be", preds)

    # AC 三档对照：B-AR（不查触发）在 SMT-AC 独有样本上的覆盖
    ar, smt = preds["ar"], preds["smt"]
    smt_ac_only = [
        index for index, item in smt.items()
        if smt_check_flags(item)["smt_ac"] and not flagged(preds["ac"][index])
    ]
    ar_covers = sum(1 for index in smt_ac_only if flagged(ar[index]))
    print(f"\n=== AC 三档对照（触发推理：无→结构→形式） ===")
    print(f"  SMT-AC 独有（B-AC 漏报）: {len(smt_ac_only)} 例，其中 B-AR（不查触发）覆盖 {ar_covers} 例")
    ar_only_benign = [
        index for index, item in ar.items()
        if flagged(item) and item["Label"] == "benign" and not smt_check_flags(smt[index])["smt_ac"]
    ]
    print(f"  B-AR 独有良性误报（SMT-AC 正确拒绝）: {len(ar_only_benign)} 例")


if __name__ == "__main__":
    main()
