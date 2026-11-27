"""IAA 的二元层面一致性（完整 500 样本）：按真值类型统计各标注者与多数票是否判为异常。

区分两件事：
  (a) 子类型名是否与发布标签一致 —— 论文附录报告的是这个；
  (b) 二元判定（Anomalous vs Benign）是否与发布标签一致 —— 检测与定位任务只依赖 (b)。

输入: annotation/iaa_key.json 与 annotation/annotations/annotator_{A,B,C}/*.jsonl
输出: analysis/outputs/iaa_binary_agreement.json"""

import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

ANN = Path(__file__).resolve().parents[1] / "annotation"
OUT = Path(__file__).resolve().parent / "outputs" / "iaa_binary_agreement.json"


def main():
    key = json.load(open(ANN / "iaa_key.json", encoding="utf-8-sig"))
    key_map = {k["sample_id"]: k for k in key["items"]}

    def gt(sid):
        k = key_map[sid]
        return k["label"], "+".join(k.get("type_codes") or [])
    labels = {}
    for a in "ABC":
        labels[a] = {}
        for f in sorted(glob.glob(str(ANN / "annotations" / f"annotator_{a}" / "*.jsonl"))):
            for line in open(f, encoding="utf-8-sig"):
                if line.strip():
                    r = json.loads(line)
                    labels[a][r["sample_id"]] = r

    rows = defaultdict(list)
    for sid, item in key_map.items():
        lab, typ = gt(sid)
        votes = [labels[a].get(sid) for a in "ABC"]
        rows[(lab, typ)].append((sid, votes))

    report = {"n": len(key_map), "types": {}}
    print(f"IAA set n={len(key_map)}")
    print(f"{'gt':8s} {'type':34s} {'n':>4s} {'majority ANOM':>13s} {'A':>6s} {'B':>6s} {'C':>6s}  subtype votes cast by annotators who said Anomalous")
    for (lab, typ), items in sorted(rows.items(), key=lambda kv: (kv[0][0], -len(kv[1]))):
        n = len(items)
        anom = lambda v: v is not None and v["label"].lower() == "anomalous"
        per = {a: sum(anom(v[i]) for _, v in items) / n for i, a in enumerate("ABC")}
        maj = sum(1 for _, v in items if sum(anom(x) for x in v) >= 2) / n
        votes = Counter(v["type"] for _, vs in items for v in vs if anom(v))
        report["types"][f"{lab}:{typ}"] = {"n": n, "majority_anom": round(maj, 3),
                                           "per_annotator_anom": {k: round(x, 3) for k, x in per.items()},
                                           "subtype_votes": dict(votes)}
        print(f"{lab:8s} {typ[:34]:34s} {n:4d} {maj:13.3f} {per['A']:6.3f} {per['B']:6.3f} {per['C']:6.3f}  {dict(votes) if votes else ''}")
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
