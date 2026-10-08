"""E2 (A2): Build the inter-annotator agreement (IAA) sample set.

- >= 500 samples total: 150 benign + 350 abnormal (stratified).
- Stratification: benign by (scenario_group, chain_length); abnormal by Type
  (proportional, floor 8) then round-robin by (Sub_Category, scenario_group,
  chain_length) via the shared diverse_sample() helper.
- Annotator view strips everything except {sample_id, scenario, rules}
  (no chain_id / label / ground_truth / method -> no leakage).
- Ground-truth key written to a separate file (annotation/iaa_key.json).
- Batches of 25 samples -> annotation/batches/batch_01.json .. batch_20.json.

Usage: python annotation/build_iaa_sample.py
"""

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.build_datasets import diverse_sample  # noqa: E402

FULL_DATASET = ROOT / "experiments" / "datasets" / "full_dataset.json"
OUT_DIR = Path(__file__).resolve().parent
BATCH_DIR = OUT_DIR / "batches"

SEED = 20260919
N_BENIGN = 150
N_ABNORMAL = 350
BATCH_SIZE = 25

# Raw data Type -> paper taxonomy code (14 classes).
TYPE_NAME_TO_CODE = {
    "Action Conflict": "AC",
    "Action Duplication": "AD",
    "Action Revert": "AR",
    "Loop": "EL",
    "Condition Bypass": "CB",
    "Action Mutex": "AM",
    "Illegal State Rule": "IS",
    "Safety Boundary Error": "BE",
    "Trigger Mutex": "TM",
    "Action Block": "AB",
    "Condition Block": "CBk",
    "Common Sense Violation": "CV",
    "Privilege Escalation": "PE",
    "Physical Cross-rule Interaction": "PD",
}


def type_codes(sample):
    """Map one abnormal sample to its list of 14-class codes (1-3 elements)."""
    gt = sample["ground_truth"]
    t = gt.get("anomaly_type")
    if t in ("Dual Anomaly", "Triple Anomaly"):
        match = re.search(r"\[(.*?)\]", str(gt.get("sub_category", "")))
        if not match:
            return []
        names = [part.strip() for part in match.group(1).split("&")]
        return [TYPE_NAME_TO_CODE[n] for n in names if n in TYPE_NAME_TO_CODE]
    code = TYPE_NAME_TO_CODE.get(t)
    return [code] if code else []


def allocate_proportional(type_counts, total, floor):
    """Largest-remainder proportional allocation with a per-class floor."""
    pool = sum(type_counts.values())
    raw = {t: total * c / pool for t, c in type_counts.items()}
    alloc = {t: max(floor, int(raw[t])) for t in type_counts}
    # Scale down / top up with largest remainders until sum == total.
    while sum(alloc.values()) > total:
        t = max(alloc, key=lambda t: alloc[t] - raw[t])
        if alloc[t] <= floor:
            t = max(alloc, key=alloc.get)
        alloc[t] -= 1
    while sum(alloc.values()) < total:
        candidates = {t: raw[t] - alloc[t] for t in alloc if alloc[t] < type_counts[t]}
        t = max(candidates, key=candidates.get)
        alloc[t] += 1
    return alloc


def main():
    data = json.loads(FULL_DATASET.read_text(encoding="utf-8"))
    samples = data["samples"]
    benign = [s for s in samples if s["label"] == "benign"]
    abnormal = [s for s in samples if s["label"] == "abnormal"]
    print(f"pool: benign={len(benign)} abnormal={len(abnormal)}")

    # --- benign selection: stratified by (scenario_group, chain_length)
    benign_flat = [
        {
            "chain_id": s["chain_id"],
            "scenario_group": s["scenario_group"],
            "chain_length": s["chain_length"],
            "_sample": s,
        }
        for s in benign
    ]
    benign_sel = diverse_sample(
        benign_flat, N_BENIGN, ("scenario_group", "chain_length"), SEED + 1
    )

    # --- abnormal selection: proportional over Type, diverse within Type
    type_counts = Counter(s["ground_truth"]["anomaly_type"] for s in abnormal)
    alloc = allocate_proportional(type_counts, N_ABNORMAL, floor=8)

    abnormal_sel = []
    for i, (t, n) in enumerate(sorted(alloc.items())):
        pool = [
            {
                "chain_id": s["chain_id"],
                "Type": s["ground_truth"]["anomaly_type"],
                "Sub_Category": s["ground_truth"]["sub_category"],
                "scenario_group": s["scenario_group"],
                "chain_length": s["chain_length"],
                "_sample": s,
            }
            for s in abnormal
            if s["ground_truth"]["anomaly_type"] == t
        ]
        chosen = diverse_sample(
            pool, n, ("Sub_Category", "scenario_group", "chain_length"), SEED + 100 + i
        )
        abnormal_sel.extend(chosen)
        print(f"  type {t!r}: allocated {n} (pool {len(pool)})")

    # --- combine, shuffle, opaque ids
    combined = [d["_sample"] for d in benign_sel] + [d["_sample"] for d in abnormal_sel]
    rng = random.Random(SEED)
    rng.shuffle(combined)

    annotator_view = []
    key = []
    for i, s in enumerate(combined, start=1):
        sid = f"IAA_{i:04d}"
        annotator_view.append(
            {"sample_id": sid, "scenario": s["scenario"], "rules": s["rules"]}
        )
        gt = s["ground_truth"]
        key.append(
            {
                "sample_id": sid,
                "chain_id": s["chain_id"],
                "label": s["label"],
                "type_codes": [] if s["label"] == "benign" else type_codes(s),
                "anomaly_type": gt.get("anomaly_type"),
                "sub_category": gt.get("sub_category"),
                "type_id": gt.get("type_id"),
            }
        )

    # --- write outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    (OUT_DIR / "samples_for_iaa.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "purpose": "inter-annotator agreement study (E2/A2)",
                    "sampling_seed": SEED,
                    "total": len(annotator_view),
                    "benign": N_BENIGN,
                    "abnormal": N_ABNORMAL,
                    "sampling_policy": (
                        "benign: round-robin stratified by scenario_group and "
                        "chain_length; abnormal: proportional over the 16 raw "
                        "Types (floor 8), round-robin by Sub_Category, "
                        "scenario_group and chain_length within each Type"
                    ),
                },
                "samples": annotator_view,
            },
            ensure_ascii=False,
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )

    (OUT_DIR / "iaa_key.json").write_text(
        json.dumps(
            {
                "metadata": {"sampling_seed": SEED, "total": len(key)},
                "items": key,
            },
            ensure_ascii=False,
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )

    n_batches = (len(annotator_view) + BATCH_SIZE - 1) // BATCH_SIZE
    for b in range(n_batches):
        chunk = annotator_view[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
        (BATCH_DIR / f"batch_{b + 1:02d}.json").write_text(
            json.dumps({"batch": b + 1, "samples": chunk}, ensure_ascii=False),
            encoding="utf-8",
        )

    # --- summary + leakage self-check
    print(f"\ntotal={len(annotator_view)} benign={N_BENIGN} abnormal={N_ABNORMAL}")
    print(f"batches: {n_batches} files of <= {BATCH_SIZE} samples in {BATCH_DIR}")
    code_dist = Counter(
        c for k in key for c in k["type_codes"]
    )
    print("component code distribution:", dict(sorted(code_dist.items())))
    banned = {"label", "ground_truth", "chain_id", "chain_type", "method",
              "scenario_group", "description", "benign_description"}
    leaked = banned & set().union(*(s.keys() for s in annotator_view))
    assert not leaked, f"LEAKAGE: {leaked} present in annotator view"
    sizes = [p.stat().st_size for p in sorted(BATCH_DIR.glob("batch_*.json"))]
    print(f"batch file sizes: min={min(sizes)}B max={max(sizes)}B mean={sum(sizes)//len(sizes)}B")
    print("leakage self-check passed (no GT fields in annotator view)")


if __name__ == "__main__":
    main()
