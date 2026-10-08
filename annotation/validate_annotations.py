"""E2: quick completeness/validity check of annotator JSONL outputs.

Usage: python annotation/validate_annotations.py
Checks per annotator: batch file count, 25 lines per file, sample_id coverage
against iaa_key.json, uniqueness, and valid label/type values.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANNO_DIR = Path(__file__).resolve().parent
CODES = {"AC", "AD", "AR", "EL", "CB", "AM", "IS", "BE", "TM", "AB", "CBk", "CV", "PE", "PD"}

key_ids = {it["sample_id"] for it in json.loads((ANNO_DIR / "iaa_key.json").read_text(encoding="utf-8"))["items"]}
print(f"key samples: {len(key_ids)}")

for adir in sorted((ANNO_DIR / "annotations").glob("annotator_*")):
    recs = {}
    problems = []
    files = sorted(adir.glob("batch_*.jsonl"))
    for f in files:
        lines = [l for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(lines) != 25:
            problems.append(f"{f.name}: {len(lines)} lines (expected 25)")
        for l in lines:
            r = json.loads(l)
            sid = r.get("sample_id")
            if sid in recs:
                problems.append(f"duplicate {sid} in {f.name}")
            recs[sid] = r
            label = str(r.get("label"))
            if label.lower().startswith("ben"):
                if r.get("type") not in (None, "BEN"):
                    problems.append(f"{sid}: benign with type {r.get('type')!r}")
            else:
                if r.get("type") not in CODES:
                    problems.append(f"{sid}: bad type {r.get('type')!r}")
    missing = key_ids - set(recs)
    extra = set(recs) - key_ids
    status = "OK" if not problems and not missing and not extra else "ISSUES"
    print(f"{adir.name}: {len(files)} files, {len(recs)} judgments, missing={len(missing)}, "
          f"extra={len(extra)} -> {status}")
    for p in problems[:10]:
        print(f"  ! {p}")
    if missing:
        print(f"  missing ids (first 5): {sorted(missing)[:5]}")
