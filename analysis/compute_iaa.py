"""E2 (A2): Inter-annotator agreement for the TARBench IAA study.

Computes pairwise Cohen's kappa, 3-rater Krippendorff's alpha (nominal), and
agreement with the benchmark ground truth (context statistics).

Usage:
    python analysis/compute_iaa.py [--selftest]

Reads:  annotation/iaa_key.json
        annotation/annotations/annotator_*/batch_*.jsonl
Writes: analysis/outputs/iaa_results.json
        analysis/outputs/iaa_summary.md
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANNO_DIR = ROOT / "annotation"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

CODES = ["AC", "AD", "AR", "EL", "CB", "AM", "IS", "BE", "TM", "AB", "CBk", "CV", "PE", "PD"]
CLASSES = ["BEN"] + CODES  # full label space: benign + 14 threat codes
PAIRS = [("A", "B"), ("A", "C"), ("B", "C")]


# ----------------------------- statistics ---------------------------------

def cohen_kappa(y1, y2):
    """Cohen's kappa for two equal-length label lists."""
    n = len(y1)
    assert n == len(y2) and n > 0
    po = sum(a == b for a, b in zip(y1, y2)) / n
    c1, c2 = Counter(y1), Counter(y2)
    classes = set(c1) | set(c2)
    pe = sum(c1[c] * c2[c] for c in classes) / (n * n)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0, po, pe
    return (po - pe) / (1 - pe), po, pe


def krippendorff_alpha_nominal(rows):
    """Nominal Krippendorff's alpha. rows: list of per-unit rating tuples.

    Uses ordered-pair coincidence matrix:
      o[c][d] += 1 for each ordered rater pair (i != j) within a unit.
      n = total entries; n_c = row sums.
      D_o = sum_{c != d} o[c][d] / n
      D_e = sum_{c != d} n_c * n_d / (n * (n - 1))
      alpha = 1 - D_o / D_e
    """
    o = defaultdict(Counter)
    n = 0
    for row in rows:
        vals = [v for v in row if v is not None]
        m = len(vals)
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i != j:
                    o[vals[i]][vals[j]] += 1
                    n += 1
    if n < 2:
        return None
    row_sums = {c: sum(o[c].values()) for c in o}
    d_o = sum(o[c][d] for c in o for d in o[c] if c != d) / n
    d_e = sum(row_sums[c] * row_sums[d] for c in row_sums for d in row_sums if c != d) / (n * (n - 1))
    if d_e == 0:
        return 1.0 if d_o == 0 else 0.0
    return 1 - d_o / d_e


def bootstrap_ci(rows, stat_fn, n_boot=1000, seed=20260919):
    """Percentile 95% CI for stat_fn(rows) via unit resampling."""
    rng = random.Random(seed)
    observed = stat_fn(rows)
    if observed is None or len(rows) < 5:
        return [None, None]
    stats = []
    for _ in range(n_boot):
        sample = [rows[i] for i in (rng.randrange(len(rows)) for _ in range(len(rows)))]
        v = stat_fn(sample)
        if v is not None:
            stats.append(v)
    if not stats:
        return [None, None]
    stats.sort()
    return [stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats))]]


def selftest():
    # Cohen's kappa: known toy example.
    k, po, pe = cohen_kappa(["a", "a", "b", "b"], ["a", "a", "b", "b"])
    assert abs(k - 1.0) < 1e-12, k
    k, po, pe = cohen_kappa(["a", "b", "a", "b"], ["a", "b", "b", "a"])
    assert abs(po - 0.5) < 1e-12 and abs(k - 0.0) < 1e-12, (po, k)  # symmetric marginals -> kappa 0
    # Krippendorff alpha: perfect agreement -> 1.
    assert abs(krippendorff_alpha_nominal([("a", "a", "a"), ("b", "b", "b")]) - 1.0) < 1e-12
    # Hand-computed example: D_o=8/24, D_e=288/552 -> alpha = 0.361111.
    rows = [("a", "a", "a"), ("b", "b", "b"), ("a", "b", "b"), ("a", "a", "b")]
    a = krippendorff_alpha_nominal(rows)
    assert abs(a - 0.361111) < 1e-4, a
    # Two raters, perfect disagreement, binary -> alpha = -0.5.
    a = krippendorff_alpha_nominal([("a", "b"), ("b", "a")])
    assert abs(a - (-0.5)) < 1e-9, a
    print("selftest passed")


# ------------------------------ data loading -------------------------------

def load_key():
    data = json.loads((ANNO_DIR / "iaa_key.json").read_text(encoding="utf-8"))
    return {it["sample_id"]: it for it in data["items"]}


def load_annotations():
    annos = {}
    for adir in sorted((ANNO_DIR / "annotations").glob("annotator_*")):
        name = adir.name.replace("annotator_", "")
        recs = {}
        for f in sorted(adir.glob("batch_*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                sid = r["sample_id"]
                if sid in recs:
                    raise AssertionError(f"duplicate {sid} in {adir.name}")
                label = "BEN" if str(r["label"]).lower().startswith("ben") else "ANOM"
                t = r.get("type")
                if label == "BEN":
                    t = "BEN"
                else:
                    if t not in CODES:
                        raise AssertionError(f"bad type {t!r} for {sid} in {adir.name}")
                recs[sid] = {
                    "label": label,
                    "type": t,
                    "confidence": r.get("confidence"),
                    "note": r.get("note", ""),
                }
        annos[name] = recs
    return annos


# -------------------------------- analysis ---------------------------------

def pair_kappa(annos, a, b, key_fn, sids):
    ya = [key_fn(annos[a][s]) for s in sids]
    yb = [key_fn(annos[b][s]) for s in sids]
    k, po, pe = cohen_kappa(ya, yb)
    rows = [(key_fn(annos[a][s]), key_fn(annos[b][s]), key_fn(annos[c][s]))
            for s in sids for c in [next(x for x in "ABC" if x not in (a, b))]]
    ci = bootstrap_ci([(ya[i], yb[i]) for i in range(len(sids))],
                      lambda rr: cohen_kappa([r[0] for r in rr], [r[1] for r in rr])[0])
    return {"kappa": round(k, 4), "percent_agreement": round(po, 4),
            "pe": round(pe, 4), "ci95": [round(x, 4) if x is not None else None for x in ci], "n": len(sids)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return

    key = load_key()
    annos = load_annotations()
    names = sorted(annos)
    assert names, "no annotations found"
    print(f"annotators: {names}  key samples: {len(key)}")
    for n in names:
        print(f"  {n}: {len(annos[n])} judgments")

    # Common coverage: samples judged by every annotator AND in the key.
    sids = sorted(set(key) & set.intersection(*(set(annos[n]) for n in names)))
    print(f"common coverage: {len(sids)} samples")

    binkey = lambda r: r["label"]                      # noqa: E731
    fullkey = lambda r: r["type"]                      # noqa: E731

    # ---- 1) binary label agreement
    binary = {"pairwise": {}, "alpha": None}
    for a, b in [(x, y) for x in names for y in names if x < y]:
        binary["pairwise"][f"{a}-{b}"] = pair_kappa(annos, a, b, binkey, sids)
    rows = [tuple(binkey(annos[n][s]) for n in names) for s in sids]
    binary["alpha"] = round(krippendorff_alpha_nominal(rows), 4)
    binary["alpha_ci95"] = [round(x, 4) if x is not None else None
                            for x in bootstrap_ci(rows, krippendorff_alpha_nominal)]

    # ---- 2) full 15-class agreement (BEN + 14 codes)
    full = {"pairwise": {}, "alpha": None}
    for a, b in [(x, y) for x in names for y in names if x < y]:
        full["pairwise"][f"{a}-{b}"] = pair_kappa(annos, a, b, fullkey, sids)
    rows = [tuple(fullkey(annos[n][s]) for n in names) for s in sids]
    full["alpha"] = round(krippendorff_alpha_nominal(rows), 4)

    # ---- 3) subclass agreement restricted to co-anomalous pairs
    anom_only = {"pairwise": {}, "alpha": None}
    for a, b in [(x, y) for x in names for y in names if x < y]:
        sub = [s for s in sids if annos[a][s]["label"] == "ANOM" and annos[b][s]["label"] == "ANOM"]
        if len(sub) >= 10:
            anom_only["pairwise"][f"{a}-{b}"] = pair_kappa(annos, a, b, fullkey, sub)
    trio_anom = [s for s in sids if all(annos[n][s]["label"] == "ANOM" for n in names)]
    if len(trio_anom) >= 10:
        rows = [tuple(fullkey(annos[n][s]) for n in names) for s in trio_anom]
        anom_only["alpha"] = round(krippendorff_alpha_nominal(rows), 4)
        anom_only["n"] = len(trio_anom)

    # ---- 4) per-class one-vs-rest kappa (15 classes), pairwise
    per_class = []
    for c in CLASSES:
        entry = {"class": c, "n_votes": sum(1 for s in sids for n in names if fullkey(annos[n][s]) == c)}
        entry["pair_kappa"] = {}
        for a, b in [(x, y) for x in names for y in names if x < y]:
            ya = ["C" if fullkey(annos[a][s]) == c else "R" for s in sids]
            yb = ["C" if fullkey(annos[b][s]) == c else "R" for s in sids]
            k, po, pe = cohen_kappa(ya, yb)
            entry["pair_kappa"][f"{a}-{b}"] = round(k, 4)
        per_class.append(entry)

    # ---- 5) agreement with ground truth (context statistics)
    def gt_metrics(judgments):
        tp = fp = fn = tn = 0
        type_in_set = type_total = 0
        type_exact = type_exact_total = 0
        for s, rec in judgments.items():
            gt = key[s]
            gt_anom = gt["label"] == "abnormal"
            pred_anom = rec["label"] == "ANOM"
            if pred_anom and gt_anom:
                tp += 1
                type_total += 1
                if rec["type"] in gt["type_codes"]:
                    type_in_set += 1
                if len(gt["type_codes"]) == 1 and rec["type"] == gt["type_codes"][0]:
                    type_exact_total += 1
                    type_exact += 1
                elif len(gt["type_codes"]) == 1:
                    type_exact_total += 1
            elif pred_anom and not gt_anom:
                fp += 1
            elif not pred_anom and gt_anom:
                fn += 1
            else:
                tn += 1
        prec = tp / (tp + fp) if tp + fp else None
        rec_ = tp / (tp + fn) if tp + fn else None
        f1 = 2 * prec * rec_ / (prec + rec_) if prec and rec_ else None
        return {
            "accuracy": round((tp + tn) / len(judgments), 4),
            "precision": round(prec, 4) if prec else None,
            "recall": round(rec_, 4) if rec_ else None,
            "f1": round(f1, 4) if f1 else None,
            "fp": fp, "fn": fn,
            "type_in_set_acc": round(type_in_set / type_total, 4) if type_total else None,
            "type_exact_acc": round(type_exact / type_exact_total, 4) if type_exact_total else None,
            "n": len(judgments),
        }

    vs_gt = {n: gt_metrics(annos[n]) for n in names}

    # ---- 6) majority vote (2-of-3) vs GT + adjudication load
    maj = {}
    for s in sids:
        labels = [annos[n][s]["label"] for n in names]
        lab = Counter(labels).most_common(1)[0][0]
        if lab == "ANOM":
            types = [annos[n][s]["type"] for n in names if annos[n][s]["label"] == "ANOM"]
            top, cnt = Counter(types).most_common(1)[0]
            maj[s] = {"label": "ANOM", "type": top if cnt >= 2 else None}
        else:
            maj[s] = {"label": "BEN", "type": "BEN"}
    vs_gt["majority"] = gt_metrics(maj)
    tie_type = sum(1 for s in sids if maj[s]["type"] is None)
    label_disagree = sum(1 for s in sids if len(set(annos[n][s]["label"] for n in names)) > 1)
    all_anom_type_disagree = sum(
        1 for s in sids
        if all(annos[n][s]["label"] == "ANOM" for n in names)
        and len(set(annos[n][s]["type"] for n in names)) > 1
    )
    adjudication = {
        "label_disagreement_samples": label_disagree,
        "all_anomalous_type_disagreement_samples": all_anom_type_disagree,
        "majority_type_ties": tie_type,
        "total": len(sids),
    }

    # ---- 7) adjudication review CSV (for owner review of disagreements)
    import csv
    review_dir = ANNO_DIR / "outputs"
    review_dir.mkdir(parents=True, exist_ok=True)
    csv_path = review_dir / "adjudication_review.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "gt_label", "gt_types"]
                   + [f"{n}_{col}" for n in names for col in ("label", "conf", "note")]
                   + ["majority_label", "majority_type",
                      "label_disagreement", "type_disagreement", "type_tie"])
        n_review = 0
        for s in sids:
            labs = [annos[n][s]["label"] for n in names]
            types = [annos[n][s]["type"] for n in names]
            lab_dis = len(set(labs)) > 1
            all_anom = all(l == "ANOM" for l in labs)
            typ_dis = all_anom and len(set(types)) > 1
            tie = maj[s]["label"] == "ANOM" and maj[s]["type"] is None
            if not (lab_dis or typ_dis or tie):
                continue
            n_review += 1
            gt = key[s]
            row = [s, gt["label"], "/".join(gt["type_codes"])]
            for n in names:
                row += [annos[n][s]["type"], annos[n][s]["confidence"], annos[n][s]["note"]]
            row += [maj[s]["label"], maj[s]["type"] or "TIE",
                    "Y" if lab_dis else "", "Y" if typ_dis else "", "Y" if tie else ""]
            w.writerow(row)
    print(f"adjudication review rows: {n_review} -> {csv_path}")

    # ---- outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "coverage": {"annotators": names, "n_samples": len(sids), "per_annotator": {n: len(annos[n]) for n in names}},
        "binary_label": binary,
        "full_15class": full,
        "anomalous_only_subclass": anom_only,
        "per_class_one_vs_rest": per_class,
        "vs_ground_truth": vs_gt,
        "adjudication": adjudication,
    }
    (OUT_DIR / "iaa_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )

    # markdown summary
    lines = ["# IAA Summary (E2/A2)", ""]
    lines.append(f"Coverage: {len(sids)} samples, annotators {', '.join(names)}.")
    lines.append("")
    lines.append("## Binary label (Benign vs Anomalous)")
    lines.append("| pair | kappa | agree | 95% CI |")
    lines.append("|---|---|---|---|")
    for k, v in binary["pairwise"].items():
        ci = "-".join(str(x) for x in v["ci95"])
        lines.append(f"| {k} | {v['kappa']} | {v['percent_agreement']} | {ci} |")
    lines.append(f"")
    lines.append(f"Krippendorff's alpha (3 raters): **{binary['alpha']}** (CI {binary['alpha_ci95']})")
    lines.append("")
    lines.append("## Full label (Benign + 14 types)")
    for k, v in full["pairwise"].items():
        lines.append(f"- {k}: kappa = {v['kappa']}, agree = {v['percent_agreement']}")
    lines.append(f"- Krippendorff's alpha (3 raters): {full['alpha']}")
    if anom_only.get("alpha") is not None:
        lines.append("")
        lines.append(f"## Subclass agreement (co-anomalous samples, n={anom_only.get('n')})")
        for k, v in anom_only["pairwise"].items():
            lines.append(f"- {k}: kappa = {v['kappa']}, agree = {v['percent_agreement']}, n={v['n']}")
        lines.append(f"- alpha = {anom_only['alpha']}")
    lines.append("")
    lines.append("## Per-class one-vs-rest kappa")
    lines.append("| class | votes | " + " | ".join(f"{a}-{b}" for a, b in PAIRS) + " |")
    lines.append("|---|---|" + "---|" * len(PAIRS))
    for e in per_class:
        if e["n_votes"] == 0:
            continue
        lines.append(f"| {e['class']} | {e['n_votes']} | " + " | ".join(str(e["pair_kappa"][f"{a}-{b}"]) for a, b in PAIRS) + " |")
    lines.append("")
    lines.append("## vs ground truth")
    lines.append("| annotator | acc | P | R | F1 | type-in-set | FP | FN |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for n in names + ["majority"]:
        m = vs_gt[n]
        lines.append(
            f"| {n} | {m['accuracy']} | {m['precision']} | {m['recall']} | {m['f1']} | "
            f"{m['type_in_set_acc']} | {m['fp']} | {m['fn']} |"
        )
    lines.append("")
    lines.append("## Adjudication load")
    lines.append(f"- label disagreements: {label_disagree}/{len(sids)}")
    lines.append(f"- type disagreements among all-anomalous: {all_anom_type_disagree}")
    lines.append(f"- majority type ties: {tie_type}")
    (OUT_DIR / "iaa_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # console digest
    print("\n== binary ==")
    for k, v in binary["pairwise"].items():
        print(f"  {k}: kappa={v['kappa']} agree={v['percent_agreement']}")
    print(f"  alpha={binary['alpha']}")
    print("== full 15-class ==")
    for k, v in full["pairwise"].items():
        print(f"  {k}: kappa={v['kappa']}")
    print(f"  alpha={full['alpha']}")
    print("== vs GT ==")
    for n in names + ["majority"]:
        m = vs_gt[n]
        print(f"  {n}: acc={m['accuracy']} F1={m['f1']} type-in-set={m['type_in_set_acc']}")
    print(f"\nwrote {OUT_DIR / 'iaa_results.json'} and iaa_summary.md")


if __name__ == "__main__":
    main()
