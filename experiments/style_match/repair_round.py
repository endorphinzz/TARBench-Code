"""风格匹配评测集，第 3 步：修复轮。

- verdict=error 的样本 -> 重新验证；
- verdict=fail 的样本 -> 带上验证失败理由重新改写（第 1 步协议），再验证。
结果追加写回两个 jsonl（读取时按 sample_id 取最后一行）。"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_paraphrase import call_api, parse_json, valid_rewrite, fragment_split, SYSTEM  # noqa: E402
from run_verify import SYSTEM as V_SYSTEM  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
PARA = BASE_DIR / "data" / "paraphrase_results.jsonl"
VER = BASE_DIR / "data" / "verify_results.jsonl"
GT_PATH = BASE_DIR.parents[1] / "experiments" / "icl" / "datasets" / "icl_eval_500_250.json"

REWRITER = "gpt-5.4"
VERIFIER = "claude-sonnet-4-5"


def load_rows(path):
    rows = {}
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        rows[r["sample_id"]] = r
    return rows


def verify_once(sample, para):
    gt = sample["ground_truth"]
    user = (
        f"Threat type: {gt.get('anomaly_type')} / {gt.get('sub_category')}\n"
        f"Ground-truth description: {gt.get('description')}\n\n"
        f"Rewrite mappings used: {json.dumps(para.get('mappings', {}), ensure_ascii=False)}\n\n"
        f"ORIGINAL FRAGMENT:\n" + "\n".join(para["original_rules"]) + "\n\n"
        f"REWRITTEN FRAGMENT:\n" + "\n".join(para["rewritten_rules"])
    )
    resp = call_api(VERIFIER, V_SYSTEM, user)
    if "error" in resp:
        return {"sample_id": sample["sample_id"], "verdict": "error", "reason": resp["error"], "in": 0, "out": 0}
    u = resp.get("usage", {})
    v = parse_json(resp["choices"][0]["message"]["content"])
    if v is None or "pass" not in v:
        return {"sample_id": sample["sample_id"], "verdict": "error", "reason": "unparseable",
                "in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0)}
    return {"sample_id": sample["sample_id"], "verdict": "pass" if v["pass"] else "fail",
            "reason": v.get("reason", ""), "in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0)}


def regenerate(sample, reason):
    frag_ids, host_ids, frag, host = fragment_split(sample)
    gt = sample["ground_truth"]
    user = (
        f"Scenario: {sample.get('scenario')}\n"
        f"Threat type: {gt.get('anomaly_type')} / {gt.get('sub_category')}\n"
        f"Ground-truth description: {gt.get('description')}\n\n"
        f"HOST RULES (style reference, do not modify):\n" + "\n".join(host) + "\n\n"
        f"FRAGMENT RULES ({len(frag)} rules, to rewrite):\n" + "\n".join(frag) + "\n\n"
        f"A previous rewrite was rejected by an independent auditor: {reason}\n"
        f"Produce a corrected rewrite that satisfies the auditor's concern while following all constraints."
    )
    for attempt in range(3):
        resp = call_api(REWRITER, SYSTEM, user)
        if "error" in resp:
            continue
        u = resp.get("usage", {})
        result = parse_json(resp["choices"][0]["message"]["content"])
        if result is None:
            continue
        ok, why = valid_rewrite(frag, result)
        if ok:
            return {"sample_id": sample["sample_id"], "ok": True, "fragment_rule_ids": frag_ids,
                    "original_rules": frag, "rewritten_rules": result["rewritten_rules"],
                    "mappings": result.get("mappings", {}), "notes": result.get("notes", ""),
                    "attempts": attempt + 1, "in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0)}
    return None


def main():
    gt_samples = {s["sample_id"]: s for s in
                  (lambda d: d["samples"] if isinstance(d, dict) else d)(json.load(open(GT_PATH, encoding="utf-8-sig")))}
    paras = load_rows(PARA)
    vers = load_rows(VER)

    errors = [sid for sid, v in vers.items() if v["verdict"] == "error"]
    fails = [sid for sid, v in vers.items() if v["verdict"] == "fail"]
    print(f"repair: {len(errors)} errors to re-verify, {len(fails)} fails to regenerate")

    with PARA.open("a", encoding="utf-8") as psink, VER.open("a", encoding="utf-8") as vsink:
        for sid in errors:
            r = verify_once(gt_samples[sid], paras[sid])
            vsink.write(json.dumps(r, ensure_ascii=False) + "\n"); vsink.flush()
        for sid in fails:
            reason = vers[sid].get("reason", "unknown")
            newp = regenerate(gt_samples[sid], reason)
            if newp is None:
                vsink.write(json.dumps({"sample_id": sid, "verdict": "fail",
                                        "reason": "regeneration failed"}, ensure_ascii=False) + "\n"); vsink.flush()
                continue
            psink.write(json.dumps(newp, ensure_ascii=False) + "\n"); psink.flush()
            v = verify_once(gt_samples[sid], newp)
            vsink.write(json.dumps(v, ensure_ascii=False) + "\n"); vsink.flush()

    final = load_rows(VER)
    from collections import Counter
    print("final verdicts:", Counter(v["verdict"] for v in final.values()))


if __name__ == "__main__":
    main()
