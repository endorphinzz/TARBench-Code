"""风格匹配评测集，第 2 步：claude-sonnet-4-5 独立验证改写片段的语义等价性。

验证者看到：原片段、改写片段、威胁类型与标注描述、改写映射表。
判定项：(1) 规则数与顺序 (2) 逐规则触发/动作逻辑等价（允许 environment_* 映射）
(3) 威胁因果结构保全。输出 pass/fail + 理由。未通过的样本进入第 3 步。

输出: data/verify_results.jsonl"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PARA = BASE_DIR / "data" / "paraphrase_results.jsonl"
GT_PATH = BASE_DIR.parents[1] / "experiments" / "icl" / "datasets" / "icl_eval_500_250.json"
OUT = BASE_DIR / "data" / "verify_results.jsonl"

API_KEY = os.environ.get("TARBENCH_API_KEY", "")
API_BASE = os.environ.get("TARBENCH_BASE_URL", "https://api.poixe.com/v1").rstrip("/")
MODEL = "claude-sonnet-4-5"
COST_CAP_USD = 5.0
PRICE_IN, PRICE_OUT = 2.40 / 1e6, 12.0 / 1e6

SYSTEM = """You are an independent auditor. A threat fragment of a smart-home trigger-action benchmark was rewritten for surface style. Your ONLY job: decide whether the rewrite preserves the fragment's semantics exactly.

Check, in order:
1. Same number of rules, same order.
2. For each rule, the rewritten trigger condition and action express the same logic as the original. The ONLY permitted device substitution is the one declared in the mappings (environment_time/environment_user/environment_weather -> concrete host device); everything else must keep device, attribute, operator, and literal meaning. Room changes are permitted only when the declared mapping or an equivalent host pairing requires them AND the causal role is unchanged.
3. The rule-to-rule causal structure (what triggers what, which chain establishes which state) is unchanged, and the overall threat is the same threat.
4. No added, removed, or merged conditions; duplicates stay duplicated.

Be strict: if any doubt exists about semantic equivalence, fail it.

Output STRICT JSON only: {"pass": true/false, "reason": "one short sentence"}"""


def call_api(model, system, user, timeout=240):
    payload = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    req = urllib.request.Request(
        API_BASE + "/chat/completions", data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json",
                 "User-Agent": "curl/8.0", "Accept": "application/json"}, method="POST")
    last = None
    for _ in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}"
            if e.code not in (408, 409, 429, 500, 502, 503, 504):
                break
        except Exception as e:
            last = str(e)
        time.sleep(5 * (_ + 1))
    return {"error": last}


def parse_json(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:])
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        s, e = cleaned.find("{"), cleaned.rfind("}")
        if s != -1 and e > s:
            try:
                return json.loads(cleaned[s:e + 1])
            except json.JSONDecodeError:
                return None
    return None


def main():
    if not API_KEY:
        sys.exit("missing TARBENCH_API_KEY")
    gt_samples = {s["sample_id"]: s for s in
                  (lambda d: d["samples"] if isinstance(d, dict) else d)(json.load(open(GT_PATH, encoding="utf-8-sig")))}
    paras = [json.loads(l) for l in PARA.open(encoding="utf-8") if json.loads(l).get("ok")]
    done = set()
    if OUT.exists():
        for line in OUT.open(encoding="utf-8"):
            try:
                done.add(json.loads(line)["sample_id"])
            except Exception:
                pass
    todo = [p for p in paras if p["sample_id"] not in done]
    print(f"to verify: {len(todo)} (of {len(paras)} rewritten ok)", flush=True)

    def verify(p):
        s = gt_samples[p["sample_id"]]
        gt = s["ground_truth"]
        user = (
            f"Threat type: {gt.get('anomaly_type')} / {gt.get('sub_category')}\n"
            f"Ground-truth description: {gt.get('description')}\n\n"
            f"Rewrite mappings used: {json.dumps(p.get('mappings', {}), ensure_ascii=False)}\n\n"
            f"ORIGINAL FRAGMENT:\n" + "\n".join(p["original_rules"]) + "\n\n"
            f"REWRITTEN FRAGMENT:\n" + "\n".join(p["rewritten_rules"])
        )
        resp = call_api(MODEL, SYSTEM, user)
        if "error" in resp:
            return {"sample_id": p["sample_id"], "verdict": "error", "reason": resp["error"], "in": 0, "out": 0}
        u = resp.get("usage", {})
        v = parse_json(resp["choices"][0]["message"]["content"])
        if v is None or "pass" not in v:
            return {"sample_id": p["sample_id"], "verdict": "error", "reason": "unparseable",
                    "in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0)}
        return {"sample_id": p["sample_id"], "verdict": "pass" if v["pass"] else "fail",
                "reason": v.get("reason", ""), "in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0)}

    spent = 0.0
    with OUT.open("a", encoding="utf-8") as sink:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(verify, p) for p in todo]
            for i, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                spent += r.get("in", 0) * PRICE_IN + r.get("out", 0) * PRICE_OUT
                sink.write(json.dumps(r, ensure_ascii=False) + "\n")
                sink.flush()
                if i % 25 == 0 or i == len(todo):
                    print(f"[{i}/{len(todo)}] spent=${spent:.2f}", flush=True)
                if spent > COST_CAP_USD:
                    print("cost cap reached; stopping", flush=True)
                    break
    verdicts = {}
    for line in OUT.open(encoding="utf-8"):
        v = json.loads(line)
        verdicts[v["verdict"]] = verdicts.get(v["verdict"], 0) + 1
    print("verdicts:", verdicts, f"spent=${spent:.2f}", flush=True)


if __name__ == "__main__":
    main()
