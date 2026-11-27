"""风格匹配评测集，第 1 步：用 gpt-5.4 把 ICL-750 的 250 个异常片段改写为宿主风格。

语义保全（硬约束）+ 表面风格迁移（environment_* 伪设备映射、格式与房间-设备搭配对齐）。
断点续跑，逐条落盘，累计费用护栏。环境变量: TARBENCH_API_KEY / TARBENCH_BASE_URL。

输出: data/paraphrase_results.jsonl（每行一个样本的结果）"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parents[1]
DATASET = ROOT / "experiments" / "icl" / "datasets" / "icl_eval_500_250.json"
OUT = BASE_DIR / "data" / "paraphrase_results.jsonl"

API_KEY = os.environ.get("TARBENCH_API_KEY", "")
API_BASE = os.environ.get("TARBENCH_BASE_URL", "https://api.poixe.com/v1").rstrip("/")
MODEL = "gpt-5.4"
COST_CAP_USD = 8.0
PRICE_IN, PRICE_OUT = 1.75 / 1e6, 10.5 / 1e6  # 实测校准价

SYSTEM = """You rewrite anomalous rule fragments of a smart-home trigger-action benchmark so that their surface style matches the surrounding host rules, WITHOUT changing what they mean.

Hard constraints (semantics must be preserved exactly):
1. Keep the number of rules and their order exactly as given.
2. Each rewritten rule must express the same trigger condition and the same action as the original: same device (unless rule 5 applies), same attribute semantics, same operator, same literal value meaning.
3. The rule-to-rule causal/interaction structure must be preserved exactly.
4. Never add or remove conditions, devices, actions, or duplicate rules.

Style transfer (surface only):
5. Replace EVERY pseudo-device named environment_time, environment_user, or environment_weather with a concrete device that (a) plays the same role and (b) is written in the host vocabulary shown below. Examples: a time trigger becomes <room>_alarm_clock.alarm_clock_time with a concrete time literal similar to host times; user presence becomes <room>_motion_sensor.motion_sensor_value == true; weather becomes a host-style sensor with an equivalent threshold. Record each replacement in "mappings".
6. Match host formatting exactly: "IF <device>.<attribute> <op> <literal> THEN <device>.<attribute> = <literal>", lowercase, single spaces.
7. Prefer room+device pairings that already occur in the host rules when the threat logic allows a room change; otherwise keep the original room and device.
8. Keep "AND" only if the original uses it. Never introduce or remove conjuncts.

Output STRICT JSON only, no markdown fences:
{"rewritten_rules": ["IF ...", "..."], "mappings": {"environment_time": "room_alarm_clock", ...}, "notes": "one short sentence"}"""


def load_samples():
    d = json.load(open(DATASET, encoding="utf-8-sig"))
    samples = d["samples"] if isinstance(d, dict) else d
    return [s for s in samples if s.get("label") == "abnormal"]


def fragment_split(sample):
    gt = sample["ground_truth"]
    st, ln = int(gt["abnormal_fragment_start_index"]), int(gt["abnormal_fragment_length"])
    frag_ids = [f"Rule_{i}" for i in range(st, st + ln)]
    host_ids = [rid for rid in sorted(sample["rules"], key=lambda k: int(k.split("_")[1])) if rid not in frag_ids]
    frag = [sample["rules"][rid] for rid in frag_ids]
    host = [sample["rules"][rid] for rid in host_ids]
    return frag_ids, host_ids, frag, host


def call_api(model, system, user, temperature=0, timeout=240):
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    if temperature is not None and not model.startswith("claude-"):
        payload["temperature"] = temperature
    req = urllib.request.Request(
        API_BASE + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "curl/8.0",
            "Accept": "application/json",
        },
        method="POST",
    )
    last = None
    for _ in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:200]
            last = f"HTTP {e.code}: {body}"
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


def valid_rewrite(frag, result):
    rules = result.get("rewritten_rules")
    if not isinstance(rules, list) or len(rules) != len(frag):
        return False, f"rule count {len(rules) if isinstance(rules, list) else '?'} != {len(frag)}"
    for r in rules:
        if not isinstance(r, str) or not re.match(r"^IF .+ THEN .+$", r):
            return False, f"malformed rule: {r[:60]}"
    if any(re.search(r"environment_(time|user|weather)", r) for r in rules):
        return False, "environment_* pseudo-device remains"
    return True, ""


def process(sample):
    frag_ids, host_ids, frag, host = fragment_split(sample)
    gt = sample["ground_truth"]
    user = (
        f"Scenario: {sample.get('scenario')}\n"
        f"Threat type: {gt.get('anomaly_type')} / {gt.get('sub_category')}\n"
        f"Ground-truth description: {gt.get('description')}\n\n"
        f"HOST RULES (style reference, do not modify):\n" + "\n".join(host) + "\n\n"
        f"FRAGMENT RULES ({len(frag)} rules, to rewrite):\n" + "\n".join(frag)
    )
    usage_tokens = {"in": 0, "out": 0}
    feedback = ""
    for attempt in range(3):
        prompt = user + (f"\n\nPrevious attempt rejected: {feedback}\nProduce a corrected rewrite." if feedback else "")
        resp = call_api(MODEL, SYSTEM, prompt)
        if "error" in resp:
            return {"sample_id": sample["sample_id"], "ok": False, "error": resp["error"], **usage_tokens}
        u = resp.get("usage", {})
        usage_tokens["in"] += u.get("prompt_tokens", 0)
        usage_tokens["out"] += u.get("completion_tokens", 0)
        result = parse_json(resp["choices"][0]["message"]["content"])
        if result is None:
            feedback = "output was not valid JSON"
            continue
        ok, why = valid_rewrite(frag, result)
        if ok:
            return {"sample_id": sample["sample_id"], "ok": True,
                    "fragment_rule_ids": frag_ids,
                    "original_rules": frag, "rewritten_rules": result["rewritten_rules"],
                    "mappings": result.get("mappings", {}), "notes": result.get("notes", ""),
                    "attempts": attempt + 1, **usage_tokens}
        feedback = why
    return {"sample_id": sample["sample_id"], "ok": False, "error": f"3 attempts failed: {feedback}", **usage_tokens}


def main():
    if not API_KEY:
        sys.exit("missing TARBENCH_API_KEY")
    samples = load_samples()
    done = set()
    if OUT.exists():
        for line in OUT.open(encoding="utf-8"):
            try:
                done.add(json.loads(line)["sample_id"])
            except Exception:
                pass
    todo = [s for s in samples if s["sample_id"] not in done]
    print(f"total {len(samples)} fragments, done {len(done)}, todo {len(todo)}", flush=True)

    spent = 0.0
    lock_written = 0
    with OUT.open("a", encoding="utf-8") as sink:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(process, s): s for s in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                spent += r.get("in", 0) * PRICE_IN + r.get("out", 0) * PRICE_OUT
                sink.write(json.dumps(r, ensure_ascii=False) + "\n")
                sink.flush()
                if i % 20 == 0 or i == len(todo):
                    ok = sum(1 for line in OUT.open(encoding="utf-8")
                             if json.loads(line).get("ok"))
                    print(f"[{i}/{len(todo)}] ok={ok} spent=${spent:.2f}", flush=True)
                if spent > COST_CAP_USD:
                    print(f"cost cap reached (${spent:.2f}); stopping", flush=True)
                    pool.shutdown(wait=False, cancel_futures=True)
                    break
    print(f"phase P finished, approx spent ${spent:.2f}", flush=True)


if __name__ == "__main__":
    main()
