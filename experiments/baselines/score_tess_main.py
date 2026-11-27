"""TESS 解释评分脚本（ICL/RAG 运行器格式）。

对逐样本预测文件 (icl_{model}_{n}shot.json / rag_*.json) 中被判为 anomalous 的 TP 样本，
用独立的 judge 模型按论文附录的 1-5 语义一致性协议对解释打分。协议与
finetune/score_tess.py 相同；judge 默认 gemini-2.5-pro（主表判分模型），可用 --judge 覆盖。

用法:
  python score_tess_main.py --pred ../icl/outputs/gpt-5.4_500_250/icl_gpt-5.4_0shot.json [--judge gemini-2.5-pro] [--limit 50]
  环境变量: TARBENCH_API_KEY / TARBENCH_BASE_URL
输出: {pred 同名}_tess.json（逐项 + 汇总）"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FINETUNE = BASE_DIR.parent / "finetune"
sys.path.insert(0, str(FINETUNE))

# 与论文一致的 TESS 系统 prompt（与 finetune/score_tess.py 同源，逐字复刻）
from score_tess import SYSTEM_PROMPT  # noqa: E402


def load_config(args):
    return {
        "api_key": args.api_key or __import__("os").environ.get("TARBENCH_API_KEY", ""),
        "base_url": (args.base_url or __import__("os").environ.get("TARBENCH_BASE_URL", "")).rstrip("/"),
    }


def call_judge(cfg, model, system, user, timeout=180):
    payload = {"model": model, "messages": [
        {"role": "system", "content": system}, {"role": "user", "content": user}]}
    if not model.startswith("claude-"):
        payload["temperature"] = 0
    req = urllib.request.Request(
        cfg["base_url"] + "/chat/completions", data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json",
                 "User-Agent": "curl/8.0", "Accept": "application/json"}, method="POST")
    last = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code not in (408, 409, 429, 500, 502, 503, 504):
                break
        except Exception as e:
            last = str(e)
        time.sleep(3 * (attempt + 1))
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True, help="ICL/RAG 格式逐样本预测 json")
    ap.add_argument("--judge", default="gemini-2.5-pro", help="judge 模型（网关型号）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--api-key"); ap.add_argument("--base-url")
    args = ap.parse_args()
    cfg = load_config(args)
    if not cfg["api_key"] or not cfg["base_url"]:
        sys.exit("缺少 TARBENCH_API_KEY/TARBENCH_BASE_URL")

    items = json.load(open(args.pred, encoding="utf-8"))
    cache_path = args.pred.with_name("." + args.pred.stem + "_tess_cache.json")
    cache = json.load(open(cache_path, encoding="utf-8")) if cache_path.exists() else {}

    scored, pending = 0, []
    for it in items:
        gt = it.get("Ground_Truth") or {}
        pred = (it.get("LLM_Prediction") or {})
        if gt.get("has_anomaly") is not True or pred.get("has_anomaly") is not True:
            continue
        reason = str(pred.get("reason") or "").strip()
        if not reason:
            continue
        pending.append((it.get("Case_ID") or it.get("sample_id") or str(scored),
                        f"1.[{gt.get('anomaly_type')}]: {gt.get('description')}", reason))
        scored += 1
    if args.limit:
        pending = pending[:args.limit]
    print(f"TP-with-explanation samples: {scored}; scoring {len(pending)}")

    out_path = args.pred.with_name(args.pred.stem + "_tess.json")
    results = []
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=4) as pool, open(out_path, "w", encoding="utf-8") as sink:
        def one(item):
            cid, gold, cand = item
            if cid in cache:
                return {"case_id": cid, **cache[cid]}
            resp = call_judge(cfg, args.judge, SYSTEM_PROMPT, f"A (reference_reasoning):\n{gold}\n\nB (candidate_reasoning):\n{cand}")
            v = parse_json(resp["choices"][0]["message"]["content"]) if "choices" in resp else None
            entry = {"score": v.get("score") if v else None, "critique": v.get("critique") if v else "error",
                     "error": resp.get("error")}
            cache[cid] = entry
            return {"case_id": cid, **entry}

        for i, fut in enumerate(cf.as_completed([pool.submit(one, p) for p in pending]), 1):
            r = fut.result()
            results.append(r)
            sink.write(json.dumps(r, ensure_ascii=False) + "\n"); sink.flush()
            if i % 25 == 0:
                print(f"[{i}/{len(pending)}]")
    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    valid = [r["score"] for r in results if isinstance(r.get("score"), (int, float))]
    if valid:
        print(f"TESS mean: {sum(valid)/len(valid):.2f} over {len(valid)} scored (judge={args.judge})")


if __name__ == "__main__":
    main()
