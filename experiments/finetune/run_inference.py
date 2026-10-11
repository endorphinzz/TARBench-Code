"""E9 vLLM 批量推理 -> 输出 evaluate_icl.py 直接可用的预测文件。

与 SFT 训练严格同源: prompt 由 datasets/prompt_contract.json 渲染,
chat template 关闭 Qwen3 thinking, 贪心解码。

用法 (云端 venv_inf 环境):
    python experiments/finetune/run_inference.py \
        --model /root/models/Qwen3-8B \
        --adapter experiments/finetune/saves/e9_m1_s1 \
        --split-file experiments/finetune/datasets/splits/test_t1.json \
        --output experiments/finetune/outputs/pred_t1_e9_m1_s1.json \
        --tag qwen3-8b-ft-m1-s1

评测 (本地或云端, 零改动复用):
    python experiments/icl/evaluate_icl.py experiments/finetune/outputs/pred_t1_e9_m1_s1.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FT_DIR = Path(__file__).resolve().parent
CONTRACT_PATH = FT_DIR / "datasets" / "prompt_contract.json"
RULE_RE = re.compile(r"Rule_\d+")


def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def render_user(sample: dict, template: dict) -> str:
    item = json.dumps(
        {
            "sample_index": 1,
            "case_id": sample["chain_id"],
            "scenario": sample.get("scenario"),
            "scenario_group": sample.get("scenario_group"),
            "chain_length": sample.get("chain_length"),
            "rules": sample["rules"],
        },
        ensure_ascii=False,
        indent=2,
    )
    return template["user_template"].replace("{samples_text}", "[\n" + item + "\n]")


def parse_result_json(text: str) -> dict | None:
    t = text.strip()
    # thinking 模式: 只保留 </think> 之后的内容 (未闭合则视为截断失败)
    if "</think>" in t:
        t = t.rsplit("</think>", 1)[1].strip()
    elif "<think>" in t:
        return None
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(t[start:end + 1])
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and isinstance(obj.get("results"), list) and obj["results"]:
        return obj["results"][0]
    if isinstance(obj, dict) and "has_anomaly" in obj:
        return obj
    return None


def gt_involved_rules(sample: dict) -> list[str]:
    """与 evaluate_icl.ground_truth_involved_rules 同口径。"""
    gt = sample.get("ground_truth", {})
    if gt.get("has_anomaly") is not True:
        return []
    explicit = sorted(
        RULE_RE.findall(" ".join(str(x) for x in gt.get("involved_rules", []) or [])),
        key=lambda k: int(k.split("_")[1]),
    )
    if explicit:
        return explicit
    start, length = gt.get("abnormal_fragment_start_index"), gt.get("abnormal_fragment_length")
    if start is None or length is None:
        return []
    return [f"Rule_{i}" for i in range(int(start), int(start) + int(length))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="base 模型路径")
    ap.add_argument("--adapter", default=None, help="LoRA adapter 目录 (缺省 = 纯 base 推理, 作 0-shot 对照)")
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tag", default="qwen3-8b")
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--max-model-len", type=int, default=5120)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--limit", type=int, default=0, help=">0 时仅推理前 N 条 (冒烟)")
    ap.add_argument("--thinking", action="store_true", help="开启 Qwen3 thinking (base 模型对照用; FT 默认关)")
    args = ap.parse_args()

    contract = load_json(CONTRACT_PATH)
    samples = load_json(Path(args.split_file))["samples"]
    if args.limit > 0:
        samples = samples[: args.limit]

    messages = [
        [
            {"role": "system", "content": contract["system"]},
            {"role": "user", "content": render_user(s, contract)},
        ]
        for s in samples
    ]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    def render(msgs):
        try:  # Qwen3 模板支持 enable_thinking
            return tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=args.thinking)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    prompts = [render(m) for m in messages]

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        enable_lora=bool(args.adapter),
        max_lora_rank=32,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    lora_req = LoRARequest("e9_adapter", 1, args.adapter) if args.adapter else None
    outputs = llm.generate(prompts, sp, lora_request=lora_req)

    items, parse_fail = [], 0
    for i, (s, out) in enumerate(zip(samples, outputs)):
        pred = parse_result_json(out.outputs[0].text)
        if pred is None:
            parse_fail += 1
            pred = {}
        gt = s.get("ground_truth", {})
        items.append({
            "Model": args.tag,
            "Shot_Count": "ft" if args.adapter else 0,
            "Sample_Index": s.get("sample_index", i + 1),
            "Sample_ID": s.get("sample_id"),
            "Scenario_Group": s.get("scenario_group"),
            "Chain_Length": s.get("chain_length"),
            "Ground_Truth": {
                "has_anomaly": gt.get("has_anomaly") is True,
                "anomaly_type": gt.get("anomaly_type"),
                "sub_category": gt.get("sub_category"),
                "involved_rules": gt_involved_rules(s),
                "abnormal_fragment_start_index": gt.get("abnormal_fragment_start_index"),
                "abnormal_fragment_length": gt.get("abnormal_fragment_length"),
            },
            "LLM_Prediction": {
                "has_anomaly": pred.get("has_anomaly"),
                "reason": str(pred.get("reason", "")),
                "involved_rules": pred.get("involved_rules", []),
            },
        })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    n_ab = sum(1 for x in items if x["Ground_Truth"]["has_anomaly"])
    print(f"[infer] wrote {out_path} | n={len(items)} (abnormal {n_ab}) | parse_fail={parse_fail}")
    if parse_fail:
        print("[warn] 存在解析失败样本 -> 评测按 unknown 计, 建议抽查输出文本")


if __name__ == "__main__":
    main()
