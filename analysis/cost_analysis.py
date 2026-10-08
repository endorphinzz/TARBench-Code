"""Aggregate per-sample input/output token usage and cost estimates from existing
benchmark output logs (ICL / RAG runs).

The runs did not log API usage fields, so input tokens are reconstructed by
re-building the exact per-batch prompts from the stored records (prompt
templates + stored ICL_Context / Retrieved_Context / rules) and tokenizing
them with tiktoken cl100k_base as a uniform estimator for all models.
Output tokens come from the stored Raw_Response (one response per 5-sample
batch; counted once per batch and amortized per sample).

Usage:
    python analysis/cost_analysis.py            # writes analysis/outputs/cost_analysis.json + cost_table.md
"""

import json
import sys
from pathlib import Path

import tiktoken

CODE_DIR = Path(__file__).resolve().parent.parent
EXP_DIR = CODE_DIR / "experiments"
OUT_DIR = CODE_DIR / "analysis" / "outputs"

BATCH_SIZE = 5
ENCODING = "cl100k_base"

# USD per 1M tokens: {"input": x, "output": y}. Fill with list prices before
# publishing USD columns; token statistics are independent of this table.
PRICES = {
    # "gpt-5.4": {"input": None, "output": None},
    # "deepseek-v4-flash": {"input": None, "output": None},
    # "gemini-3.1-flash-lite-preview": {"input": None, "output": None},
}

RUNS = [
    # (condition_label, model, output_file, prompt_md, mode)
    ("zero-shot", "gpt-5.4", "icl/outputs/gpt-5.4_500_250/icl_gpt-5.4_0shot.json", "icl/zero_shot_prompt.md", "icl0"),
    ("zero-shot", "deepseek-v4-flash", "icl/outputs/deepseek-v4-flash_500_250/icl_deepseek-v4-flash_0shot.json", "icl/zero_shot_prompt.md", "icl0"),
    ("zero-shot", "gemini-3.1-flash-lite-preview", "icl/outputs/gemini-3.1-flash-lite-preview_500_250/icl_gemini-3.1-flash-lite-preview_0shot.json", "icl/zero_shot_prompt.md", "icl0"),
    ("ICL-1", "gpt-5.4", "icl/outputs/gpt-5.4_500_250/icl_gpt-5.4_1shot.json", "icl/icl_prompt.md", "icln"),
    ("ICL-1", "deepseek-v4-flash", "icl/outputs/deepseek-v4-flash_500_250/icl_deepseek-v4-flash_1shot.json", "icl/icl_prompt.md", "icln"),
    ("ICL-1", "gemini-3.1-flash-lite-preview", "icl/outputs/gemini-3.1-flash-lite-preview_500_250/icl_gemini-3.1-flash-lite-preview_1shot.json", "icl/icl_prompt.md", "icln"),
    ("ICL-3", "gpt-5.4", "icl/outputs/gpt-5.4_500_250/icl_gpt-5.4_3shot.json", "icl/icl_prompt.md", "icln"),
    ("ICL-3", "deepseek-v4-flash", "icl/outputs/deepseek-v4-flash_500_250/icl_deepseek-v4-flash_3shot.json", "icl/icl_prompt.md", "icln"),
    ("ICL-3", "gemini-3.1-flash-lite-preview", "icl/outputs/gemini-3.1-flash-lite-preview_500_250/icl_gemini-3.1-flash-lite-preview_3shot.json", "icl/icl_prompt.md", "icln"),
    ("ICL-5", "gpt-5.4", "icl/outputs/gpt-5.4_500_250/icl_gpt-5.4_5shot.json", "icl/icl_prompt.md", "icln"),
    ("ICL-5", "deepseek-v4-flash", "icl/outputs/deepseek-v4-flash_500_250/icl_deepseek-v4-flash_5shot.json", "icl/icl_prompt.md", "icln"),
    ("ICL-5", "gemini-3.1-flash-lite-preview", "icl/outputs/gemini-3.1-flash-lite-preview_500_250/icl_gemini-3.1-flash-lite-preview_5shot.json", "icl/icl_prompt.md", "icln"),
    ("no-retrieval", "gpt-5.4", "rag/outputs/gpt-5.4_500_250/rag_gpt-5.4_without_rag.json", "rag/rag_without_rag_prompt.md", "rag0"),
    ("no-retrieval", "deepseek-v4-flash", "rag/outputs/deepseek-v4-flash_500_250/rag_deepseek-v4-flash_without_rag.json", "rag/rag_without_rag_prompt.md", "rag0"),
    ("no-retrieval", "gemini-3.1-flash-lite-preview", "rag/outputs/gemini-3.1-flash-lite-preview_500_250/rag_gemini-3.1-flash-lite-preview_without_rag.json", "rag/rag_without_rag_prompt.md", "rag0"),
    ("no-retrieval", "claude-sonnet-4-5", "rag/outputs/claude-sonnet-4-5_500_250/rag_claude-sonnet-4-5_without_rag.json", "rag/rag_without_rag_prompt.md", "rag0"),
    ("RAG", "gpt-5.4", "rag/outputs/gpt-5.4_500_250/rag_gpt-5.4.json", "rag/rag_prompt.md", "rag"),
    ("RAG", "deepseek-v4-flash", "rag/outputs/deepseek-v4-flash_500_250/rag_deepseek-v4-flash.json", "rag/rag_prompt.md", "rag"),
    ("RAG", "gemini-3.1-flash-lite-preview", "rag/outputs/gemini-3.1-flash-lite-preview_500_250/rag_gemini-3.1-flash-lite-preview.json", "rag/rag_prompt.md", "rag"),
    ("RAG", "claude-sonnet-4-5", "rag/outputs/claude-sonnet-4-5_500_250/rag_claude-sonnet-4-5.json", "rag/rag_prompt.md", "rag"),
]


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_section(text, heading):
    lines = text.splitlines()
    marker = f"## {heading}"
    start = None
    for index, line in enumerate(lines):
        if line.strip() == marker:
            start = index + 1
            break
    if start is None:
        raise ValueError(f"Missing section: {marker}")
    end = len(lines)
    for index in range(start, len(lines)):
        if lines[index].startswith("## "):
            end = index
            break
    return "\n".join(lines[start:end]).strip()


def load_prompt_template(path):
    text = Path(path).read_text(encoding="utf-8")
    return {
        "system": extract_section(text, "System Prompt"),
        "user": extract_section(text, "User Prompt"),
    }


def rules_list_to_dict(rules_list):
    out = {}
    for item in rules_list:
        rule_id, rule_text = item.split(": ", 1)
        out[rule_id] = rule_text
    return out


def sample_common_fields(record, scenario):
    return {
        "sample_index": record["Sample_Index"],
        "case_id": record["Case_ID"],
        "scenario": scenario,
        "scenario_group": record["Scenario_Group"],
        "chain_length": record["Chain_Length"],
        "rules": rules_list_to_dict(record["Rules"]),
    }


def build_zero_shot_item(record, scenario):
    return json.dumps(sample_common_fields(record, scenario), ensure_ascii=False, indent=2)


def build_icl_item(record, scenario):
    item = sample_common_fields(record, scenario)
    item["shot_count"] = record["Shot_Count"]
    item["anomaly_type_definitions"] = record["ICL_Context"]["type_definitions"]
    item["benign_reference"] = record["ICL_Context"]["benign_reference"]
    item["few_shot_examples"] = record["ICL_Context"]["few_shot_examples"]
    item["target_sample"] = {
        "scenario": scenario,
        "scenario_group": record["Scenario_Group"],
        "chain_length": record["Chain_Length"],
        "rules": rules_list_to_dict(record["Rules"]),
    }
    item.pop("scenario_group")
    item.pop("chain_length")
    item.pop("rules")
    return json.dumps(item, ensure_ascii=False, indent=2)


def build_rag_item(record, scenario, include_retrieved):
    item = sample_common_fields(record, scenario)
    if include_retrieved:
        item["retrieved_context"] = record["Retrieved_Context"]
    return json.dumps(item, ensure_ascii=False, indent=2)


def load_scenarios(dataset_path):
    data = load_json(dataset_path)
    return {s["chain_id"]: s.get("scenario") for s in data["samples"]}


def mean(values):
    return sum(values) / len(values) if values else 0.0


def analyze_run(label, model, rel_path, prompt_rel, mode, enc):
    records = load_json(EXP_DIR / rel_path)
    template = load_prompt_template(EXP_DIR / prompt_rel)
    dataset_name = "icl_dataset.json" if mode in ("icl0", "icln") else "rag_dataset.json"
    scenarios = load_scenarios(EXP_DIR / "datasets" / dataset_name)

    batches = {}
    for record in records:
        batch_id = (record["Sample_Index"] - 1) // BATCH_SIZE
        batches.setdefault(batch_id, []).append(record)

    input_tokens, output_tokens, n_samples = [], [], 0
    for batch_id in sorted(batches):
        batch = batches[batch_id]
        if mode == "icl0":
            rendered = [build_zero_shot_item(r, scenarios.get(r["Case_ID"])) for r in batch]
        elif mode == "icln":
            rendered = [build_icl_item(r, scenarios.get(r["Case_ID"])) for r in batch]
        elif mode == "rag0":
            rendered = [build_rag_item(r, scenarios.get(r["Case_ID"]), False) for r in batch]
        else:
            rendered = [build_rag_item(r, scenarios.get(r["Case_ID"]), True) for r in batch]
        samples_text = "[\n" + ",\n".join(rendered) + "\n]"
        user_prompt = template["user"].replace("{samples_text}", samples_text)
        input_tokens.append(len(enc.encode(template["system"])) + len(enc.encode(user_prompt)))

        responses = {r["Raw_Response"] for r in batch}
        output_tokens.append(sum(len(enc.encode(resp)) for resp in responses))
        n_samples += len(batch)

    total_in, total_out = sum(input_tokens), sum(output_tokens)
    result = {
        "condition": label,
        "model": model,
        "runs_file": rel_path,
        "n_samples": n_samples,
        "n_calls": len(batches),
        "batch_size": BATCH_SIZE,
        "mean_input_tokens_per_call": round(mean(input_tokens), 1),
        "std_input_tokens_per_call": round(
            (sum((x - mean(input_tokens)) ** 2 for x in input_tokens) / len(input_tokens)) ** 0.5, 1
        ),
        "mean_output_tokens_per_call": round(mean(output_tokens), 1),
        "input_tokens_per_sample": round(total_in / n_samples, 1),
        "output_tokens_per_sample": round(total_out / n_samples, 1),
    }
    price = PRICES.get(model)
    if price and price.get("input") is not None and price.get("output") is not None:
        usd_per_1k = (total_in / 1e6 * price["input"] + total_out / 1e6 * price["output"]) / n_samples * 1000
        result["usd_per_1000_samples"] = round(usd_per_1k, 2)
    return result


def render_markdown(all_results):
    lines = [
        "# Token Cost Analysis (estimated, tiktoken cl100k_base)",
        "",
        "Per-sample tokens amortized over 5-sample batched calls.",
        "",
        "| Condition | Model | Input tok/sample | Output tok/sample | Mean input tok/call | Mean output tok/call |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in all_results:
        lines.append(
            f"| {r['condition']} | {r['model']} | {r['input_tokens_per_sample']:.0f} "
            f"| {r['output_tokens_per_sample']:.0f} | {r['mean_input_tokens_per_call']:.0f} "
            f"| {r['mean_output_tokens_per_call']:.0f} |"
        )
    price_models = [m for m, p in PRICES.items() if p.get("input") is not None]
    if price_models:
        lines += ["", "| Condition | Model | USD per 1,000 samples |", "|---|---|---:|"]
        for r in all_results:
            if "usd_per_1000_samples" in r:
                lines.append(f"| {r['condition']} | {r['model']} | {r['usd_per_1000_samples']:.2f} |")
    return "\n".join(lines) + "\n"


def main():
    enc = tiktoken.get_encoding(ENCODING)
    all_results = []
    for label, model, rel_path, prompt_rel, mode in RUNS:
        path = EXP_DIR / rel_path
        if not path.exists():
            print(f"SKIP (missing): {rel_path}", file=sys.stderr)
            continue
        result = analyze_run(label, model, rel_path, prompt_rel, mode, enc)
        all_results.append(result)
        print(
            f"{label:12s} {model:34s} in/sample={result['input_tokens_per_sample']:8.0f} "
            f"out/sample={result['output_tokens_per_sample']:6.0f}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "cost_analysis.json").write_text(
        json.dumps({"encoding": ENCODING, "batch_size": BATCH_SIZE, "results": all_results}, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "cost_table.md").write_text(render_markdown(all_results), encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'cost_analysis.json'} and {OUT_DIR / 'cost_table.md'}")


if __name__ == "__main__":
    main()
