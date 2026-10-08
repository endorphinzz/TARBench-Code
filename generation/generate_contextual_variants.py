#!/usr/bin/env python3
"""E5-6 情境化变体生成：设备共享（device-shared）版异常链。

与发布版 build_abnormal_chains.py 的唯一构造差异：
  候选宿主从「设备 + env 参数完全互斥」放宽为「与异常片段共享 >=1 个设备名」，
  其余（随机插入位置、描述改写、格式化管线）完全一致。

用途（伪影复测）：发布版异常链的插入片段与宿主设备 100% 隔离。
若设备域隔离是词面捷径的来源，共享设备的变体上 B-LEX AUC 应显著下降；
若仍高，则伪影主要来自真人片段与合成宿主之间的写作风格分布偏移。

输出：
  data/abnormal/abnormal_chains_gpt54_contextual_variants.json
  analysis/outputs/contextual_variants_build_stats.json
  analysis/outputs/contextual_variants_human_check.csv  （人工抽样核对表）
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

GENERATION_DIR = Path(__file__).resolve().parent
ROOT = GENERATION_DIR.parent
sys.path.insert(0, str(GENERATION_DIR))

from build_abnormal_chains import (  # noqa: E402
    DEFAULT_ABNORMAL_INPUTS,
    DEFAULT_BENIGN,
    build_output_chain,
    build_source_index,
    format_dataset,
    load_abnormal_fragments,
    load_benign_chains,
)

DEFAULT_OUTPUT = ROOT / "data" / "abnormal" / "abnormal_chains_gpt54_contextual_variants.json"
DEFAULT_STATS_OUTPUT = ROOT / "analysis" / "outputs" / "contextual_variants_build_stats.json"
DEFAULT_CHECK_OUTPUT = ROOT / "analysis" / "outputs" / "contextual_variants_human_check.csv"
DEFAULT_TARGET_CHAINS = 300
DEFAULT_CHECK_SAMPLE = 30


def device_shared_filter(fragment: dict, benign_chain: dict) -> bool:
    """变体候选条件：与片段共享 >=1 个设备名（与发布版互斥条件正好相反）。"""
    return bool(fragment["devices"] & benign_chain["devices"])


def build_variant_chains(
    fragments: list[dict],
    benign_chains: list[dict],
    target_chains: int,
    max_insertions_per_fragment: int,
    seed: int,
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)

    candidate_lists: list[list[dict]] = []
    for fragment in fragments:
        candidate_lists.append(
            [chain for chain in benign_chains if device_shared_filter(fragment, chain)]
        )

    fragments_with_candidates = sum(1 for c in candidate_lists if c)
    candidate_count_distribution = Counter(len(c) for c in candidate_lists)

    frag_order = list(range(len(fragments)))
    rng.shuffle(frag_order)

    built_chains: list[dict] = []
    per_type_built: Counter = Counter()
    overlap_sizes: Counter = Counter()
    env_overlap_count = 0
    fragments_used: list[str] = []

    for frag_idx in frag_order:
        if len(built_chains) >= target_chains:
            break
        fragment = fragments[frag_idx]
        candidates = candidate_lists[frag_idx]
        if not candidates:
            continue

        selected_count = min(max_insertions_per_fragment, len(candidates))
        for benign_chain in rng.sample(candidates, selected_count):
            insert_pos = rng.randint(0, len(benign_chain["rules"]))
            device_overlap = sorted(fragment["devices"] & benign_chain["devices"])
            env_overlap = sorted(fragment["env_params"] & benign_chain["env_params"])
            if env_overlap:
                env_overlap_count += 1
            overlap_sizes[len(device_overlap)] += 1

            chain = build_output_chain(
                chain_idx=len(built_chains) + 1,
                fragment=fragment,
                benign_chain=benign_chain,
                insert_pos=insert_pos,
                candidate_count=len(candidates),
            )
            chain["chain_id"] = f"VA_{len(built_chains) + 1:06d}"
            chain["device_overlap"] = device_overlap
            chain["env_overlap"] = env_overlap
            built_chains.append(chain)
            per_type_built[str(fragment.get("Type"))] += 1
            fragments_used.append(str(fragment.get("Type_ID")))

        if len(built_chains) >= target_chains:
            break

    stats = {
        "target_chains": target_chains,
        "built_chains": len(built_chains),
        "shortfall": max(0, target_chains - len(built_chains)),
        "total_fragments": len(fragments),
        "total_benign_chains": len(benign_chains),
        "fragments_with_device_shared_candidates": fragments_with_candidates,
        "fragments_used": len(set(fragments_used)),
        "candidate_count_distribution": {
            str(k): v for k, v in sorted(candidate_count_distribution.items())
        },
        "device_overlap_size_distribution": {
            str(k): v for k, v in sorted(overlap_sizes.items())
        },
        "chains_with_env_overlap": env_overlap_count,
        "built_by_anomaly_type": dict(sorted(per_type_built.items())),
        "random_seed": seed,
    }
    return built_chains, stats


def write_human_check_sheet(
    chains: list[dict],
    output_path: Path,
    sample_size: int,
    seed: int,
) -> None:
    rng = random.Random(seed + 1)
    sampled = chains if len(chains) <= sample_size else rng.sample(chains, sample_size)

    def join_rules(start: int, end: int, chain: dict) -> str:
        return " | ".join(
            chain[f"Rule_{i}"] for i in range(start, end + 1) if f"Rule_{i}" in chain
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "variant_id",
                "anomaly_type",
                "host_chain_id",
                "host_scenario",
                "insert_pos_1based",
                "fragment_length",
                "shared_devices",
                "host_rules_before",
                "fragment_rules",
                "host_rules_after",
                "semantic_coherence_1to3",
                "natural_embedding_1to3",
                "notes",
            ]
        )
        for chain in sampled:
            start = int(chain["abnormal_fragment_start_index"])
            length = int(chain["abnormal_fragment_length"])
            chain_length = int(chain["chain_length"])
            fragment_rules = [
                chain[f"Rule_{i}"] for i in range(start, start + length)
            ]
            writer.writerow(
                [
                    chain["chain_id"],
                    chain.get("Type"),
                    chain.get("source_benign_chain_id"),
                    chain.get("scenario"),
                    start,
                    length,
                    "; ".join(chain.get("device_overlap", [])),
                    join_rules(1, start - 1, chain),
                    " | ".join(fragment_rules),
                    join_rules(start + length, chain_length, chain),
                    "",
                    "",
                    "",
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-chains", type=int, default=DEFAULT_TARGET_CHAINS)
    parser.add_argument("--max-insertions-per-fragment", type=int, default=1)
    parser.add_argument("--check-sample", type=int, default=DEFAULT_CHECK_SAMPLE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stats-output", type=Path, default=DEFAULT_STATS_OUTPUT)
    parser.add_argument("--check-output", type=Path, default=DEFAULT_CHECK_OUTPUT)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only regenerate the human check sheet from an existing variant dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.check_only:
        existing = json.loads(args.output.resolve().read_text(encoding="utf-8"))
        write_human_check_sheet(
            existing["chains"], args.check_output, args.check_sample, args.seed
        )
        print(
            f"Regenerated human check sheet ({args.check_sample} samples) "
            f"from {args.output} -> {args.check_output}"
        )
        return

    benign_input = DEFAULT_BENIGN.resolve()
    abnormal_inputs = [path.resolve() for path in DEFAULT_ABNORMAL_INPUTS]

    benign_chains = load_benign_chains(benign_input)
    fragments = load_abnormal_fragments(abnormal_inputs)
    print(f"Loaded benign chains: {len(benign_chains)}")
    print(f"Loaded abnormal fragments: {len(fragments)}")

    built_chains, stats = build_variant_chains(
        fragments=fragments,
        benign_chains=benign_chains,
        target_chains=args.target_chains,
        max_insertions_per_fragment=args.max_insertions_per_fragment,
        seed=args.seed,
    )
    print(f"Built device-shared variant chains: {stats['built_chains']} (target {args.target_chains})")
    print(f"Fragments with device-shared candidates: {stats['fragments_with_device_shared_candidates']}")
    print(f"Built by anomaly type: {stats['built_by_anomaly_type']}")

    raw_output = {
        "metadata": {
            "variant": "device_shared_contextual",
            "benign_input": str(benign_input),
            "abnormal_inputs": [str(path) for path in abnormal_inputs],
            "candidate_inclusion": {
                "device_match": "share_at_least_one_full_device_name",
                "environment_match": "not_constrained",
            },
            "insertion_policy": "insert abnormal fragment contiguously at a random position in the benign chain",
            "abnormal_fragment_start_index_base": 1,
            **stats,
        },
        "chains": built_chains,
    }

    source_index = build_source_index(abnormal_inputs)
    formatted_output = format_dataset(raw_output, source_index)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(formatted_output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_human_check_sheet(built_chains, args.check_output, args.check_sample, args.seed)

    print(f"Output: {args.output}")
    print(f"Stats: {args.stats_output}")
    print(f"Human check sheet ({args.check_sample} samples): {args.check_output}")


if __name__ == "__main__":
    main()
