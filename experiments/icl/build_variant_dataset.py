"""构建设备共享变体的 ICL 复评数据集。

组成：300 条设备共享异常变体 + 284 条宿主良性链（变体的插入宿主，逐链配对）。
schema 与 build_full_dataset.py / icl_dataset.json 完全一致，
供 run_icl_benchmark.py --dataset-file 直接使用。

输出: experiments/icl/datasets/icl_eval_variants_300_284.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ICL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ICL_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "baselines"))

from build_full_dataset import build_sample, extract_rules  # noqa: E402

VARIANT_SOURCE = PROJECT_ROOT / "data" / "abnormal" / "abnormal_chains_gpt54_contextual_variants.json"
BENIGN_SOURCE = PROJECT_ROOT / "data" / "benign" / "benign_long_chains_gpt54.json"
OUTPUT_PATH = ICL_DIR / "datasets" / "icl_eval_variants_300_284.json"
SEED = 20260513


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    variant_chains = load_json(VARIANT_SOURCE)["chains"]
    benign_chains = load_json(BENIGN_SOURCE)["chains"]
    by_chain_id = {chain["chain_id"]: chain for chain in benign_chains}

    host_ids = []
    for chain in variant_chains:
        host_id = chain["source_benign_chain_id"]
        if host_id not in by_chain_id:
            raise KeyError(f"Host chain not found in benign source: {host_id}")
        if host_id not in host_ids:
            host_ids.append(host_id)
    host_chains = [by_chain_id[host_id] for host_id in host_ids]

    abnormal_samples = [build_sample(chain, "abnormal", i) for i, chain in enumerate(variant_chains, start=1)]
    benign_samples = [build_sample(chain, "benign", i) for i, chain in enumerate(host_chains, start=1)]

    combined = abnormal_samples + benign_samples
    random.Random(SEED).shuffle(combined)
    for index, sample in enumerate(combined, start=1):
        sample["sample_index"] = index
        sample["sample_id"] = f"VAR_{sample['label'].upper()}_{index:06d}"
        sample["mixed_index"] = index

    dataset = {
        "metadata": {
            "dataset": "contextual_variants",
            "variant": "device_shared_contextual",
            "sampling_seed": SEED,
            "source_files": {
                "variants": str(VARIANT_SOURCE),
                "benign_hosts": str(BENIGN_SOURCE),
            },
            "sample_counts": {
                "abnormal_variants": len(abnormal_samples),
                "benign_hosts": len(benign_samples),
                "total": len(combined),
            },
            "sampling_policy": "all device-shared variant chains + their unique host benign chains",
        },
        "samples": combined,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    print(
        f"abnormal_variants={len(abnormal_samples)}, benign_hosts={len(benign_samples)}, "
        f"total={len(combined)}"
    )


if __name__ == "__main__":
    main()
