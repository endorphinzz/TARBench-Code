"""构建全量评测集 full_dataset.json（与论文主表 Table 2 一致）。

包含源数据的全部链：10,583 benign + 4,600 abnormal = 15,183 样本。
schema 与 build_datasets.py 产出的 icl/rag 数据集完全一致，
供 run_baselines.py 直接使用。
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "datasets"

BENIGN_SOURCE = ROOT_DIR / "data" / "benign" / "benign_long_chains_gpt54.json"
ABNORMAL_SOURCE = ROOT_DIR / "data" / "abnormal" / "abnormal_chains_gpt54.json"

SEED = 20260513


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_rules(chain):
    rule_keys = [key for key in chain if re.fullmatch(r"Rule_\d+", key)]
    rule_keys.sort(key=lambda key: int(key.split("_", 1)[1]))
    return {key: chain[key] for key in rule_keys}


def build_sample(chain, label, sample_index):
    rules = extract_rules(chain)
    sample = {
        "sample_index": sample_index,
        "sample_id": f"FULL_{label.upper()}_{sample_index:06d}",
        "chain_id": chain.get("chain_id"),
        "chain_type": chain.get("chain_type"),
        "label": label,
        "scenario": chain.get("scenario"),
        "scenario_group": chain.get("scenario_group"),
        "method": chain.get("method"),
        "chain_length": chain.get("chain_length", len(rules)),
        "rules": rules,
    }
    if label == "benign":
        sample["description"] = chain.get("description")
        sample["ground_truth"] = {
            "has_anomaly": False,
            "involved_rules": [],
            "anomaly_type": None,
        }
    else:
        sample["benign_description"] = chain.get("benign_description")
        sample["ground_truth"] = {
            "has_anomaly": True,
            "anomaly_type": chain.get("Type"),
            "type_id": chain.get("Type_ID"),
            "sub_category": chain.get("Sub_Category"),
            "description": chain.get("Description"),
            "abnormal_fragment_start_index": chain.get("abnormal_fragment_start_index"),
            "abnormal_fragment_length": chain.get("abnormal_fragment_length"),
            "abnormal_fragment_id_list": chain.get("abnormal_fragment_id_list", []),
        }
    return sample


def count_by(samples, key_path):
    counter = Counter()
    for sample in samples:
        current = sample
        for key in key_path:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
        counter[str(current if current is not None else "__missing__")] += 1
    return dict(sorted(counter.items(), key=lambda item: item[0]))


def main():
    benign_pool = load_json(BENIGN_SOURCE)["chains"]
    abnormal_pool = load_json(ABNORMAL_SOURCE)["chains"]

    benign_samples = [
        build_sample(chain, "benign", index)
        for index, chain in enumerate(benign_pool, start=1)
    ]
    abnormal_samples = [
        build_sample(chain, "abnormal", index)
        for index, chain in enumerate(abnormal_pool, start=1)
    ]

    combined = benign_samples + abnormal_samples
    random.Random(SEED).shuffle(combined)
    for index, sample in enumerate(combined, start=1):
        sample["mixed_index"] = index

    dataset = {
        "metadata": {
            "dataset": "full",
            "sampling_seed": SEED,
            "source_files": {
                "benign": str(BENIGN_SOURCE),
                "abnormal": str(ABNORMAL_SOURCE),
            },
            "sample_counts": {
                "benign": len(benign_samples),
                "abnormal": len(abnormal_samples),
                "total": len(combined),
            },
            "sampling_policy": "all chains from source files (no subsampling)",
            "distributions": {
                "label": count_by(combined, ["label"]),
                "abnormal_type": count_by(
                    [s for s in combined if s["label"] == "abnormal"],
                    ["ground_truth", "anomaly_type"],
                ),
            },
        },
        "samples": combined,
    }

    output_path = OUTPUT_DIR / "full_dataset.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote {output_path}")
    print(f"benign={len(benign_samples)}, abnormal={len(abnormal_samples)}, total={len(combined)}")


if __name__ == "__main__":
    main()
