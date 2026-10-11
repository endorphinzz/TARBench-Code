"""生成 E9 全部 LLaMA-Factory 训练配置 (yaml) + dataset_info.json。

运行 (云端, 权重就位后):
    python experiments/finetune/configs/make_configs.py --model-path /root/models/Qwen3-8B

产出 (本目录):
    e9_m1_s{1,2,3}.yaml    主实验 3 seeds (sft_train_full)
    e9_m3_cls.yaml          仅分类目标消融 (sft_train_cls_only)
    e9_m2_{500,2000}.yaml   数据规模消融
    smoke.yaml              30 step 冒烟
    dataset_info.json       注册到 ../datasets/ (路径绝对化, 与 CWD 无关)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

CFG_DIR = Path(__file__).resolve().parent
DATA_DIR = CFG_DIR.parent / "datasets"
SAVES_DIR = CFG_DIR.parent / "saves"

DEV_NAME = "e9_dev"

# name, sft 文件, 数据集注册名, seed
RUNS = [
    ("e9_m1_s1", "sft_train_full.json", "e9_train_full", 20261008),
    ("e9_m1_s2", "sft_train_full.json", "e9_train_full", 20261009),
    ("e9_m1_s3", "sft_train_full.json", "e9_train_full", 20261010),
    ("e9_m3_cls", "sft_train_cls_only.json", "e9_train_cls_only", 20261008),
    ("e9_m2_500", "sft_train_scale_500.json", "e9_train_scale_500", 20261008),
    ("e9_m2_2000", "sft_train_scale_2000.json", "e9_train_scale_2000", 20261008),
]

TEMPLATE = """\
### model
model_name_or_path: {model_path}
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 32
lora_alpha: 64
lora_target: all
enable_thinking: false

### QLoRA (24GB 4090)
quantization_bit: 4
quantization_method: bitsandbytes

### data
dataset: {dataset}
dataset_dir: {data_dir}
template: qwen3
cutoff_len: 4096
overwrite_cache: true
preprocessing_num_workers: 8

### eval on dev (按 eval_loss 选最优 checkpoint)
do_eval: true
eval_dataset: {dev}
per_device_eval_batch_size: 4
eval_strategy: steps
eval_steps: {eval_steps}
save_steps: {eval_steps}
save_total_limit: 2
load_best_model_at_end: true
metric_for_best_model: eval_loss
greater_is_better: false

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 32
learning_rate: 1.0e-4
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
gradient_checkpointing: true
{max_steps_line}
### misc
output_dir: {save_dir}
logging_steps: 20
plot_loss: true
report_to: none
seed: {seed}
"""

EFFECTIVE_BS = 1 * 32  # per_device * grad_accum (bs=1: 4090 24GB 上 CE logits 峰值减半)


def eval_steps_for(n_samples: int) -> int:
    total = math.ceil(n_samples / EFFECTIVE_BS) * 2  # 2 epoch
    step = max(10, round(total / 5 / 10) * 10)
    return step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/root/models/Qwen3-8B")
    ap.add_argument("--train-size", type=int, default=7540,
                    help="train_full 样本数, 用于估算 eval_steps (以实际为准, 偏差只影响评测频次)")
    args = ap.parse_args()

    # dataset_info.json: 注册全部 SFT 文件 (绝对路径, 与运行 CWD 无关)
    sft_files = ["sft_dev.json"] + [r[1] for r in RUNS]
    info = {}
    for f in sft_files:
        name = "e9_" + f[len("sft_"):-len(".json")].replace("train_", "train_")
        info[name] = {
            "file_name": str(DATA_DIR / f),
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system"},
        }
    (CFG_DIR / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sizes = {
        "e9_train_full": args.train_size,
        "e9_train_cls_only": args.train_size,
        "e9_train_scale_500": 500,
        "e9_train_scale_2000": 2000,
    }

    for name, sft_file, ds_name, seed in RUNS:
        cfg = TEMPLATE.format(
            model_path=args.model_path,
            dataset=ds_name,
            data_dir=str(CFG_DIR),
            dev=DEV_NAME,
            eval_steps=eval_steps_for(sizes[ds_name]),
            max_steps_line="",
            save_dir=str(SAVES_DIR / name),
            seed=seed,
        )
        (CFG_DIR / f"{name}.yaml").write_text(cfg, encoding="utf-8")
        print(f"wrote {name}.yaml (eval_steps={eval_steps_for(sizes[ds_name])})")

    # 冒烟: 30 step, 小数据集
    smoke = TEMPLATE.format(
        model_path=args.model_path,
        dataset="e9_train_scale_500",
        data_dir=str(CFG_DIR),
        dev=DEV_NAME,
        eval_steps=15,
        max_steps_line="max_steps: 30",
        save_dir=str(SAVES_DIR / "smoke"),
        seed=20261008,
    )
    (CFG_DIR / "smoke.yaml").write_text(smoke, encoding="utf-8")
    print("wrote smoke.yaml")


if __name__ == "__main__":
    main()
