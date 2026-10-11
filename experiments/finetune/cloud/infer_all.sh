#!/usr/bin/env bash
# E9 批量推理: (base 0-shot 对照 + 全部 FT adapter) x (T1/T2/T3)
# 在 venv_inf 环境跑; 预测文件直接用 evaluate_icl.py 评测
set -euo pipefail

ROOT=/root/TARBench_Code
PY=/root/shared-nvme/venv_inf/bin/python
MODEL=/root/shared-nvme/models/Qwen3-8B
FT=$ROOT/experiments/finetune
OUT=$FT/outputs

ADAPTERS=(e9_m1_s1 e9_m1_s2 e9_m1_s3 e9_m3_cls e9_m2_500 e9_m2_2000)
SPLITS=(test_t1 test_t2_icl750 test_t3_variants)

mkdir -p "$OUT"

# base 0-shot 对照 (与 FT 同 prompt / 同贪心解码 / 同 vLLM 栈)
for sp in "${SPLITS[@]}"; do
    f=$OUT/pred_${sp}_base0shot.json
    [ -f "$f" ] || "$PY" "$FT/run_inference.py" \
        --model "$MODEL" --split-file "$FT/datasets/splits/${sp}.json" \
        --output "$f" --tag qwen3-8b-base-0shot
done

# FT adapters
for ad in "${ADAPTERS[@]}"; do
    adapter_dir=$FT/saves/$ad
    [ -f "$adapter_dir/adapter_model.safetensors" ] || { echo "skip $ad (无 adapter)"; continue; }
    for sp in "${SPLITS[@]}"; do
        f=$OUT/pred_${sp}_${ad}.json
        [ -f "$f" ] || "$PY" "$FT/run_inference.py" \
            --model "$MODEL" --adapter "$adapter_dir" \
            --split-file "$FT/datasets/splits/${sp}.json" \
            --output "$f" --tag "qwen3-8b-${ad}"
    done
done

echo "=== 推理完成, 评测示例 ==="
echo "  python $ROOT/experiments/icl/evaluate_icl.py $OUT/pred_test_t1_e9_m1_s1.json"
