#!/usr/bin/env bash
# E9 训练队列: 6 个 run 串行接力 (tmux 内执行, 断线不影响)
# 顺序: M1 3 seeds -> M3 cls-only -> M2 规模消融
# 任一 run 失败立即停止, 已完成的 adapter 保留在 saves/
set -euo pipefail

ROOT=/root/TARBench_Code
LF=/root/venv_sft/bin/llamafactory-cli
CFG=$ROOT/experiments/finetune/configs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() {
    local name=$1
    if [ -f "$ROOT/experiments/finetune/saves/$name/adapter_model.safetensors" ]; then
        echo "=== skip $name (已完成) ==="
        return 0
    fi
    echo "=== train $name ($(date '+%F %T')) ==="
    # 成功判据 = adapter 落盘 (llamafactory 收尾阶段可能以非零码退出但产物齐全, 属假失败)
    "$LF" train "$CFG/$name.yaml" 2>&1 | tee "$ROOT/experiments/finetune/saves/${name}.log" || true
    if [ -f "$ROOT/experiments/finetune/saves/$name/adapter_model.safetensors" ]; then
        echo "=== $name 完成 (adapter 已验证, $(date '+%F %T')) ==="
    else
        echo "!!! $name 失败 (无 adapter 产物), 队列停止"
        exit 1
    fi
}

run e9_m1_s1
run e9_m1_s2
run e9_m1_s3
run e9_m3_cls
run e9_m2_500
run e9_m2_2000

echo "=== 队列全部完成 ($(date '+%F %T')) ==="
echo "下一步推理: bash $ROOT/experiments/finetune/cloud/infer_all.sh"
