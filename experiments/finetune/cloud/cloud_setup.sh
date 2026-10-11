#!/usr/bin/env bash
# E9 云端环境一键搭建 (并行智算云 4090 实例, WebSSH 进入后执行)
# 用法: bash experiments/finetune/cloud/cloud_setup.sh
# 前置: 已上传并解压代码到 /root/TARBench_Code (见 README_E9)
set -euo pipefail

export HF_ENDPOINT=https://hf-mirror.com
export PIP_NO_CACHE_DIR=1
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export HF_HOME=/root/shared-nvme/hf_home
ROOT=/root/TARBench_Code
DATA=/root/shared-nvme          # 50GB 数据盘 (JuiceFS)
MODEL_DIR=$DATA/models/Qwen3-8B
VENV_INF=$DATA/venv_inf

echo "=== [1/5] GPU/磁盘检查 ==="
nvidia-smi || { echo "nvidia-smi 失败"; exit 1; }
df -h /root "$DATA" | tail -2
AVAIL_GB=$(df --output=avail -BG "$DATA" | tail -1 | tr -dc '0-9')
if [ "$AVAIL_GB" -lt 40 ]; then echo "数据盘不足 40GB: ${AVAIL_GB}GB"; exit 1; fi
# 训练 checkpoint 输出目录指向数据盘 (中间 checkpoint 可达数 GB)
mkdir -p "$DATA/saves"
ln -sfn "$DATA/saves" "$ROOT/experiments/finetune/saves"

echo "=== [2/5] 训练环境 venv_sft (复用镜像自带 PyTorch) ==="
if [ ! -x /root/venv_sft/bin/pip ]; then
    python3 -m venv --system-site-packages /root/venv_sft 2>/dev/null || {
        pip3 install -q virtualenv
        python3 -m virtualenv --system-site-packages -p python3 /root/venv_sft
    }
fi
/root/venv_sft/bin/pip install -U pip
/root/venv_sft/bin/pip install "llamafactory>=0.9.2" bitsandbytes || {
    echo "pip 安装 LLaMA-Factory 失败, 改用源码安装"
    [ -d /root/LLaMA-Factory ] || git clone https://github.com/hiyouga/LLaMA-Factory.git /root/LLaMA-Factory
    /root/venv_sft/bin/pip install -e /root/LLaMA-Factory bitsandbytes
}

echo "=== [3/5] 推理环境 venv_inf (vLLM, 装在数据盘) ==="
if [ ! -d "$VENV_INF" ]; then
    python3 -m venv "$VENV_INF"
fi
"$VENV_INF/bin/pip" install -U pip
"$VENV_INF/bin/pip" install vllm

echo "=== [4/5] 下载 Qwen3-8B 权重 (~16GB, hf-mirror) ==="
if [ ! -f "$MODEL_DIR/config.json" ]; then
    "$VENV_INF/bin/huggingface-cli" download Qwen/Qwen3-8B \
        --local-dir "$MODEL_DIR" --max-workers 4
fi

echo "=== [5/5] 生成训练配置 ==="
python3 "$ROOT/experiments/finetune/configs/make_configs.py" --model-path "$MODEL_DIR"

echo ""
echo "环境就绪。下一步:"
echo "  1) 冒烟:   /root/venv_sft/bin/llamafactory-cli train $ROOT/experiments/finetune/configs/smoke.yaml"
echo "  2) 队列:   tmux new -s train 'bash $ROOT/experiments/finetune/cloud/train_queue.sh'"
