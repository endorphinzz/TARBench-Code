#!/usr/bin/env bash
# 训练队列结束后自动接推理 (单个实例, 由 nohup 启动)
# 逻辑: 等 train_queue.sh 退出 -> 验证 6 个 adapter -> 启动 infer_all.sh
LOG=/root/auto_infer.log
SAVES=/root/TARBench_Code/experiments/finetune/saves

# 单实例锁: 防止 ssh 重试等造成的重复启动
exec 9>/root/.auto_infer.lock
if ! flock -n 9; then
    echo "another chain instance running, exit $(date '+%F %T')" >> "$LOG"
    exit 0
fi

echo "chain started $(date '+%F %T')" >> "$LOG"
while pgrep -f 'train_queu[e].sh' >/dev/null; do
    sleep 300
done
n=$(ls "$SAVES"/*/adapter_model.safetensors 2>/dev/null | wc -l)
echo "QUEUE_DONE adapters=$n $(date '+%F %T')" >> "$LOG"
if [ "$n" -eq 6 ]; then
    echo "launching infer_all $(date '+%F %T')" >> "$LOG"
    bash /root/TARBench_Code/experiments/finetune/cloud/infer_all.sh >> "$LOG" 2>&1
    echo "INFER_ALL_EXITED $(date '+%F %T')" >> "$LOG"
else
    echo "adapters=$n != 6, 不启动推理, 等待人工检查" >> "$LOG"
fi
