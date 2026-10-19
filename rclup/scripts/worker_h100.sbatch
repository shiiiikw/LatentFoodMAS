#!/bin/bash
#SBATCH -A cis260058p
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --constraint=h100-80
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --job-name=rclup_worker
#SBATCH --no-requeue
#SBATCH --output=/ocean/projects/cis260058p/kshi4/rclup_runs/logs/worker_%j.log
# Persistent inference worker: starts a vLLM OpenAI-compatible server for the frozen
# Qwen3-8B backbone, publishes host/port, then idles so drivers can attach with
# `srun --jobid=$SLURM_JOB_ID --overlap`.
set -uo pipefail
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=5
export VLLM_LOGGING_LEVEL=INFO
PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python
MODEL=/ocean/projects/cis260058p/kshi4/hf_cache_data/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
RUNS=/ocean/projects/cis260058p/kshi4/rclup_runs
PORT=$((20000 + SLURM_JOB_ID % 40000))
STATUS=$RUNS/worker_${SLURM_JOB_ID}.json
# Stage weights to node-local scratch: vLLM's mmap loading from Lustre runs at ~10 MB/s.
if [ -n "${LOCAL:-}" ] && [ -d "$LOCAL" ]; then
  STAGED=$LOCAL/qwen3-8b; mkdir -p "$STAGED"
  if cp -rL "$MODEL"/. "$STAGED"/ 2>/dev/null; then echo "staged weights to $STAGED"; MODEL=$STAGED; else echo "staging failed; loading from /ocean"; fi
fi
echo "node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES port=$PORT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
TOTAL_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$TOTAL_MB" -gt 60000 ]; then UTIL=0.55; else UTIL=0.45; fi
echo "{\"job_id\": \"$SLURM_JOB_ID\", \"host\": \"$(hostname)\", \"port\": $PORT, \"status\": \"starting\", \"gpu_mb\": $TOTAL_MB, \"util\": $UTIL}" > $STATUS
$PY -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name qwen3-8b --port "$PORT" --host 0.0.0.0 \
  --dtype bfloat16 --gpu-memory-utilization "$UTIL" --max-model-len 16384 \
  --enable-prefix-caching --max-num-seqs 96 --seed 0 \
  --disable-log-requests > $RUNS/logs/vllm_${SLURM_JOB_ID}.log 2>&1 &
VPID=$!
for i in $(seq 1 180); do
  if curl -s "http://127.0.0.1:$PORT/v1/models" | grep -q qwen3-8b; then
    echo "{\"job_id\": \"$SLURM_JOB_ID\", \"host\": \"$(hostname)\", \"port\": $PORT, \"status\": \"ready\", \"gpu_mb\": $TOTAL_MB, \"util\": $UTIL}" > $STATUS
    echo "vLLM ready after ${i}x5s"; break
  fi
  if ! kill -0 $VPID 2>/dev/null; then echo "vLLM died"; tail -30 $RUNS/logs/vllm_${SLURM_JOB_ID}.log; echo "{\"job_id\": \"$SLURM_JOB_ID\", \"status\": \"failed\"}" > $STATUS; exit 1; fi
  sleep 5
done
trap "kill $VPID 2>/dev/null; echo '{\"job_id\": \"$SLURM_JOB_ID\", \"status\": \"stopped\"}' > $STATUS" EXIT
wait $VPID
