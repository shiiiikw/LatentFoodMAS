#!/bin/bash
# Full HoosierHelp baseline runs (48 specs x 5 behaviors = 240 episodes each), split across two vLLM workers.
#   usage: hoosier_baselines_full.sh <worker_job> baseline1,baseline2,...
cd /jet/home/kshi4/LatentFoodMAS
JOB=$1; NAMES=$2
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/baselines; mkdir -p $LOGS
for i in $(seq 1 120); do grep -q '"ready"' $R/worker_$JOB.json 2>/dev/null && break; sleep 10; done
grep -q '"ready"' $R/worker_$JOB.json || { echo "worker $JOB not ready"; exit 1; }
for NAME in ${NAMES//,/ }; do
  echo "=== $NAME start $(date)"
  srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=1 --mem=0 --export=ALL \
    $PY -m rclup.hoosier_baselines --baselines $NAME --workers 10 --out eval/hoosier_baselines --cap-usd 250 > $LOGS/full_$NAME.log 2>&1
  echo "=== $NAME rc=$? $(date)"
  grep -E '"end_to_end_match"|"errors"' $LOGS/full_$NAME.log | tail -2
done
echo "ALL DONE"
