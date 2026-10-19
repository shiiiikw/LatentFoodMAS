#!/bin/bash
# Full-scale Phase A collection for HoosierHelp on a dedicated worker job (waits for its vLLM to be ready).
JOB=$1
PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python
R=/ocean/projects/cis260058p/kshi4/rclup_runs
until [ -f $R/worker_$JOB.json ] && grep -q '"ready"' $R/worker_$JOB.json; do sleep 30; done
export RCLUP_WORKER_JOB=$JOB PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=1 --mem=0 --export=ALL bash -c "cd /jet/home/kshi4/LatentFoodMAS && $PY -m rclup.collect --dataset hoosier --partition representation --roots 193 --seeds 13,29,47 --max-commits 5 --max-questions 6 --prefixes-per-root 2 --workers 10 --out collect/hoosier_repr_full --cap-usd 60 && $PY -m rclup.collect --dataset hoosier --partition action_calibration --roots 25 --seeds 13,29,47 --max-commits 5 --max-questions 6 --prefixes-per-root 2 --workers 10 --out collect/hoosier_cal_full --cap-usd 60"
echo "[$(date +%H:%M:%S)] FULL HOOSIER COLLECT DONE"
