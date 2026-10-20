#!/bin/bash
# Rerun the budget-aware VOI at cap 8 with the corrected fallback reserve (after v7_baselines2).
JOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/v7
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
until grep -q "V7 BASELINES2 DONE" $R/logs/v7_baselines2.log 2>/dev/null; do sleep 60; done
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=2 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$? $(grep -E '"end_to_end_match"' $LOGS/$name.log | tail -1)"; }
B5=normal,rambling,impatience,self_contradictory,unsupported_request
step voi_det_t8 $PY -m rclup.run_episodes --dataset hoosier --split test --variant v6 --intake-policy atomic --behaviors $B5 --limit 240 --workers 10 --cap-usd 300 --controller voi_det --max-turns 8 --out eval/v7/turns8/hoosier_voi_det_b5
echo "[$(date +%H:%M:%S)] V7 BASELINES3 DONE"
