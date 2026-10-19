#!/bin/bash
JOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/final; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=1 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$?"; }
B5=normal,rambling,impatience,self_contradictory,unsupported_request; CAP=260; E=eval/final
for c in disagreement_ask max_uncertainty fixed_order; do
  step $c $PY -m rclup.run_episodes --dataset hoosier --split test --controller $c --variant v5 --behaviors $B5 --limit 240 --workers 10 --out $E/hoosier_${c}_b5 --cap-usd $CAP
done
echo "[$(date +%H:%M:%S)] CHAIN B DONE"
