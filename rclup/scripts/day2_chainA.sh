#!/bin/bash
# Day 2, worker A: seeds 1/2 for the main method (voi_det) and fixed order at caps 6/7/8 (atomic), then max-uncertainty sweep
# (ablation: receiver-task information replaced by local confidence).
JOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/day2; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
until grep -q '"ready"' $R/worker_$JOB.json 2>/dev/null; do sleep 20; done
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=2 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$? $(grep -E '"end_to_end_match"' $LOGS/$name.log | tail -1)"; }
B5=normal,rambling,impatience,self_contradictory,unsupported_request; E=eval/v7
V="--dataset hoosier --split test --variant v6 --intake-policy atomic --behaviors $B5 --limit 240 --workers 10 --cap-usd 320"
for S in 1 2; do for T in 8 6 7; do for C in voi_det fixed_order; do
  step ${C}_t${T}_s$S $PY -m rclup.run_episodes $V --controller $C --max-turns $T --seed $S --out $E/seed$S/turns$T/hoosier_${C}_b5
done; done; done
for T in 8 6 7 5 10; do step max_uncertainty_t$T $PY -m rclup.run_episodes $V --controller max_uncertainty --max-turns $T --out $E/turns$T/hoosier_max_uncertainty_b5; done
echo "[$(date +%H:%M:%S)] DAY2 A DONE"
