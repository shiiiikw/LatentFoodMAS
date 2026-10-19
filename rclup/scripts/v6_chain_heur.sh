#!/bin/bash
# v6 heuristic controllers (full 240) + turn-cap sweep for fixed order / VOI / never-ask.
JOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/v6; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=2 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$? $(grep -E '"end_to_end_match"' $LOGS/$name.log | tail -1)"; }
B5=normal,rambling,impatience,self_contradictory,unsupported_request; E=eval/v6
COMMON="--dataset hoosier --split test --variant v6 --behaviors $B5 --limit 240 --workers 10 --cap-usd 260"
for C in fixed_order voi_det text_mas disagreement_ask max_uncertainty; do
  step $C $PY -m rclup.run_episodes $COMMON --controller $C --out $E/hoosier_${C}_b5
done
for T in 6 7 10; do for C in fixed_order voi_det text_mas; do
  step ${C}_t$T $PY -m rclup.run_episodes $COMMON --controller $C --max-turns $T --out $E/turns$T/hoosier_${C}_b5
done; done
echo "[$(date +%H:%M:%S)] V6 HEURISTIC CHAIN DONE"
