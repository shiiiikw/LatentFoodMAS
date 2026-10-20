#!/bin/bash
# Atomic questions (one field per question): the three baselines at caps 8/6/10, plus never-ask at 8.
JOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/v7; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=2 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$? $(grep -E '"end_to_end_match"' $LOGS/$name.log | tail -1)"; }
B5=normal,rambling,impatience,self_contradictory,unsupported_request; E=eval/v7
V="--dataset hoosier --split test --variant v6 --intake-policy atomic --behaviors $B5 --limit 240 --workers 10 --cap-usd 290"
for T in 8 6 10; do for C in fixed_order voi_det disagreement_ask; do
  step ${C}_t$T $PY -m rclup.run_episodes $V --controller $C --max-turns $T --out $E/turns$T/hoosier_${C}_b5
done; done
step text_mas_t8 $PY -m rclup.run_episodes $V --controller text_mas --max-turns 8 --out $E/turns8/hoosier_text_mas_b5
echo "[$(date +%H:%M:%S)] V7 BASELINES DONE"
