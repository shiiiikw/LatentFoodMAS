#!/bin/bash
# Atomic questions: myopic VOI baseline at caps 5/6/7/8/10 and the other baselines at caps 5 and 7 (after v7_baselines).
JOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/v7; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
until grep -q "V7 BASELINES DONE" $R/logs/v7_baselines.log 2>/dev/null; do sleep 60; done
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=2 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$? $(grep -E '"end_to_end_match"' $LOGS/$name.log | tail -1)"; }
B5=normal,rambling,impatience,self_contradictory,unsupported_request; E=eval/v7
V="--dataset hoosier --split test --variant v6 --intake-policy atomic --behaviors $B5 --limit 240 --workers 10 --cap-usd 300"
for T in 8 6 10 5 7; do step voi_myopic_t$T $PY -m rclup.run_episodes $V --controller voi_myopic --max-turns $T --out $E/turns$T/hoosier_voi_myopic_b5; done
for T in 5 7; do for C in fixed_order disagreement_ask voi_det; do
  step ${C}_t$T $PY -m rclup.run_episodes $V --controller $C --max-turns $T --out $E/turns$T/hoosier_${C}_b5
done; done
echo "[$(date +%H:%M:%S)] V7 BASELINES2 DONE"
