#!/bin/bash
# After worker A's chain: ask-everything (disagreement-first) at cap 10 for the free-budget table.
JOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/workerA
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
until grep -q "WORKER A DONE" $R/logs/workerA_chain.log 2>/dev/null; do sleep 60; done
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=2 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$? $(grep -E '"end_to_end_match"' $LOGS/$name.log | tail -1)"; }
B5=normal,rambling,impatience,self_contradictory,unsupported_request
step v6_disagreement_ask_t10 $PY -m rclup.run_episodes --dataset hoosier --split test --variant v6 --behaviors $B5 --limit 240 --workers 10 --cap-usd 270 --controller disagreement_ask --max-turns 10 --out eval/v6/turns10/hoosier_disagreement_ask_b5
echo "[$(date +%H:%M:%S)] WORKER A TAIL DONE"
