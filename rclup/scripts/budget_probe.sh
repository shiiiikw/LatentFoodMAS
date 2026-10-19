#!/bin/bash
# Turn-budget headroom probes on v5 (text controller): intake grouping, skipped fields, turn-cap sweep.
JOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/budget; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=2 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$? $(grep -E '"end_to_end_match"' $LOGS/$name.log | tail -1)"; }
B5=normal,rambling,impatience,self_contradictory,unsupported_request
COMMON="--dataset hoosier --split test --controller text_mas --variant v5 --behaviors $B5 --limit 240 --workers 10 --cap-usd 250"
step two      $PY -m rclup.run_episodes $COMMON --intake-policy two --out eval/budget/text_two
step skipdocs $PY -m rclup.run_episodes $COMMON --intake-policy skip_docs --out eval/budget/text_skipdocs
step turns10  $PY -m rclup.run_episodes $COMMON --max-turns 10 --out eval/budget/text_turns10
step turns6   $PY -m rclup.run_episodes $COMMON --max-turns 6 --out eval/budget/text_turns6
echo "[$(date +%H:%M:%S)] PROBE DONE"
