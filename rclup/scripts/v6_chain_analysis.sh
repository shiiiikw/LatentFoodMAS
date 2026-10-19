#!/bin/bash
# Offline question-selection analysis on the v6 labels (reviewer situations S1-S5), after the v6 carriers exist.
JOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/v6
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
until grep -q "V6 ABLATION CHAIN DONE" $LOGS/chain_ablate.log 2>/dev/null; do sleep 60; done
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=2 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$?"; }
CARR="richtext=models/hoosier6_A_richtext/carrier.pt,richtext_nodep=models/hoosier6_A_richtext_no_dependency/carrier.pt,richtext_norecv=models/hoosier6_A_richtext_no_receiver_state/carrier.pt"
ABL="richtext_nodep:dependency,richtext_norecv:receiver_state"
step analysis_val $PY -m rclup.analysis_questions --collect collect/hoosier_v6_repr --dataset hoosier --carriers $CARR --ablate $ABL --val-only --out eval/analysis_questions_hoosier_v6_val
step analysis_all $PY -m rclup.analysis_questions --collect collect/hoosier_v6_repr --dataset hoosier --carriers $CARR --ablate $ABL --out eval/analysis_questions_hoosier_v6_all
echo "[$(date +%H:%M:%S)] V6 ANALYSIS DONE"
