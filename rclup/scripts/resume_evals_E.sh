#!/bin/bash
# v4: all modules + direct-to-person disagreement/unknown questions, confirmations off. Waits for chain D.
JOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/pipeline_hoosier_full; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=1 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$?"; }
until grep -q "EVALS D DONE" $R/logs/resume_evals_D.log 2>/dev/null; do sleep 30; done
E=eval/full4; B5=normal,rambling,impatience,self_contradictory,unsupported_request
step v4_rclup_b5 $PY -m rclup.run_episodes --dataset hoosier --split test --controller rclup --carrier $R/models/hoosierF_B/carrier.pt --calibrator $R/models/hoosierF_C/calibrator.pt --lam 0.02 --behaviors $B5 --limit 240 --workers 10 --out $E/hoosier_rclup_b5 --cap-usd 150
step v4_disagree_b5 $PY -m rclup.run_episodes --dataset hoosier --split test --controller disagreement_ask --behaviors $B5 --limit 240 --workers 10 --out $E/hoosier_disagreement_b5 --cap-usd 150
step v4_report $PY -m rclup.report --dataset hoosier --reference eval/full3/hoosier_text_mas_b5 --runs eval/full3/hoosier_text_mas_b5,$E/hoosier_disagreement_b5,$E/hoosier_rclup_b5 --out $E/report_hoosier_b5
echo "[$(date +%H:%M:%S)] EVALS E DONE"
