#!/bin/bash
# Final-solution evaluation: v5 workflow, all shared-state controllers, 240 conversations (cached simulated user; resumes).
JOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/eval_v5; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=1 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$?"; }
B5=normal,rambling,impatience,self_contradictory,unsupported_request; CAP=250; E=eval/full5; V=v5
for c in text_mas disagreement_ask max_uncertainty fixed_order voi_det; do
  step v5_$c $PY -m rclup.run_episodes --dataset hoosier --split test --controller $c --variant $V --behaviors $B5 --limit 240 --workers 10 --out $E/hoosier_${c}_b5 --cap-usd $CAP
done
step v5_richtext_oldlabels $PY -m rclup.run_episodes --dataset hoosier --split test --controller rich_text_mc --variant $V --carrier $R/models/hoosierF_A_richtext/carrier.pt --calibrator $R/models/hoosierF_C_richtext/calibrator.pt --lam 0.02 --behaviors $B5 --limit 240 --workers 10 --out $E/hoosier_richtext_oldlabels_b5 --cap-usd $CAP
step v5_report $PY -m rclup.report --dataset hoosier --reference $E/hoosier_text_mas_b5 --runs $E/hoosier_text_mas_b5,$E/hoosier_disagreement_ask_b5,$E/hoosier_max_uncertainty_b5,$E/hoosier_fixed_order_b5,$E/hoosier_voi_det_b5,$E/hoosier_richtext_oldlabels_b5 --out $E/report_hoosier_b5
echo "[$(date +%H:%M:%S)] EVAL V5 DONE"
