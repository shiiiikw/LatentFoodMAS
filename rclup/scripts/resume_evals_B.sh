#!/bin/bash
JOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/pipeline_hoosier_full; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=1 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$?"; }
E=eval/full
step eval_generic $PY -m rclup.run_episodes --dataset hoosier --split test --controller generic_latent_mc --carrier $R/models/hoosierF_A_generic/carrier.pt --calibrator $R/models/hoosierF_C_generic/calibrator.pt --lam 0.02 --limit 48 --workers 10 --out $E/hoosier_generic_mc --cap-usd 150
step eval_scalar $PY -m rclup.run_episodes --dataset hoosier --split test --controller scalar_mc --carrier $R/models/hoosierF_A_scalar/carrier.pt --calibrator $R/models/hoosierF_C_scalar/calibrator.pt --lam 0.02 --limit 48 --workers 10 --out $E/hoosier_scalar_mc --cap-usd 150
step eval_qref $PY -m rclup.run_episodes --dataset hoosier --split test --controller rclup_q_ref --carrier $R/models/hoosierF_A/carrier.pt --lam 0.02 --limit 48 --workers 10 --out $E/hoosier_rclup_qref --cap-usd 150
step eval_rclup_budget1 $PY -m rclup.run_episodes --dataset hoosier --split test --controller rclup --carrier $R/models/hoosierF_B/carrier.pt --calibrator $R/models/hoosierF_C/calibrator.pt --lam 0.02 --items-budget 1 --limit 48 --workers 10 --out $E/hoosier_rclup_budget1 --cap-usd 150
step eval_rclup_budget4 $PY -m rclup.run_episodes --dataset hoosier --split test --controller rclup --carrier $R/models/hoosierF_B/carrier.pt --calibrator $R/models/hoosierF_C/calibrator.pt --lam 0.02 --items-budget 4 --limit 48 --workers 10 --out $E/hoosier_rclup_budget4 --cap-usd 150
step eval_ngqa_disagree $PY -m rclup.run_episodes --dataset ngqa --split test --controller disagreement_ask --limit 300 --workers 10 --disclosure none --out eval/v3/ngqa_disagreement_none --cap-usd 150
step report_ngqa $PY -m rclup.report --dataset ngqa --reference eval/v3/ngqa_text_mas_none --runs eval/v3/ngqa_text_mas_none,eval/v3/ngqa_disagreement_none,eval/v3/ngqa_richtext_mc_none,eval/v3/ngqa_rclup_none --out eval/v3/report_ngqa_none
echo "[$(date +%H:%M:%S)] EVALS B DONE"
