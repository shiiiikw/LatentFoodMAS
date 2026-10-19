#!/bin/bash
# NGQA-Interactive v3: larger zero-disclosure collection (240 multi-status roots, 3 seeds) after the v2 chain, on worker $1.
JOB=$1
PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python
R=/ocean/projects/cis260058p/kshi4/rclup_runs
LOGS=$R/logs/pipeline_ngqa_v3; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=1 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$?"; }
C=collect/ngqa_repr_v2,collect/ngqa_repr_v3
step retarget $PY -m rclup.collect --dataset ngqa --retarget --carrier $R/models/ngqa3_A/carrier.pt --seeds 13 --max-questions 4 --workers 10 --out collect/ngqa_repr_v3 --cap-usd 90 --force
step train_B $PY -m rclup.train --phase B --init $R/models/ngqa3_A/carrier.pt --collect $C --out models/ngqa3_B --epochs 8
step train_C $PY -m rclup.train --phase C --init $R/models/ngqa3_B/carrier.pt --collect collect/ngqa_cal_v2,collect/ngqa_cal_v3 --out models/ngqa3_C
step train_C_richtext $PY -m rclup.train --phase C --init $R/models/ngqa3_A_richtext/carrier.pt --collect collect/ngqa_cal_v2,collect/ngqa_cal_v3 --out models/ngqa3_C_richtext --no-latent
E=eval/v3
step eval_rclup $PY -m rclup.run_episodes --dataset ngqa --split test --controller rclup --carrier $R/models/ngqa3_B/carrier.pt --calibrator $R/models/ngqa3_C/calibrator.pt --lam 0.02 --limit 300 --workers 10 --disclosure none --out $E/ngqa_rclup_none --cap-usd 90
step eval_text_mas $PY -m rclup.run_episodes --dataset ngqa --split test --controller text_mas --limit 300 --workers 10 --disclosure none --out $E/ngqa_text_mas_none --cap-usd 90
step eval_richtext $PY -m rclup.run_episodes --dataset ngqa --split test --controller rich_text_mc --carrier $R/models/ngqa3_A_richtext/carrier.pt --calibrator $R/models/ngqa3_C_richtext/calibrator.pt --lam 0.02 --limit 300 --workers 10 --disclosure none --out $E/ngqa_richtext_mc_none --cap-usd 90
step eval_disagree $PY -m rclup.run_episodes --dataset ngqa --split test --controller disagreement_ask --limit 300 --workers 10 --disclosure none --out $E/ngqa_disagreement_none --cap-usd 90
step report $PY -m rclup.report --dataset ngqa --reference $E/ngqa_text_mas_none --runs $E/ngqa_text_mas_none,$E/ngqa_disagreement_none,$E/ngqa_richtext_mc_none,$E/ngqa_rclup_none --out $E/report_ngqa_none
echo "[$(date +%H:%M:%S)] NGQA V3 DONE"
