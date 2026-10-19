#!/bin/bash
# After validation_pipeline.sh: Phase D reporting-only calibration data + summary tables.
PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python
R=/ocean/projects/cis260058p/kshi4/rclup_runs
LOGS=$R/logs/pipeline_v1; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$?"; }
until grep -q "PIPELINE DONE" $R/logs/pipeline_v1.log 2>/dev/null; do sleep 30; done
E=eval/v1
# Phase D: run the frozen deployed policy on policy-calibration roots (reporting-only calibration data)
step eval_hoosier_rclup_polcal $PY -m rclup.run_episodes --dataset hoosier --split policy_calibration --controller rclup --carrier $R/models/hoosier_B/carrier.pt --calibrator $R/models/hoosier_C/calibrator.pt --lam 0.02 --limit 17 --workers 8 --out $E/hoosier_rclup_polcal --cap-usd 40
step eval_ngqa_rclup_polcal $PY -m rclup.run_episodes --dataset ngqa --split policy_calibration --controller rclup --carrier $R/models/ngqa_B/carrier.pt --calibrator $R/models/ngqa_C/calibrator.pt --lam 0.02 --limit 30 --workers 8 --disclosure none --out $E/ngqa_rclup_polcal --cap-usd 40
# Tables
step report_hoosier $PY -m rclup.report --dataset hoosier --reference baselines/v2/hoosier_test_text_mas --runs baselines/v2/hoosier_test_text_mas,$E/hoosier_disagreement,$E/hoosier_richtext_mc,$E/hoosier_rclup --out $E/report_hoosier
for disc in half none; do
  step report_ngqa_$disc $PY -m rclup.report --dataset ngqa --reference baselines/v3/ngqa_test_text_mas_$disc --runs baselines/v3/ngqa_test_text_mas_$disc,$E/ngqa_disagreement_$disc,$E/ngqa_richtext_mc_$disc,$E/ngqa_rclup_$disc --out $E/report_ngqa_$disc
done
echo "[$(date +%H:%M:%S)] POSTPROCESS DONE"
