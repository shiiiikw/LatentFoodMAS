#!/bin/bash
# Validation-scale pipeline (runs inside the vLLM worker allocation via srun --overlap).
# Steps continue on failure; each step logs to $LOGS/<step>.log.
PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python
R=/ocean/projects/cis260058p/kshi4/rclup_runs
LOGS=$R/logs/pipeline_v1; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$?"; }
# 6. Closed-loop evaluation on the held-out test roots (budget 2 atomic items)
E=eval/v1; mkdir -p $R/$E
step eval_hoosier_rclup $PY -m rclup.run_episodes --dataset hoosier --split test --controller rclup --carrier $R/models/hoosier_B/carrier.pt --calibrator $R/models/hoosier_C/calibrator.pt --lam 0.02 --limit 48 --workers 8 --out $E/hoosier_rclup --cap-usd 40
step eval_hoosier_richtext $PY -m rclup.run_episodes --dataset hoosier --split test --controller rich_text_mc --carrier $R/models/hoosier_A_richtext/carrier.pt --lam 0.02 --limit 48 --workers 8 --out $E/hoosier_richtext_mc --cap-usd 40
step eval_hoosier_disagree $PY -m rclup.run_episodes --dataset hoosier --split test --controller disagreement_ask --limit 48 --workers 8 --out $E/hoosier_disagreement --cap-usd 40
for disc in half none; do
  step eval_ngqa_rclup_$disc $PY -m rclup.run_episodes --dataset ngqa --split test --controller rclup --carrier $R/models/ngqa_B/carrier.pt --calibrator $R/models/ngqa_C/calibrator.pt --lam 0.02 --limit 150 --workers 8 --disclosure $disc --out $E/ngqa_rclup_$disc --cap-usd 40
  step eval_ngqa_richtext_$disc $PY -m rclup.run_episodes --dataset ngqa --split test --controller rich_text_mc --carrier $R/models/ngqa_A_richtext/carrier.pt --lam 0.02 --limit 150 --workers 8 --disclosure $disc --out $E/ngqa_richtext_mc_$disc --cap-usd 40
  step eval_ngqa_disagree_$disc $PY -m rclup.run_episodes --dataset ngqa --split test --controller disagreement_ask --limit 150 --workers 8 --disclosure $disc --out $E/ngqa_disagreement_$disc --cap-usd 40
done
echo "[$(date +%H:%M:%S)] PIPELINE DONE"
