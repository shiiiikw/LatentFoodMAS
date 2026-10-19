#!/bin/bash
# Validation-scale pipeline (runs inside the vLLM worker allocation via srun --overlap).
# Steps continue on failure; each step logs to $LOGS/<step>.log.
PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python
R=/ocean/projects/cis260058p/kshi4/rclup_runs
LOGS=$R/logs/pipeline_v1; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$?"; }
true
# 1. Hoosier re-collection with the full question menu
step collect_hoosier_v2 $PY -m rclup.collect --dataset hoosier --partition representation --roots 24 --seeds 13,29 --max-commits 4 --max-questions 4 --prefixes-per-root 2 --workers 8 --out collect/hoosier_repr_v2 --cap-usd 40
# 3. Phase A training: RCLUP + matched controls
for ds in hoosier:hoosier_repr_v2 ngqa:ngqa_repr_v1; do
  d=${ds%%:*}; c=${ds##*:}
  step train_${d}_A $PY -m rclup.train --collect collect/$c --out models/${d}_A --epochs 12
  step train_${d}_A_richtext $PY -m rclup.train --collect collect/$c --out models/${d}_A_richtext --epochs 12 --no-latent
  step train_${d}_A_generic $PY -m rclup.train --collect collect/$c --out models/${d}_A_generic --epochs 12 --generic-latent
  step train_${d}_A_scalar $PY -m rclup.train --collect collect/$c --out models/${d}_A_scalar --epochs 12 --scalar
done
# 4. Phase B retargeting (executor-matched question outcomes) for the RCLUP carrier, then fit g_k
step retarget_hoosier $PY -m rclup.collect --dataset hoosier --retarget --carrier $R/models/hoosier_A/carrier.pt --seeds 13,29 --workers 8 --out collect/hoosier_repr_v2 --cap-usd 40
step retarget_ngqa $PY -m rclup.collect --dataset ngqa --retarget --carrier $R/models/ngqa_A/carrier.pt --seeds 13,29 --workers 8 --out collect/ngqa_repr_v1 --cap-usd 40
step train_hoosier_B $PY -m rclup.train --phase B --init $R/models/hoosier_A/carrier.pt --collect collect/hoosier_repr_v2 --out models/hoosier_B --epochs 8
step train_ngqa_B $PY -m rclup.train --phase B --init $R/models/ngqa_A/carrier.pt --collect collect/ngqa_repr_v1 --out models/ngqa_B --epochs 8
# 5. Phase C calibrators on action-calibration prefixes
step train_hoosier_C $PY -m rclup.train --phase C --init $R/models/hoosier_B/carrier.pt --collect collect/hoosier_cal_v1 --out models/hoosier_C
step train_ngqa_C $PY -m rclup.train --phase C --init $R/models/ngqa_B/carrier.pt --collect collect/ngqa_cal_v1 --out models/ngqa_C
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
