#!/bin/bash
# NGQA-Interactive v2: no self-disclosure, multi-status roots for fitting; waits for the v1 postprocess to finish.
PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python
R=/ocean/projects/cis260058p/kshi4/rclup_runs
LOGS=$R/logs/pipeline_ngqa_v2; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$?"; }
until grep -q "POSTPROCESS DONE" $R/logs/pipeline_v1_post.log 2>/dev/null; do sleep 30; done
step collect_ngqa_v2 $PY -m rclup.collect --dataset ngqa --partition representation --roots 48 --seeds 13,29 --max-commits 3 --max-questions 4 --prefixes-per-root 2 --workers 8 --disclosure none --min-statuses 2 --out collect/ngqa_repr_v2 --cap-usd 40
step collect_ngqa_cal_v2 $PY -m rclup.collect --dataset ngqa --partition action_calibration --roots 16 --seeds 13,29 --max-commits 3 --max-questions 4 --prefixes-per-root 2 --workers 8 --disclosure none --min-statuses 2 --out collect/ngqa_cal_v2 --cap-usd 40
step train_ngqa2_A $PY -m rclup.train --collect collect/ngqa_repr_v2 --out models/ngqa2_A --epochs 12
step train_ngqa2_A_richtext $PY -m rclup.train --collect collect/ngqa_repr_v2 --out models/ngqa2_A_richtext --epochs 12 --no-latent
step train_ngqa2_A_generic $PY -m rclup.train --collect collect/ngqa_repr_v2 --out models/ngqa2_A_generic --epochs 12 --generic-latent
step train_ngqa2_A_scalar $PY -m rclup.train --collect collect/ngqa_repr_v2 --out models/ngqa2_A_scalar --epochs 12 --scalar
step retarget_ngqa2 $PY -m rclup.collect --dataset ngqa --retarget --carrier $R/models/ngqa2_A/carrier.pt --seeds 13,29 --workers 8 --out collect/ngqa_repr_v2 --cap-usd 40
step train_ngqa2_B $PY -m rclup.train --phase B --init $R/models/ngqa2_A/carrier.pt --collect collect/ngqa_repr_v2 --out models/ngqa2_B --epochs 8
step train_ngqa2_C $PY -m rclup.train --phase C --init $R/models/ngqa2_B/carrier.pt --collect collect/ngqa_cal_v2 --out models/ngqa2_C
E=eval/v2; mkdir -p $R/$E
step eval_ngqa2_rclup $PY -m rclup.run_episodes --dataset ngqa --split test --controller rclup --carrier $R/models/ngqa2_B/carrier.pt --calibrator $R/models/ngqa2_C/calibrator.pt --lam 0.02 --limit 150 --workers 8 --disclosure none --out $E/ngqa_rclup_none --cap-usd 40
step eval_ngqa2_richtext $PY -m rclup.run_episodes --dataset ngqa --split test --controller rich_text_mc --carrier $R/models/ngqa2_A_richtext/carrier.pt --lam 0.02 --limit 150 --workers 8 --disclosure none --out $E/ngqa_richtext_mc_none --cap-usd 40
step eval_ngqa2_scalar $PY -m rclup.run_episodes --dataset ngqa --split test --controller scalar_mc --carrier $R/models/ngqa2_A_scalar/carrier.pt --lam 0.02 --limit 150 --workers 8 --disclosure none --out $E/ngqa_scalar_mc_none --cap-usd 40
step report_ngqa2 $PY -m rclup.report --dataset ngqa --reference baselines/v3/ngqa_test_text_mas_none --runs baselines/v3/ngqa_test_text_mas_none,eval/v1/ngqa_disagreement_none,$E/ngqa_richtext_mc_none,$E/ngqa_scalar_mc_none,$E/ngqa_rclup_none --out $E/report_ngqa_none
echo "[$(date +%H:%M:%S)] NGQA V2 DONE"
