#!/bin/bash
# NGQA native knowledge-QA chain (no OpenAI): baseline -> label collection -> train -> controller evals.
JOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/ngqa_native; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=1 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$?"; }
E=eval/ngqa_full
step text_mas $PY -m rclup.run_episodes --dataset ngqa --split test --controller text_mas --disclosure full --limit 300 --workers 10 --out $E/text_mas --cap-usd 150
step collect $PY -m rclup.collect --dataset ngqa --partition representation --roots 300 --seeds 13,29,47 --max-commits 3 --max-questions 6 --prefixes-per-root 2 --workers 10 --disclosure full --out collect/ngqa_native_repr --cap-usd 150
step collect_cal $PY -m rclup.collect --dataset ngqa --partition action_calibration --roots 60 --seeds 13,29,47 --max-commits 3 --max-questions 6 --prefixes-per-root 2 --workers 10 --disclosure full --out collect/ngqa_native_cal --cap-usd 150
step train_A $PY -m rclup.train --collect collect/ngqa_native_repr --out models/ngqaN_A --epochs 15
step train_A_richtext $PY -m rclup.train --collect collect/ngqa_native_repr --out models/ngqaN_A_richtext --epochs 15 --no-latent
step train_C $PY -m rclup.train --phase C --init $R/models/ngqaN_A/carrier.pt --collect collect/ngqa_native_cal --out models/ngqaN_C
step train_C_richtext $PY -m rclup.train --phase C --init $R/models/ngqaN_A_richtext/carrier.pt --collect collect/ngqa_native_cal --out models/ngqaN_C_richtext --no-latent
step eval_rclup $PY -m rclup.run_episodes --dataset ngqa --split test --controller rclup_q_ref --disclosure full --carrier $R/models/ngqaN_A/carrier.pt --calibrator $R/models/ngqaN_C/calibrator.pt --lam 0.02 --limit 300 --workers 10 --out $E/rclup --cap-usd 150
step eval_richtext $PY -m rclup.run_episodes --dataset ngqa --split test --controller rich_text_mc --disclosure full --carrier $R/models/ngqaN_A_richtext/carrier.pt --calibrator $R/models/ngqaN_C_richtext/calibrator.pt --lam 0.02 --limit 300 --workers 10 --out $E/richtext_mc --cap-usd 150
step eval_disagree $PY -m rclup.run_episodes --dataset ngqa --split test --controller disagreement_ask --disclosure full --limit 300 --workers 10 --out $E/disagreement --cap-usd 150
step report $PY -m rclup.report --dataset ngqa --reference $E/text_mas --runs $E/text_mas,$E/disagreement,$E/richtext_mc,$E/rclup --out $E/report
echo "[$(date +%H:%M:%S)] NGQA NATIVE DONE"
