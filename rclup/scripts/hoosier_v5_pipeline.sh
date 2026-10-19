#!/bin/bash
# v5-workflow labels (single seed, develop stage) -> rich-text controller -> calibrator -> closed-loop eval on v5.
JOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/hoosier_v5; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=2 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$?"; }
C=collect/hoosier_v5_repr; CAP=250
step collect $PY -m rclup.collect --dataset hoosier --partition representation --roots 200 --seeds 13 --max-commits 3 --max-questions 6 --prefixes-per-root 2 --workers 10 --variant v5 --no-latent --out $C --cap-usd $CAP
step collect_cal $PY -m rclup.collect --dataset hoosier --partition action_calibration --roots 30 --seeds 13 --max-commits 3 --max-questions 6 --prefixes-per-root 2 --workers 10 --variant v5 --no-latent --out collect/hoosier_v5_cal --cap-usd $CAP
step train_A $PY -m rclup.train --collect $C --out models/hoosier5_A_richtext --epochs 15 --no-latent
step train_C $PY -m rclup.train --phase C --init $R/models/hoosier5_A_richtext/carrier.pt --collect collect/hoosier_v5_cal --out models/hoosier5_C_richtext --no-latent
B5=normal,rambling,impatience,self_contradictory,unsupported_request; E=eval/full5
step eval_richtext $PY -m rclup.run_episodes --dataset hoosier --split test --controller rich_text_mc --variant v5 --carrier $R/models/hoosier5_A_richtext/carrier.pt --calibrator $R/models/hoosier5_C_richtext/calibrator.pt --lam 0.02 --behaviors $B5 --limit 240 --workers 10 --out $E/hoosier_richtext_v5labels_b5 --cap-usd $CAP
step report $PY -m rclup.report --dataset hoosier --reference $E/hoosier_text_mas_b5 --runs $E/hoosier_text_mas_b5,$E/hoosier_disagreement_ask_b5,$E/hoosier_max_uncertainty_b5,$E/hoosier_fixed_order_b5,$E/hoosier_voi_det_b5,$E/hoosier_richtext_oldlabels_b5,$E/hoosier_richtext_v5labels_b5 --out $E/report_hoosier_b5_full
echo "[$(date +%H:%M:%S)] HOOSIER V5 PIPELINE DONE"
