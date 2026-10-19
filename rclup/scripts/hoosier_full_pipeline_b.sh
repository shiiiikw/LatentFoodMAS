#!/bin/bash
# Full-scale HoosierHelp chain on worker $1: train A (4 variants) -> retarget -> B -> C -> closed-loop evals (budget sweep, 5 behaviors for main rows).
JOB=$1
PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python
R=/ocean/projects/cis260058p/kshi4/rclup_runs
LOGS=$R/logs/pipeline_hoosier_full; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=1 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$?"; }
C=collect/hoosier_repr_full
step retarget $PY -m rclup.collect --dataset hoosier --retarget --carrier $R/models/hoosierF_A/carrier.pt --seeds 13 --max-questions 4 --workers 10 --out $C --cap-usd 90 --force
step train_B $PY -m rclup.train --phase B --init $R/models/hoosierF_A/carrier.pt --collect $C --out models/hoosierF_B --epochs 8
step train_C $PY -m rclup.train --phase C --init $R/models/hoosierF_B/carrier.pt --collect collect/hoosier_cal_full --out models/hoosierF_C
step train_C_richtext $PY -m rclup.train --phase C --init $R/models/hoosierF_A_richtext/carrier.pt --collect collect/hoosier_cal_full --out models/hoosierF_C_richtext --no-latent
step train_C_generic $PY -m rclup.train --phase C --init $R/models/hoosierF_A_generic/carrier.pt --collect collect/hoosier_cal_full --out models/hoosierF_C_generic --generic-latent
step train_C_scalar $PY -m rclup.train --phase C --init $R/models/hoosierF_A_scalar/carrier.pt --collect collect/hoosier_cal_full --out models/hoosierF_C_scalar --scalar
E=eval/full
B5=normal,rambling,impatience,self_contradictory,unsupported_request
# main rows: all 5 behaviors (240 conversations), budget 2
step eval_text_mas_b5 $PY -m rclup.run_episodes --dataset hoosier --split test --controller text_mas --behaviors $B5 --limit 240 --workers 10 --out $E/hoosier_text_mas_b5 --cap-usd 90
step eval_rclup_b5 $PY -m rclup.run_episodes --dataset hoosier --split test --controller rclup --carrier $R/models/hoosierF_B/carrier.pt --calibrator $R/models/hoosierF_C/calibrator.pt --lam 0.02 --behaviors $B5 --limit 240 --workers 10 --out $E/hoosier_rclup_b5 --cap-usd 90
step eval_richtext_b5 $PY -m rclup.run_episodes --dataset hoosier --split test --controller rich_text_mc --carrier $R/models/hoosierF_A_richtext/carrier.pt --calibrator $R/models/hoosierF_C_richtext/calibrator.pt --lam 0.02 --behaviors $B5 --limit 240 --workers 10 --out $E/hoosier_richtext_mc_b5 --cap-usd 90
step eval_disagree_b5 $PY -m rclup.run_episodes --dataset hoosier --split test --controller disagreement_ask --behaviors $B5 --limit 240 --workers 10 --out $E/hoosier_disagreement_b5 --cap-usd 90
# ablation rows: normal behavior, budget 2
step eval_generic $PY -m rclup.run_episodes --dataset hoosier --split test --controller generic_latent_mc --carrier $R/models/hoosierF_A_generic/carrier.pt --calibrator $R/models/hoosierF_C_generic/calibrator.pt --lam 0.02 --limit 48 --workers 10 --out $E/hoosier_generic_mc --cap-usd 90
step eval_scalar $PY -m rclup.run_episodes --dataset hoosier --split test --controller scalar_mc --carrier $R/models/hoosierF_A_scalar/carrier.pt --calibrator $R/models/hoosierF_C_scalar/calibrator.pt --lam 0.02 --limit 48 --workers 10 --out $E/hoosier_scalar_mc --cap-usd 90
step eval_qref $PY -m rclup.run_episodes --dataset hoosier --split test --controller rclup_q_ref --carrier $R/models/hoosierF_A/carrier.pt --lam 0.02 --limit 48 --workers 10 --out $E/hoosier_rclup_qref --cap-usd 90
for b in 1 4; do
  step eval_rclup_budget$b $PY -m rclup.run_episodes --dataset hoosier --split test --controller rclup --carrier $R/models/hoosierF_B/carrier.pt --calibrator $R/models/hoosierF_C/calibrator.pt --lam 0.02 --items-budget $b --limit 48 --workers 10 --out $E/hoosier_rclup_budget$b --cap-usd 90
done
step report_b5 $PY -m rclup.report --dataset hoosier --reference $E/hoosier_text_mas_b5 --runs $E/hoosier_text_mas_b5,$E/hoosier_disagreement_b5,$E/hoosier_richtext_mc_b5,$E/hoosier_rclup_b5 --out $E/report_hoosier_b5
echo "[$(date +%H:%M:%S)] HOOSIER FULL DONE"
