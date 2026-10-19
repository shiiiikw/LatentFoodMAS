#!/bin/bash
# v6 (budget-aware question menu): labels -> learned rich-text controller -> eval; then turn-cap sweep for the learned controller.
JOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/v6; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=2 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$? $(grep -E '"end_to_end_match"' $LOGS/$name.log | tail -1)"; }
C=collect/hoosier_v6_repr; CAL=collect/hoosier_v6_cal; CAP=260; rm -rf $R/$C $R/$CAL
step collect $PY -m rclup.collect --dataset hoosier --partition representation --roots 200 --seeds 13 --max-commits 3 --max-questions 6 --prefixes-per-root 2 --workers 10 --variant v6 --no-latent --out $C --cap-usd $CAP
step collect_cal $PY -m rclup.collect --dataset hoosier --partition action_calibration --roots 30 --seeds 13 --max-commits 3 --max-questions 6 --prefixes-per-root 2 --workers 10 --variant v6 --no-latent --out $CAL --cap-usd $CAP
step train_A $PY -m rclup.train --collect $C --out models/hoosier6_A_richtext --epochs 15 --no-latent
step train_C $PY -m rclup.train --phase C --init $R/models/hoosier6_A_richtext/carrier.pt --collect $CAL --out models/hoosier6_C_richtext --no-latent
B5=normal,rambling,impatience,self_contradictory,unsupported_request; E=eval/v6
RT="--controller rich_text_mc --carrier $R/models/hoosier6_A_richtext/carrier.pt --calibrator $R/models/hoosier6_C_richtext/calibrator.pt --lam 0.02"
step richtext $PY -m rclup.run_episodes --dataset hoosier --split test $RT --variant v6 --behaviors $B5 --limit 240 --workers 10 --out $E/hoosier_richtext_mc_b5 --cap-usd $CAP
for T in 6 7 10; do
  step richtext_t$T $PY -m rclup.run_episodes --dataset hoosier --split test $RT --variant v6 --max-turns $T --behaviors $B5 --limit 240 --workers 10 --out $E/turns$T/hoosier_richtext_mc_b5 --cap-usd $CAP
done
echo "[$(date +%H:%M:%S)] V6 LABEL CHAIN DONE"
