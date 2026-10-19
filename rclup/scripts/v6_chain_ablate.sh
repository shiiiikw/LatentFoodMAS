#!/bin/bash
# Reviewer chain (b) on v6 labels: same labels, same architecture, only the controller inputs change (no dependency features /
# no receiver state). Waits for the v6 label chain to finish training before starting.
JOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/v6; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
until grep -q "END train_C" $LOGS/chain_labels.log 2>/dev/null; do sleep 60; done
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=2 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$? $(grep -E '"end_to_end_match"' $LOGS/$name.log | tail -1)"; }
C=collect/hoosier_v6_repr; CAL=collect/hoosier_v6_cal; B5=normal,rambling,impatience,self_contradictory,unsupported_request; E=eval/v6
for ABL in dependency receiver_state; do
  step train_A_$ABL $PY -m rclup.train --collect $C --out models/hoosier6_A_richtext_no_$ABL --epochs 15 --no-latent --ablate $ABL
  step train_C_$ABL $PY -m rclup.train --phase C --init $R/models/hoosier6_A_richtext_no_$ABL/carrier.pt --collect $CAL --out models/hoosier6_C_richtext_no_$ABL --no-latent --ablate $ABL
  step eval_$ABL $PY -m rclup.run_episodes --dataset hoosier --split test --controller rich_text_mc --variant v6 --ablate $ABL \
    --carrier $R/models/hoosier6_A_richtext_no_$ABL/carrier.pt --calibrator $R/models/hoosier6_C_richtext_no_$ABL/calibrator.pt --lam 0.02 \
    --behaviors $B5 --limit 240 --workers 10 --out $E/hoosier_richtext_mc_no_${ABL}_b5 --cap-usd 260
done
echo "[$(date +%H:%M:%S)] V6 ABLATION CHAIN DONE"
