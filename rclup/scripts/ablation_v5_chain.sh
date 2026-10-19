#!/bin/bash
# Reviewer chain (b): controller-input ablations on the v5 labels (no new collection): dependency features off / receiver-state off.
# Train A+C on cached v5 labels, then closed-loop eval on v5 (5 behaviors, 240 episodes), paired with eval/final/hoosier_richtext_mc_b5.
JOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/ablation_v5; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=2 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$?"; }
C=collect/hoosier_v5_repr; CAL=collect/hoosier_v5_cal
for ABL in dependency receiver_state; do
  step train_A_$ABL $PY -m rclup.train --collect $C --out models/hoosier5_A_richtext_no_$ABL --epochs 15 --no-latent --ablate $ABL
  step train_C_$ABL $PY -m rclup.train --phase C --init $R/models/hoosier5_A_richtext_no_$ABL/carrier.pt --collect $CAL --out models/hoosier5_C_richtext_no_$ABL --no-latent --ablate $ABL
done
for i in $(seq 1 180); do grep -q '"ready"' $R/worker_$JOB.json 2>/dev/null && break; sleep 10; done
B5=normal,rambling,impatience,self_contradictory,unsupported_request
for ABL in dependency receiver_state; do
  step eval_$ABL $PY -m rclup.run_episodes --dataset hoosier --split test --controller rich_text_mc --variant v5 --ablate $ABL \
    --carrier $R/models/hoosier5_A_richtext_no_$ABL/carrier.pt --calibrator $R/models/hoosier5_C_richtext_no_$ABL/calibrator.pt --lam 0.02 \
    --behaviors $B5 --limit 240 --workers 10 --out eval/final/hoosier_richtext_mc_no_${ABL}_b5 --cap-usd 250
done
echo "[$(date +%H:%M:%S)] ABLATION V5 DONE"
