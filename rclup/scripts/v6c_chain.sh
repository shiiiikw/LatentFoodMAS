#!/bin/bash
# v6 learned controller on the corrected labels (unresolved commit masked): train 3 variants, choose lam on validation,
# closed-loop at caps 10/8/7/6, input ablations at cap 8, offline analysis.
JOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/v6c; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=2 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$? $(grep -E '"end_to_end_match"' $LOGS/$name.log | tail -1)"; }
CAP=280; MIX=collect/hoosier_v6b_mix; CAL=collect/hoosier_v6b_cal; B5=normal,rambling,impatience,self_contradictory,unsupported_request; E=eval/v6
for ABL in none dependency receiver_state; do
  if [ $ABL = none ]; then TAG=richtext; AF=""; else TAG=richtext_no_$ABL; AF="--ablate $ABL"; fi
  step train_A_$TAG $PY -m rclup.train --collect $MIX --out models/hoosier6c_A_$TAG --epochs 15 --no-latent $AF
  step train_C_$TAG $PY -m rclup.train --phase C --init $R/models/hoosier6c_A_$TAG/carrier.pt --collect $CAL --out models/hoosier6c_C_$TAG --no-latent $AF
done
CARR="richtext=models/hoosier6c_A_richtext/carrier.pt"
for L in 0.0 0.02 0.05 0.1 0.2; do
  step lamval_$L $PY -m rclup.analysis_questions --collect $MIX --dataset hoosier --carriers $CARR --val-only --lam $L --out eval/analysis_lam/val_$L
done
BEST=$($PY - <<'PYEOF'
import json
R="/ocean/projects/cis260058p/kshi4/rclup_runs/eval/analysis_lam"
best=None
for L in ("0.0","0.02","0.05","0.1","0.2"):
    row=[r for r in json.load(open(f"{R}/val_{L}/table.json"))["rules"] if r["rule"]=="learned:richtext"][0]
    key=(round(row["mean_error"],4), row["ask_rate"])
    if best is None or key<best[0]: best=(key,L)
print(best[1])
PYEOF
)
echo "chosen lam=$BEST"; echo $BEST > $R/eval/analysis_lam/chosen_lam.txt
RT="--controller rich_text_mc --carrier $R/models/hoosier6c_A_richtext/carrier.pt --calibrator $R/models/hoosier6c_C_richtext/calibrator.pt --lam $BEST"
for T in 8 6 7 10; do
  step learned_t$T $PY -m rclup.run_episodes --dataset hoosier --split test $RT --variant v6 --max-turns $T --behaviors $B5 --limit 240 --workers 10 --out $E/lam/turns$T/hoosier_richtext_mc_b5 --cap-usd $CAP
done
for ABL in dependency receiver_state; do
  TAG=richtext_no_$ABL
  step eval_$TAG $PY -m rclup.run_episodes --dataset hoosier --split test --controller rich_text_mc --carrier $R/models/hoosier6c_A_$TAG/carrier.pt --calibrator $R/models/hoosier6c_C_$TAG/calibrator.pt --lam $BEST --ablate $ABL --variant v6 --behaviors $B5 --limit 240 --workers 10 --out $E/hoosier_${TAG}_mc_b5 --cap-usd $CAP
done
CARR3="richtext=models/hoosier6c_A_richtext/carrier.pt,richtext_nodep=models/hoosier6c_A_richtext_no_dependency/carrier.pt,richtext_norecv=models/hoosier6c_A_richtext_no_receiver_state/carrier.pt"
ABLS="richtext_nodep:dependency,richtext_norecv:receiver_state"
step analysis_val $PY -m rclup.analysis_questions --collect $MIX --dataset hoosier --carriers $CARR3 --ablate $ABLS --val-only --lam $BEST --out eval/analysis_questions_hoosier_v6_val
step analysis_t8 $PY -m rclup.analysis_questions --collect collect/hoosier_v6b_repr --dataset hoosier --carriers $CARR3 --ablate $ABLS --lam $BEST --out eval/analysis_questions_hoosier_v6_t8
step analysis_t6 $PY -m rclup.analysis_questions --collect collect/hoosier_v6b_repr_t6 --dataset hoosier --carriers $CARR3 --ablate $ABLS --lam $BEST --out eval/analysis_questions_hoosier_v6_t6
echo "[$(date +%H:%M:%S)] V6C CHAIN DONE"
