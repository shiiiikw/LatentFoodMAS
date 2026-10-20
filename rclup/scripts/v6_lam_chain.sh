#!/bin/bash
# After the v6 learned controller is trained: choose the asking cost lam on the offline validation split (cap-8 labels),
# then evaluate the learned controller closed-loop at caps 10/8/7/6 with that lam. Also ask-everything at cap 10 on worker A.
JOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/v6b; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
until grep -q "END train_C_richtext" $R/logs/v6b_chain.log 2>/dev/null; do sleep 60; done
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=2 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$? $(grep -E '"end_to_end_match"' $LOGS/$name.log | tail -1)"; }
CARR="richtext=models/hoosier6b_A_richtext/carrier.pt"
for L in 0.0 0.02 0.05 0.1 0.2; do
  step lamval_$L $PY -m rclup.analysis_questions --collect collect/hoosier_v6b_mix --dataset hoosier --carriers $CARR --val-only --lam $L --out eval/analysis_lam/val_$L
done
BEST=$($PY - <<'PYEOF'
import json
R="/ocean/projects/cis260058p/kshi4/rclup_runs/eval/analysis_lam"
best=None
for L in ("0.0","0.02","0.05","0.1","0.2"):
    t=json.load(open(f"{R}/val_{L}/table.json"))
    row=[r for r in t["rules"] if r["rule"]=="learned:richtext"][0]
    key=(round(row["mean_error"],4), row["ask_rate"])   # lowest validation error, then fewer asks
    if best is None or key<best[0]: best=(key,L)
print(best[1])
PYEOF
)
echo "chosen lam=$BEST"; echo $BEST > $R/eval/analysis_lam/chosen_lam.txt
B5=normal,rambling,impatience,self_contradictory,unsupported_request; CAP=270; E=eval/v6
RT="--controller rich_text_mc --carrier $R/models/hoosier6b_A_richtext/carrier.pt --calibrator $R/models/hoosier6b_C_richtext/calibrator.pt --lam $BEST"
for T in 10 8 7 6; do
  step learned_lam_t$T $PY -m rclup.run_episodes --dataset hoosier --split test $RT --variant v6 --max-turns $T --behaviors $B5 --limit 240 --workers 10 --out $E/lam/turns$T/hoosier_richtext_mc_b5 --cap-usd $CAP
done
echo "[$(date +%H:%M:%S)] LAM CHAIN DONE"
