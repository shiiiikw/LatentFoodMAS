#!/bin/bash
# Learned stop rule with fixed question order: choose lam on validation (learnedord rule), then closed-loop at caps 8/6/7/10.
JOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/v6ord; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB RCLUP_ORDERED_GROUPS=1
until grep -q "WORKER A TAIL DONE" $R/logs/workerA_tail.log 2>/dev/null; do sleep 60; done
until [ -f $R/models/hoosier6c_C_richtext/calibrator.pt ]; do sleep 60; done
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=2 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$? $(grep -E '"end_to_end_match"' $LOGS/$name.log | tail -1)"; }
MIX=collect/hoosier_v6b_mix; CARR="richtext=models/hoosier6c_A_richtext/carrier.pt"
for L in 0.0 0.02 0.05 0.1 0.2; do
  step lamval_$L $PY -m rclup.analysis_questions --collect $MIX --dataset hoosier --carriers $CARR --val-only --lam $L --out eval/analysis_lam_ord/val_$L
done
BEST=$($PY - <<'PYEOF'
import json
R="/ocean/projects/cis260058p/kshi4/rclup_runs/eval/analysis_lam_ord"
best=None
for L in ("0.0","0.02","0.05","0.1","0.2"):
    row=[r for r in json.load(open(f"{R}/val_{L}/table.json"))["rules"] if r["rule"]=="learnedord:richtext"][0]
    key=(round(row["mean_error"],4), row["ask_rate"])
    if best is None or key<best[0]: best=(key,L)
print(best[1])
PYEOF
)
echo "chosen lam=$BEST"; echo $BEST > $R/eval/analysis_lam_ord/chosen_lam.txt
B5=normal,rambling,impatience,self_contradictory,unsupported_request; E=eval/v6
RT="--controller rich_text_mc --carrier $R/models/hoosier6c_A_richtext/carrier.pt --calibrator $R/models/hoosier6c_C_richtext/calibrator.pt --lam $BEST"
for T in 8 6 7 10; do
  step ord_t$T $PY -m rclup.run_episodes --dataset hoosier --split test $RT --variant v6 --max-turns $T --behaviors $B5 --limit 240 --workers 10 --out $E/ord/turns$T/hoosier_richtext_mc_b5 --cap-usd 280
done
echo "[$(date +%H:%M:%S)] V6 ORDERED CHAIN DONE"
