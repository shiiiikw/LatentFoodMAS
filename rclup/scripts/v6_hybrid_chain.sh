#!/bin/bash
# Hybrid learned + VOI: choose beta on validation (lam from the learned selection), closed-loop at caps 8/6/7/10. Runs after the v6c chain.
JOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/v6voi; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
until grep -q "V6C CHAIN DONE" $R/logs/v6c_chain.log 2>/dev/null; do sleep 60; done
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=2 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$? $(grep -E '"end_to_end_match"' $LOGS/$name.log | tail -1)"; }
MIX=collect/hoosier_v6b_mix; CARR="richtext=models/hoosier6c_A_richtext/carrier.pt"; LAM=$(cat $R/eval/analysis_lam/chosen_lam.txt)
for B in 0.25 0.5 0.75; do
  step betaval_$B $PY -m rclup.analysis_questions --collect $MIX --dataset hoosier --carriers $CARR --val-only --lam $LAM --beta $B --out eval/analysis_beta/val_$B
done
BEST=$($PY - <<'PYEOF'
import json
R="/ocean/projects/cis260058p/kshi4/rclup_runs/eval/analysis_beta"
best=None
for B in ("0.25","0.5","0.75"):
    row=[r for r in json.load(open(f"{R}/val_{B}/table.json"))["rules"] if r["rule"]=="learnedvoi:richtext"][0]
    key=(round(row["mean_error"],4), row["ask_rate"])
    if best is None or key<best[0]: best=(key,B)
print(best[1])
PYEOF
)
echo "chosen beta=$BEST (lam=$LAM)"; echo $BEST > $R/eval/analysis_beta/chosen_beta.txt
export RCLUP_VOI_BETA=$BEST
B5=normal,rambling,impatience,self_contradictory,unsupported_request; E=eval/v6
RT="--controller rich_text_mc --carrier $R/models/hoosier6c_A_richtext/carrier.pt --calibrator $R/models/hoosier6c_C_richtext/calibrator.pt --lam $LAM"
for T in 8 6 7 10; do
  step voi_t$T $PY -m rclup.run_episodes --dataset hoosier --split test $RT --variant v6 --max-turns $T --behaviors $B5 --limit 240 --workers 10 --out $E/voi/turns$T/hoosier_richtext_mc_b5 --cap-usd 280
done
echo "[$(date +%H:%M:%S)] V6 HYBRID CHAIN DONE"
