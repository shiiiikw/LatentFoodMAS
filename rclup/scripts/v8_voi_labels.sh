#!/bin/bash
# Atomic questions, labels with the budget-aware VOI policy as branch continuation (learned = one-step improvement over it):
# collect at caps 8 and 6, calibration, train, select lam/beta, hybrid at caps 5/6/7/8/10. Runs on the given worker once ready.
JOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/v8; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
until grep -q '"ready"' $R/worker_$JOB.json 2>/dev/null; do sleep 30; done
until grep -q "WORKER3 DONE" $R/logs/v7_worker3.log 2>/dev/null; do sleep 60; done
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=2 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$? $(grep -E '"end_to_end_match"' $LOGS/$name.log | tail -1)"; }
CAP=320; C8=collect/hoosier_v8_repr; C6=collect/hoosier_v8_repr_t6; CAL=collect/hoosier_v8_cal; rm -rf $R/$C8 $R/$C6 $R/$CAL
COL="--dataset hoosier --partition representation --roots 200 --seeds 13 --max-commits 3 --max-questions 6 --prefixes-per-root 2 --workers 10 --variant v6 --intake-policy atomic --continuation voi --no-latent --cap-usd $CAP"
step collect8 $PY -m rclup.collect $COL --out $C8
step collect6 $PY -m rclup.collect ${COL/--roots 200/--roots 100} --max-turns 6 --out $C6
step collect_cal $PY -m rclup.collect --dataset hoosier --partition action_calibration --roots 30 --seeds 13 --max-commits 3 --max-questions 6 --prefixes-per-root 2 --workers 10 --variant v6 --intake-policy atomic --continuation voi --no-latent --out $CAL --cap-usd $CAP
MIX=$R/collect/hoosier_v8_mix; rm -rf $MIX; mkdir -p $MIX/prefixes
for f in $R/$C8/prefixes/*.pt; do ln -s $f $MIX/prefixes/t8__$(basename $f); done
for f in $R/$C6/prefixes/*.pt; do ln -s $f $MIX/prefixes/t6__$(basename $f); done
step train_A $PY -m rclup.train --collect collect/hoosier_v8_mix --out models/hoosier8_A_richtext --epochs 15 --no-latent
step train_C $PY -m rclup.train --phase C --init $R/models/hoosier8_A_richtext/carrier.pt --collect $CAL --out models/hoosier8_C_richtext --no-latent
CARR="richtext=models/hoosier8_A_richtext/carrier.pt"; AN=eval/analysis_v8; mkdir -p $R/$AN
for L in 0.0 0.05 0.1 0.2; do step lamval_$L $PY -m rclup.analysis_questions --collect collect/hoosier_v8_mix --dataset hoosier --carriers $CARR --val-only --lam $L --out $AN/lam_$L; done
LAM=$($PY -c "
import json;R='$R/$AN'
rows=[(L,[r for r in json.load(open(f'{R}/lam_{L}/table.json'))['rules'] if r['rule']=='learned:richtext'][0]) for L in ('0.0','0.05','0.1','0.2')]
print(min(rows,key=lambda x:(round(x[1]['mean_error'],4),x[1]['ask_rate']))[0])")
for B in 0.25 0.5 0.75; do step betaval_$B $PY -m rclup.analysis_questions --collect collect/hoosier_v8_mix --dataset hoosier --carriers $CARR --val-only --lam $LAM --beta $B --out $AN/beta_$B; done
BETA=$($PY -c "
import json;R='$R/$AN'
rows=[(B,[r for r in json.load(open(f'{R}/beta_{B}/table.json'))['rules'] if r['rule']=='learnedvoi:richtext'][0]) for B in ('0.25','0.5','0.75')]
print(min(rows,key=lambda x:(round(x[1]['mean_error'],4),x[1]['ask_rate']))[0])")
echo "chosen lam=$LAM beta=$BETA"; echo $LAM > $R/$AN/chosen_lam.txt; echo $BETA > $R/$AN/chosen_beta.txt; export RCLUP_VOI_BETA=$BETA
B5=normal,rambling,impatience,self_contradictory,unsupported_request; E=eval/v8
RT="--controller rich_text_mc --carrier $R/models/hoosier8_A_richtext/carrier.pt --calibrator $R/models/hoosier8_C_richtext/calibrator.pt --lam $LAM --variant v6 --intake-policy atomic --behaviors $B5 --limit 240 --workers 10 --cap-usd $CAP"
for T in 8 6 10 7 5; do step voi_t$T $PY -m rclup.run_episodes --dataset hoosier --split test $RT --max-turns $T --out $E/voi/turns$T/hoosier_richtext_mc_b5; done
for T in 8 6; do step learned_t$T $PY -m rclup.run_episodes --dataset hoosier --split test $RT --max-turns $T --out $E/lam/turns$T/hoosier_richtext_mc_b5; done
echo "[$(date +%H:%M:%S)] V8 DONE"
