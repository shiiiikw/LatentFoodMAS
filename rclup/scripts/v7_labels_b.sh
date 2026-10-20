#!/bin/bash
# Atomic questions: labels at caps 8 and 6 (fixed-order continuation), calibration labels, train, choose lam/beta on validation,
# closed-loop for the three learned variants at caps 8/6/10, input ablations at 8, offline analysis.
JOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/v7; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
until grep -q "END collect8" $R/logs/v7_labels.log 2>/dev/null; do sleep 20; done; kill $(cat $R/logs/v7_labels_pid.txt) 2>/dev/null; sleep 3; for st in $(squeue -s -j $JOB -h -o "%i" | grep "\."); do scancel $st; done; echo "took over from the original chain"
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=2 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$? $(grep -E '"end_to_end_match"' $LOGS/$name.log | tail -1)"; }
CAP=300; C8=collect/hoosier_v7_repr; C6=collect/hoosier_v7_repr_t6; CAL=collect/hoosier_v7_cal; rm -rf $R/$C6 $R/$CAL
COL="--dataset hoosier --partition representation --roots 200 --seeds 13 --max-commits 3 --max-questions 6 --prefixes-per-root 2 --workers 10 --variant v6 --intake-policy atomic --no-latent --cap-usd $CAP"
echo "cap-8 labels already collected by the original chain"
step collect6 $PY -m rclup.collect ${COL/--roots 200/--roots 100} --max-turns 6 --out $C6
step collect_cal $PY -m rclup.collect --dataset hoosier --partition action_calibration --roots 30 --seeds 13 --max-commits 3 --max-questions 6 --prefixes-per-root 2 --workers 10 --variant v6 --intake-policy atomic --no-latent --out $CAL --cap-usd $CAP
MIX=$R/collect/hoosier_v7_mix; rm -rf $MIX; mkdir -p $MIX/prefixes
for f in $R/$C8/prefixes/*.pt; do ln -s $f $MIX/prefixes/t8__$(basename $f); done
for f in $R/$C6/prefixes/*.pt; do ln -s $f $MIX/prefixes/t6__$(basename $f); done
for ABL in none dependency receiver_state; do
  if [ $ABL = none ]; then TAG=richtext; AF=""; else TAG=richtext_no_$ABL; AF="--ablate $ABL"; fi
  step train_A_$TAG $PY -m rclup.train --collect collect/hoosier_v7_mix --out models/hoosier7_A_$TAG --epochs 15 --no-latent $AF
  step train_C_$TAG $PY -m rclup.train --phase C --init $R/models/hoosier7_A_$TAG/carrier.pt --collect $CAL --out models/hoosier7_C_$TAG --no-latent $AF
done
CARR="richtext=models/hoosier7_A_richtext/carrier.pt"; AN=eval/analysis_v7
for L in 0.0 0.05 0.1 0.2; do step lamval_$L $PY -m rclup.analysis_questions --collect collect/hoosier_v7_mix --dataset hoosier --carriers $CARR --val-only --lam $L --out $AN/lam_$L; done
LAM=$($PY -c "
import json;R='$R/$AN'
rows=[(L,[r for r in json.load(open(f'{R}/lam_{L}/table.json'))['rules'] if r['rule']=='learned:richtext'][0]) for L in ('0.0','0.05','0.1','0.2')]
print(min(rows,key=lambda x:(round(x[1]['mean_error'],4),x[1]['ask_rate']))[0])")
for B in 0.25 0.5 0.75; do step betaval_$B $PY -m rclup.analysis_questions --collect collect/hoosier_v7_mix --dataset hoosier --carriers $CARR --val-only --lam $LAM --beta $B --out $AN/beta_$B; done
BETA=$($PY -c "
import json;R='$R/$AN'
rows=[(B,[r for r in json.load(open(f'{R}/beta_{B}/table.json'))['rules'] if r['rule']=='learnedvoi:richtext'][0]) for B in ('0.25','0.5','0.75')]
print(min(rows,key=lambda x:(round(x[1]['mean_error'],4),x[1]['ask_rate']))[0])")
echo "chosen lam=$LAM beta=$BETA"; mkdir -p $R/$AN; echo $LAM > $R/$AN/chosen_lam.txt; echo $BETA > $R/$AN/chosen_beta.txt
B5=normal,rambling,impatience,self_contradictory,unsupported_request; E=eval/v7
RT="--controller rich_text_mc --carrier $R/models/hoosier7_A_richtext/carrier.pt --calibrator $R/models/hoosier7_C_richtext/calibrator.pt --lam $LAM --variant v6 --intake-policy atomic --behaviors $B5 --limit 240 --workers 10 --cap-usd $CAP"
for T in 8 6 10; do
  RCLUP_VOI_BETA=$BETA step voi_t$T $PY -m rclup.run_episodes --dataset hoosier --split test $RT --max-turns $T --out $E/voi/turns$T/hoosier_richtext_mc_b5
done
for T in 8 6 10; do
  step learned_t$T $PY -m rclup.run_episodes --dataset hoosier --split test $RT --max-turns $T --out $E/lam/turns$T/hoosier_richtext_mc_b5
done
for T in 8 6 10; do
  RCLUP_ORDERED_GROUPS=1 step ord_t$T $PY -m rclup.run_episodes --dataset hoosier --split test $RT --max-turns $T --out $E/ord/turns$T/hoosier_richtext_mc_b5
done
for ABL in dependency receiver_state; do TAG=richtext_no_$ABL
  RCLUP_VOI_BETA=$BETA step eval_$TAG $PY -m rclup.run_episodes --dataset hoosier --split test --controller rich_text_mc --carrier $R/models/hoosier7_A_$TAG/carrier.pt --calibrator $R/models/hoosier7_C_$TAG/calibrator.pt --lam $LAM --ablate $ABL --variant v6 --intake-policy atomic --max-turns 8 --behaviors $B5 --limit 240 --workers 10 --cap-usd $CAP --out $E/turns8/hoosier_${TAG}_mc_b5
done
CARR3="richtext=models/hoosier7_A_richtext/carrier.pt,richtext_nodep=models/hoosier7_A_richtext_no_dependency/carrier.pt,richtext_norecv=models/hoosier7_A_richtext_no_receiver_state/carrier.pt"
ABLS="richtext_nodep:dependency,richtext_norecv:receiver_state"
step analysis_val $PY -m rclup.analysis_questions --collect collect/hoosier_v7_mix --dataset hoosier --carriers $CARR3 --ablate $ABLS --val-only --lam $LAM --beta $BETA --out $AN/situations_val
step analysis_t8 $PY -m rclup.analysis_questions --collect $C8 --dataset hoosier --carriers $CARR3 --ablate $ABLS --lam $LAM --beta $BETA --out $AN/situations_t8
step analysis_t6 $PY -m rclup.analysis_questions --collect $C6 --dataset hoosier --carriers $CARR3 --ablate $ABLS --lam $LAM --beta $BETA --out $AN/situations_t6
echo "[$(date +%H:%M:%S)] V7 LABELS DONE"
