#!/bin/bash
# v6 corrected label pipeline (fixed-order continuation after the forced decision, fixed resolver):
# labels at caps 8 and 6 -> calibration labels -> train on the mix -> eval at caps 8/7/6 -> input ablations -> offline analysis.
JOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/v6b; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=2 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$? $(grep -E '"end_to_end_match"' $LOGS/$name.log | tail -1)"; }
CAP=270; C8=collect/hoosier_v6b_repr; C6=collect/hoosier_v6b_repr_t6; CAL=collect/hoosier_v6b_cal; rm -rf $R/$C8 $R/$C6 $R/$CAL
COL="--dataset hoosier --partition representation --roots 200 --seeds 13 --max-commits 3 --max-questions 6 --prefixes-per-root 2 --workers 10 --variant v6 --no-latent --cap-usd $CAP"
step collect8 $PY -m rclup.collect $COL --out $C8
step collect6 $PY -m rclup.collect $COL --max-turns 6 --out $C6
step collect_cal $PY -m rclup.collect --dataset hoosier --partition action_calibration --roots 30 --seeds 13 --max-commits 3 --max-questions 6 --prefixes-per-root 2 --workers 10 --variant v6 --no-latent --out $CAL --cap-usd $CAP
MIX=$R/collect/hoosier_v6b_mix; rm -rf $MIX; mkdir -p $MIX/prefixes
for f in $R/$C8/prefixes/*.pt; do ln -s $f $MIX/prefixes/t8__$(basename $f); done
for f in $R/$C6/prefixes/*.pt; do ln -s $f $MIX/prefixes/t6__$(basename $f); done
B5=normal,rambling,impatience,self_contradictory,unsupported_request; E=eval/v6
for ABL in none dependency receiver_state; do
  if [ $ABL = none ]; then TAG=richtext; AF=""; else TAG=richtext_no_$ABL; AF="--ablate $ABL"; fi
  step train_A_$TAG $PY -m rclup.train --collect collect/hoosier_v6b_mix --out models/hoosier6b_A_$TAG --epochs 15 --no-latent $AF
  step train_C_$TAG $PY -m rclup.train --phase C --init $R/models/hoosier6b_A_$TAG/carrier.pt --collect $CAL --out models/hoosier6b_C_$TAG --no-latent $AF
  RT="--controller rich_text_mc --carrier $R/models/hoosier6b_A_$TAG/carrier.pt --calibrator $R/models/hoosier6b_C_$TAG/calibrator.pt --lam 0.02 $AF"
  step eval_$TAG $PY -m rclup.run_episodes --dataset hoosier --split test $RT --variant v6 --behaviors $B5 --limit 240 --workers 10 --out $E/hoosier_${TAG}_mc_b5 --cap-usd $CAP
  if [ $ABL = none ]; then for T in 6 7; do
    step eval_${TAG}_t$T $PY -m rclup.run_episodes --dataset hoosier --split test $RT --variant v6 --max-turns $T --behaviors $B5 --limit 240 --workers 10 --out $E/turns$T/hoosier_richtext_mc_b5 --cap-usd $CAP
  done; fi
done
CARR="richtext=models/hoosier6b_A_richtext/carrier.pt,richtext_nodep=models/hoosier6b_A_richtext_no_dependency/carrier.pt,richtext_norecv=models/hoosier6b_A_richtext_no_receiver_state/carrier.pt"
ABLS="richtext_nodep:dependency,richtext_norecv:receiver_state"
step analysis_val $PY -m rclup.analysis_questions --collect $C8 --dataset hoosier --carriers $CARR --ablate $ABLS --val-only --out eval/analysis_questions_hoosier_v6_val
step analysis_all $PY -m rclup.analysis_questions --collect $C8 --dataset hoosier --carriers $CARR --ablate $ABLS --out eval/analysis_questions_hoosier_v6_all
step analysis_t6 $PY -m rclup.analysis_questions --collect $C6 --dataset hoosier --carriers $CARR --ablate $ABLS --out eval/analysis_questions_hoosier_v6_t6
echo "[$(date +%H:%M:%S)] V6B CHAIN DONE"
