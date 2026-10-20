#!/bin/bash
# Third worker: after the atomic controllers are trained, run the input ablations (cap 8) and the hybrid at caps 5/7.
JOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/v7; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
until grep -q '"ready"' $R/worker_$JOB.json 2>/dev/null; do sleep 20; done
until [ -f $R/eval/analysis_v7/chosen_beta.txt ]; do sleep 30; done
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=2 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$? $(grep -E '"end_to_end_match"' $LOGS/$name.log | tail -1)"; }
AN=eval/analysis_v7; LAM=$(cat $R/$AN/chosen_lam.txt); BETA=$(cat $R/$AN/chosen_beta.txt); export RCLUP_VOI_BETA=$BETA
B5=normal,rambling,impatience,self_contradictory,unsupported_request; E=eval/v7; CAP=300
RT="--controller rich_text_mc --carrier $R/models/hoosier7_A_richtext/carrier.pt --calibrator $R/models/hoosier7_C_richtext/calibrator.pt --lam $LAM --variant v6 --intake-policy atomic --behaviors $B5 --limit 240 --workers 10 --cap-usd $CAP"
for T in 7 5; do step w3_voi_t$T $PY -m rclup.run_episodes --dataset hoosier --split test $RT --max-turns $T --out $E/voi/turns$T/hoosier_richtext_mc_b5; done
for ABL in dependency receiver_state; do TAG=richtext_no_$ABL
  step w3_eval_$TAG $PY -m rclup.run_episodes --dataset hoosier --split test --controller rich_text_mc --carrier $R/models/hoosier7_A_$TAG/carrier.pt --calibrator $R/models/hoosier7_C_$TAG/calibrator.pt --lam $LAM --ablate $ABL --variant v6 --intake-policy atomic --max-turns 8 --behaviors $B5 --limit 240 --workers 10 --cap-usd $CAP --out $E/turns8/hoosier_${TAG}_mc_b5
done
echo "[$(date +%H:%M:%S)] WORKER3 DONE"
