#!/bin/bash
# Atomic questions: the hybrid (ours) at caps 5 and 7 (after v7_labels).
JOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/v7; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
until grep -q "V7 LABELS DONE" $R/logs/v7_labels_b.log 2>/dev/null; do sleep 60; done
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=2 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$? $(grep -E '"end_to_end_match"' $LOGS/$name.log | tail -1)"; }
AN=eval/analysis_v7; LAM=$(cat $R/$AN/chosen_lam.txt); BETA=$(cat $R/$AN/chosen_beta.txt); export RCLUP_VOI_BETA=$BETA
B5=normal,rambling,impatience,self_contradictory,unsupported_request; E=eval/v7
RT="--controller rich_text_mc --carrier $R/models/hoosier7_A_richtext/carrier.pt --calibrator $R/models/hoosier7_C_richtext/calibrator.pt --lam $LAM --variant v6 --intake-policy atomic --behaviors $B5 --limit 240 --workers 10 --cap-usd 300"
for T in 5 7; do step voi_t$T $PY -m rclup.run_episodes --dataset hoosier --split test $RT --max-turns $T --out $E/voi/turns$T/hoosier_richtext_mc_b5; done
echo "[$(date +%H:%M:%S)] V7 LEARNED2 DONE"
