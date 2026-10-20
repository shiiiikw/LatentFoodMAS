#!/bin/bash
# Worker B after its baselines: the learned stop-rule variant (fixed order) at caps 8/6/10, taking load off worker A.
JOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/v7
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB RCLUP_ORDERED_GROUPS=1
until grep -q "V7 BASELINES3 DONE" $R/logs/v7_baselines3.log 2>/dev/null; do sleep 60; done
until [ -f $R/eval/analysis_v7/chosen_beta.txt ]; do sleep 30; done
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=2 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$? $(grep -E '"end_to_end_match"' $LOGS/$name.log | tail -1)"; }
AN=eval/analysis_v7; LAM=$(cat $R/$AN/chosen_lam.txt)
B5=normal,rambling,impatience,self_contradictory,unsupported_request; E=eval/v7
RT="--controller rich_text_mc --carrier $R/models/hoosier7_A_richtext/carrier.pt --calibrator $R/models/hoosier7_C_richtext/calibrator.pt --lam $LAM --variant v6 --intake-policy atomic --behaviors $B5 --limit 240 --workers 10 --cap-usd 300"
for T in 8 6 10; do step wB_ord_t$T $PY -m rclup.run_episodes --dataset hoosier --split test $RT --max-turns $T --out $E/ord/turns$T/hoosier_richtext_mc_b5; done
echo "[$(date +%H:%M:%S)] WORKER B TAIL DONE"
