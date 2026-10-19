#!/bin/bash
# After the resolver grounding fix: rerun the v5 main systems and the v6 heuristic runs that finished on the old code,
# so every table row comes from the same code. Gated on the v6 heuristic chain finishing.
JOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/rerun; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
until grep -q "V6 HEURISTIC CHAIN DONE" $R/logs/v6/chain_heur.log 2>/dev/null; do sleep 60; done
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=2 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$? $(grep -E '"end_to_end_match"' $LOGS/$name.log | tail -1)"; }
B5=normal,rambling,impatience,self_contradictory,unsupported_request; CAP=260
V6="--dataset hoosier --split test --variant v6 --behaviors $B5 --limit 240 --workers 10 --cap-usd $CAP"
for C in fixed_order disagreement_ask; do rm -rf $R/eval/v6/hoosier_${C}_b5; step v6_$C $PY -m rclup.run_episodes $V6 --controller $C --out eval/v6/hoosier_${C}_b5; done
V5="--dataset hoosier --split test --variant v5 --behaviors $B5 --limit 240 --workers 10 --cap-usd $CAP"; E=eval/final_fixed
step v5_text $PY -m rclup.run_episodes $V5 --controller text_mas --out $E/hoosier_text_mas_b5
step v5_nofb $PY -m rclup.run_episodes --dataset hoosier --split test --variant v5nofb --behaviors $B5 --limit 240 --workers 10 --cap-usd $CAP --controller text_mas --out $E/hoosier_text_mas_nofb_b5
step v5_voi $PY -m rclup.run_episodes $V5 --controller voi_det --out $E/hoosier_voi_det_b5
step v5_richtext $PY -m rclup.run_episodes $V5 --controller rich_text_mc --carrier $R/models/hoosier5_A_richtext/carrier.pt --calibrator $R/models/hoosier5_C_richtext/calibrator.pt --lam 0.02 --out $E/hoosier_richtext_mc_b5
for C in disagreement_ask max_uncertainty fixed_order; do step v5_$C $PY -m rclup.run_episodes $V5 --controller $C --out $E/hoosier_${C}_b5; done
echo "[$(date +%H:%M:%S)] RERUN DONE"
