#!/bin/bash
# Worker A: v6 heuristic controllers (cap 8), turn-cap sweep (6/7/10), then v5 reruns -- all on the fixed resolver.
JOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/workerA; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=2 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$? $(grep -E '"end_to_end_match"' $LOGS/$name.log | tail -1)"; }
B5=normal,rambling,impatience,self_contradictory,unsupported_request; CAP=270; E=eval/v6
V6="--dataset hoosier --split test --variant v6 --behaviors $B5 --limit 240 --workers 10 --cap-usd $CAP"
for C in fixed_order voi_det disagreement_ask max_uncertainty; do step v6_$C $PY -m rclup.run_episodes $V6 --controller $C --out $E/hoosier_${C}_b5; done
for T in 6 7; do for C in fixed_order voi_det disagreement_ask; do step v6_${C}_t$T $PY -m rclup.run_episodes $V6 --controller $C --max-turns $T --out $E/turns$T/hoosier_${C}_b5; done; done
for C in fixed_order voi_det; do step v6_${C}_t10 $PY -m rclup.run_episodes $V6 --controller $C --max-turns 10 --out $E/turns10/hoosier_${C}_b5; done
V5="--dataset hoosier --split test --variant v5 --behaviors $B5 --limit 240 --workers 10 --cap-usd $CAP"; F=eval/final_fixed
step v5_text $PY -m rclup.run_episodes $V5 --controller text_mas --out $F/hoosier_text_mas_b5
step v5_nofb $PY -m rclup.run_episodes --dataset hoosier --split test --variant v5nofb --behaviors $B5 --limit 240 --workers 10 --cap-usd $CAP --controller text_mas --out $F/hoosier_text_mas_nofb_b5
step v5_voi $PY -m rclup.run_episodes $V5 --controller voi_det --out $F/hoosier_voi_det_b5
step v5_richtext $PY -m rclup.run_episodes $V5 --controller rich_text_mc --carrier $R/models/hoosier5_A_richtext/carrier.pt --calibrator $R/models/hoosier5_C_richtext/calibrator.pt --lam 0.02 --out $F/hoosier_richtext_mc_b5
for C in disagreement_ask max_uncertainty fixed_order; do step v5_$C $PY -m rclup.run_episodes $V5 --controller $C --out $F/hoosier_${C}_b5; done
echo "[$(date +%H:%M:%S)] WORKER A DONE"
