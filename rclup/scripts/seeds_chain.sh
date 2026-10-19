#!/bin/bash
# Main-table seeds 1 and 2 for the v5 systems (same 240 episodes, different sampling seeds; user replies cached by transcript).
JOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/seeds; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=2 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$?"; }
B5=normal,rambling,impatience,self_contradictory,unsupported_request
for S in 1 2; do
  step text_s$S $PY -m rclup.run_episodes --dataset hoosier --split test --controller text_mas --variant v5 --seed $S --behaviors $B5 --limit 240 --workers 10 --out eval/final/seed$S/hoosier_text_mas_b5 --cap-usd 250
  step voi_s$S $PY -m rclup.run_episodes --dataset hoosier --split test --controller voi_det --variant v5 --seed $S --behaviors $B5 --limit 240 --workers 10 --out eval/final/seed$S/hoosier_voi_det_b5 --cap-usd 250
  step richtext_s$S $PY -m rclup.run_episodes --dataset hoosier --split test --controller rich_text_mc --variant v5 --seed $S --carrier $R/models/hoosier5_A_richtext/carrier.pt --calibrator $R/models/hoosier5_C_richtext/calibrator.pt --lam 0.02 --behaviors $B5 --limit 240 --workers 10 --out eval/final/seed$S/hoosier_richtext_mc_b5 --cap-usd 250
  step nofb_s$S $PY -m rclup.run_episodes --dataset hoosier --split test --controller text_mas --variant v5nofb --seed $S --behaviors $B5 --limit 240 --workers 10 --out eval/final/seed$S/hoosier_text_mas_nofb_b5 --cap-usd 250
done
echo "[$(date +%H:%M:%S)] SEEDS DONE"
