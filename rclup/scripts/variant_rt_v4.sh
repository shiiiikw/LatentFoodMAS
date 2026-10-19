#!/bin/bash
JOB=49451001; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/pipeline_hoosier_full
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=1 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$?"; }
until grep -q "RT v3 DONE" /ocean/projects/cis260058p/kshi4/rclup_runs/logs/variant_rt_v3.log 2>/dev/null; do sleep 30; done
E=eval/full4; V=v4; B5=normal,rambling,impatience,self_contradictory,unsupported_request
step ${V}_richtext_b5 $PY -m rclup.run_episodes --dataset hoosier --split test --controller rich_text_mc --variant $V --carrier $R/models/hoosierF_A_richtext/carrier.pt --calibrator $R/models/hoosierF_C_richtext/calibrator.pt --lam 0.02 --behaviors $B5 --limit 240 --workers 10 --out $E/hoosier_richtext_mc_b5 --cap-usd 150
step ${V}_report2 $PY -m rclup.report --dataset hoosier --reference $E/hoosier_text_mas_b5 --runs $E/hoosier_text_mas_b5,$E/hoosier_disagreement_b5,$E/hoosier_richtext_mc_b5,$E/hoosier_rclup_b5 --out $E/report_hoosier_b5
echo "[$(date +%H:%M:%S)] RT v4 DONE"
