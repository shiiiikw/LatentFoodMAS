#!/bin/bash
# Waits for the third worker, then: v5 labels (single seed) -> rich-text controller -> calibrator -> eval on v5.
NEWJOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/final; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
until [ "$(squeue -j $NEWJOB -h -o %T)" = "RUNNING" ]; do sleep 60; done
until RCLUP_WORKER_JOB=$NEWJOB timeout 60 $PY -c "from rclup.backbone import discover_worker; discover_worker()" >/dev/null 2>&1; do sleep 30; done
echo "[$(date +%H:%M:%S)] worker $NEWJOB ready"; export RCLUP_WORKER_JOB=$NEWJOB
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$NEWJOB --overlap --ntasks=1 --cpus-per-task=2 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$?"; }
C=collect/hoosier_v5_repr; CAP=260; rm -rf $R/$C $R/collect/hoosier_v5_cal
step collect $PY -m rclup.collect --dataset hoosier --partition representation --roots 200 --seeds 13 --max-commits 3 --max-questions 6 --prefixes-per-root 2 --workers 10 --variant v5 --no-latent --out $C --cap-usd $CAP
step collect_cal $PY -m rclup.collect --dataset hoosier --partition action_calibration --roots 30 --seeds 13 --max-commits 3 --max-questions 6 --prefixes-per-root 2 --workers 10 --variant v5 --no-latent --out collect/hoosier_v5_cal --cap-usd $CAP
step train_A $PY -m rclup.train --collect $C --out models/hoosier5_A_richtext --epochs 15 --no-latent
step train_C $PY -m rclup.train --phase C --init $R/models/hoosier5_A_richtext/carrier.pt --collect collect/hoosier_v5_cal --out models/hoosier5_C_richtext --no-latent
B5=normal,rambling,impatience,self_contradictory,unsupported_request; E=eval/final
step richtext $PY -m rclup.run_episodes --dataset hoosier --split test --controller rich_text_mc --variant v5 --carrier $R/models/hoosier5_A_richtext/carrier.pt --calibrator $R/models/hoosier5_C_richtext/calibrator.pt --lam 0.02 --behaviors $B5 --limit 240 --workers 10 --out $E/hoosier_richtext_mc_b5 --cap-usd $CAP
echo "[$(date +%H:%M:%S)] CHAIN C DONE"
