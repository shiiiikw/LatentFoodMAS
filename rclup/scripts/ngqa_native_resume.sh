#!/bin/bash
# Wait for the fresh worker to serve, then re-run the NGQA native chain on it (collect skips finished roots).
NEWJOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1
until [ "$(squeue -j $NEWJOB -h -o %T)" = "RUNNING" ]; do sleep 60; done
echo "[$(date +%H:%M:%S)] worker $NEWJOB running; waiting for vLLM"
until RCLUP_WORKER_JOB=$NEWJOB timeout 60 $PY -c "from rclup.backbone import discover_worker; discover_worker()" >/dev/null 2>&1; do sleep 30; done
echo "[$(date +%H:%M:%S)] vLLM ready on $NEWJOB"
# let the old chain die first (old worker ends ~1.4h after 03:xx); if it is still alive, the new run simply shares the work
until ! grep -q "START collect" $R/logs/ngqa_native_chain.log || grep -q "END collect" $R/logs/ngqa_native_chain.log; do sleep 60; done
bash rclup/scripts/ngqa_native_chain.sh $NEWJOB
