#!/bin/bash
# Controller-input ablations (trained on the cached 332-prefix labels; no new collection) + offline question-selection analysis.
JOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/ablation; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=2 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$?"; }
C=collect/hoosier_repr_full
step train_A_rt_nodep $PY -m rclup.train --collect $C --out models/hoosierF_A_richtext_nodep --epochs 15 --no-latent --ablate dependency
step train_C_rt_nodep $PY -m rclup.train --phase C --init $R/models/hoosierF_A_richtext_nodep/carrier.pt --collect collect/hoosier_cal_full --out models/hoosierF_C_richtext_nodep --no-latent --ablate dependency
step train_A_rt_norecv $PY -m rclup.train --collect $C --out models/hoosierF_A_richtext_norecv --epochs 15 --no-latent --ablate receiver_state
step train_C_rt_norecv $PY -m rclup.train --phase C --init $R/models/hoosierF_A_richtext_norecv/carrier.pt --collect collect/hoosier_cal_full --out models/hoosierF_C_richtext_norecv --no-latent --ablate receiver_state
CARR="richtext=models/hoosierF_A_richtext/carrier.pt,latent=models/hoosierF_A/carrier.pt,richtext_nodep=models/hoosierF_A_richtext_nodep/carrier.pt,richtext_norecv=models/hoosierF_A_richtext_norecv/carrier.pt"
ABL="richtext_nodep:dependency,richtext_norecv:receiver_state"
step analysis_val $PY -m rclup.analysis_questions --collect $C --dataset hoosier --carriers $CARR --ablate $ABL --val-only --out eval/analysis_questions_hoosier_val
step analysis_all $PY -m rclup.analysis_questions --collect $C --dataset hoosier --carriers $CARR --ablate $ABL --out eval/analysis_questions_hoosier_all
echo "[$(date +%H:%M:%S)] ABLATION DONE"
