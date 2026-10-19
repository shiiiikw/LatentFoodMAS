#!/bin/bash
# Final table on frozen code v2; usage: final_v2.sh <job> <run1,run2,...>  where run = controller[:variant]
JOB=$1; RUNS_=$2; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/final; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=2 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$?"; }
B5=normal,rambling,impatience,self_contradictory,unsupported_request; CAP=270; E=eval/final
for spec in ${RUNS_//,/ }; do
  c=${spec%%:*}; v=${spec#*:}; [ "$v" = "$spec" ] && v=v5
  case $c in
    rich_text_mc) out=$E/hoosier_richtext_mc_b5; extra="--carrier $R/models/hoosier5_A_richtext/carrier.pt --calibrator $R/models/hoosier5_C_richtext/calibrator.pt --lam 0.02";;
    text_mas) if [ "$v" = "v5" ]; then out=$E/hoosier_text_mas_b5; elif [ "$v" = "v5nofb" ]; then out=$E/hoosier_text_mas_nofb_b5; else out=$E/hoosier_text_mas_${v}_b5; fi; extra="";;
    *) out=$E/hoosier_${c}_b5; extra="";;
  esac
  step ${c}_${v} $PY -m rclup.run_episodes --dataset hoosier --split test --controller $c --variant $v $extra --behaviors $B5 --limit 240 --workers 10 --out $out --cap-usd $CAP
done
echo "[$(date +%H:%M:%S)] FINAL_V2 $JOB DONE"
