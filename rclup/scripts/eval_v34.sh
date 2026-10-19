#!/bin/bash
# HoosierHelp v3 and v4 workflows: all controllers on 240 conversations (simulated-user replies cached; resumes existing records).
JOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/eval_v34; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=1 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$?"; }
strip() { [ -f $R/$1/records.jsonl ] && $PY - "$R/$1/records.jsonl" <<'PY'
import sys, json
p=sys.argv[1]; rows=[l for l in open(p) if l.strip()]
keep=[l for l in rows if not any(k in l for k in ('RateLimitError','insufficient_quota','No valid proposals'))]
open(p,'w').write(''.join(keep)); print(p, 'kept', len(keep), 'of', len(rows))
PY
}
B5=normal,rambling,impatience,self_contradictory,unsupported_request; CAP=220
run_variant() {
  V=$1; E=$2
  for c in text_mas disagreement richtext_mc max_uncertainty fixed_order voi_det; do strip $E/hoosier_${c}_b5; done
  step ${V}_text_mas $PY -m rclup.run_episodes --dataset hoosier --split test --controller text_mas --variant $V --behaviors $B5 --limit 240 --workers 10 --out $E/hoosier_text_mas_b5 --cap-usd $CAP
  step ${V}_disagree $PY -m rclup.run_episodes --dataset hoosier --split test --controller disagreement_ask --variant $V --behaviors $B5 --limit 240 --workers 10 --out $E/hoosier_disagreement_b5 --cap-usd $CAP
  step ${V}_maxunc $PY -m rclup.run_episodes --dataset hoosier --split test --controller max_uncertainty --variant $V --behaviors $B5 --limit 240 --workers 10 --out $E/hoosier_max_uncertainty_b5 --cap-usd $CAP
  step ${V}_fixed $PY -m rclup.run_episodes --dataset hoosier --split test --controller fixed_order --variant $V --behaviors $B5 --limit 240 --workers 10 --out $E/hoosier_fixed_order_b5 --cap-usd $CAP
  step ${V}_voi $PY -m rclup.run_episodes --dataset hoosier --split test --controller voi_det --variant $V --behaviors $B5 --limit 240 --workers 10 --out $E/hoosier_voi_det_b5 --cap-usd $CAP
  step ${V}_richtext $PY -m rclup.run_episodes --dataset hoosier --split test --controller rich_text_mc --variant $V --carrier $R/models/hoosierF_A_richtext/carrier.pt --calibrator $R/models/hoosierF_C_richtext/calibrator.pt --lam 0.02 --behaviors $B5 --limit 240 --workers 10 --out $E/hoosier_richtext_mc_b5 --cap-usd $CAP
  step ${V}_report $PY -m rclup.report --dataset hoosier --reference $E/hoosier_text_mas_b5 --runs $E/hoosier_text_mas_b5,$E/hoosier_disagreement_b5,$E/hoosier_max_uncertainty_b5,$E/hoosier_fixed_order_b5,$E/hoosier_voi_det_b5,$E/hoosier_richtext_mc_b5 --out $E/report_hoosier_b5
  echo "[$(date +%H:%M:%S)] VARIANT $V DONE"
}
run_variant v3 eval/full3
run_variant v5 eval/full5
run_variant v4 eval/full4
echo "[$(date +%H:%M:%S)] EVAL V34 DONE"
