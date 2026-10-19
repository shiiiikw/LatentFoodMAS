#!/bin/bash
# Budget-pressure labels: a second v6 collection at a 6-turn cap, merged with the 8-turn labels; retrain; eval at caps 6/7/8.
JOB=$1; PY=/jet/home/kshi4/.conda/envs/ugrpo/bin/python; R=/ocean/projects/cis260058p/kshi4/rclup_runs; LOGS=$R/logs/v6; mkdir -p $LOGS
cd /jet/home/kshi4/LatentFoodMAS; export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RCLUP_WORKER_JOB=$JOB
until grep -q "END collect_cal" $LOGS/chain_labels.log 2>/dev/null; do sleep 60; done
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=2 --mem=0 --export=ALL "$@" > $LOGS/$name.log 2>&1; echo "[$(date +%H:%M:%S)] END $name rc=$? $(grep -E '"end_to_end_match"' $LOGS/$name.log | tail -1)"; }
C6=collect/hoosier_v6_repr_t6; CAP=260; rm -rf $R/$C6
step collect_t6 $PY -m rclup.collect --dataset hoosier --partition representation --roots 200 --seeds 13 --max-commits 3 --max-questions 6 --prefixes-per-root 2 --workers 10 --variant v6 --max-turns 6 --no-latent --out $C6 --cap-usd $CAP
MIX=$R/collect/hoosier_v6_mix; rm -rf $MIX; mkdir -p $MIX/prefixes
for f in $R/collect/hoosier_v6_repr/prefixes/*.pt; do ln -s $f $MIX/prefixes/t8__$(basename $f); done
for f in $R/$C6/prefixes/*.pt; do ln -s $f $MIX/prefixes/t6__$(basename $f); done
step train_A_mix $PY -m rclup.train --collect collect/hoosier_v6_mix --out models/hoosier6_A_richtext_mix --epochs 15 --no-latent
step train_C_mix $PY -m rclup.train --phase C --init $R/models/hoosier6_A_richtext_mix/carrier.pt --collect collect/hoosier_v6_cal --out models/hoosier6_C_richtext_mix --no-latent
B5=normal,rambling,impatience,self_contradictory,unsupported_request; E=eval/v6
RT="--controller rich_text_mc --carrier $R/models/hoosier6_A_richtext_mix/carrier.pt --calibrator $R/models/hoosier6_C_richtext_mix/calibrator.pt --lam 0.02"
step richtext_mix $PY -m rclup.run_episodes --dataset hoosier --split test $RT --variant v6 --behaviors $B5 --limit 240 --workers 10 --out $E/hoosier_richtext_mix_b5 --cap-usd $CAP
for T in 6 7; do
  step richtext_mix_t$T $PY -m rclup.run_episodes --dataset hoosier --split test $RT --variant v6 --max-turns $T --behaviors $B5 --limit 240 --workers 10 --out $E/turns$T/hoosier_richtext_mix_b5 --cap-usd $CAP
done
echo "[$(date +%H:%M:%S)] V6 MIX CHAIN DONE"
