#!/bin/bash
# Probe sweep for σ-QRT G2 v2 critic capacity sweep.
#
# For each of 6 lanes × 3 mid-training steps {5000, 7000, 9000}, run
# probe_critic_sensitivity to measure Q sensitivity. Output JSON layout:
#   results/v7_g2_v2_capacity_probes/${LANE}_step${STEP}.json
#
# Resume-safe: skips existing JSON.
#
# Gates on ckpt existence — if a step ckpt isn't saved yet, skip it (the
# script will exit with a partial-done message and can be re-run later).
#
# Run when lanes have crossed step 5000+. CPU-mode probe (~5 min each),
# so 18 probes sequential ~ 90 min total. Acceptable.

set -uo pipefail
cd ~/data/jskang/sigma-qrt/RLinf
source ~/data/jskang/sigma/.venv/bin/activate
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=''

RUN=runs/g2_v2_critic_capacity_20260515_101958
BUF=data/sigma_qrt/libero_long/transitions_B2500.pkl
OUT=results/v7_g2_v2_capacity_probes
LOG=logs/probe_v7_g2_v2_capacity
mkdir -p $OUT $LOG

probe() {
    local lane=$1; local step=$2
    local ckpt=$RUN/$lane/ckpt_step${step}.pt
    local outjson=$OUT/${lane}_step${step}.json
    local logf=$LOG/${lane}_step${step}.log
    if [ -f "$outjson" ]; then
        echo "[$(date +%H:%M:%S)] SKIP $lane step$step (already done)"
        return 0
    fi
    if [ ! -f "$ckpt" ]; then
        echo "[$(date +%H:%M:%S)] WAIT $lane step$step (ckpt not saved yet)"
        return 0
    fi
    echo "[$(date +%H:%M:%S)] RUN  $lane step$step"
    timeout 1500 python -X utf8 experiments/probe_critic_sensitivity.py \
        --ckpt $ckpt \
        --buffer_path $BUF \
        --num_states 100 \
        --num_perturb 20 \
        --perturb_std 0.1 \
        --device cpu \
        --output $outjson \
        > $logf 2>&1
    local rc=$?
    echo "[$(date +%H:%M:%S)] DONE $lane step$step rc=$rc"
    return 0
}

# 6 lanes × 3 steps = 18 probes
for lane in cap_small_seed1 cap_small_seed2 cap_med_seed1 cap_med_seed2 cap_large_seed1 cap_large_seed2; do
    for step in 5000 7000 9000; do
        probe $lane $step
    done
done

# aggregate
python - << 'PYEOF' 2>&1 | tee $OUT/SUMMARY.log
import json
from pathlib import Path
import statistics as st
out = Path('results/v7_g2_v2_capacity_probes')
rows = []
for f in sorted(out.glob('cap_*_step*.json')):
    try:
        d = json.loads(f.read_text())
        name = f.stem  # cap_small_seed1_step5000
        parts = name.split('_')
        # cap, small/med/large, seedX, stepN
        cap = parts[1]
        seed = parts[2].replace('seed','')
        step = parts[3].replace('step','')
        rows.append({
            'cap': cap, 'seed': int(seed), 'step': int(step),
            'q_std_mean': d.get('q_std_mean'),
            'q_std_max': d.get('q_std_max'),
            'q_range_mean': d.get('q_range_mean'),
            'q_mean_overall': d.get('q_mean_overall'),
            'verdict': d.get('verdict'),
        })
    except Exception as e:
        print(f'ERR parsing {f}: {e}')

print('cap     seed step    q_std_mean   q_range_mean q_mean      verdict')
for r in rows:
    print('{c:<7} {s:<4} {st:<6} {qs:>10}   {qr:>10}   {qm:>8}   {v}'.format(
        c=r['cap'], s=r['seed'], st=r['step'],
        qs=round(r['q_std_mean'], 4) if r['q_std_mean'] is not None else 'NA',
        qr=round(r['q_range_mean'], 4) if r['q_range_mean'] is not None else 'NA',
        qm=round(r['q_mean_overall'], 4) if r['q_mean_overall'] is not None else 'NA',
        v=r.get('verdict', 'NA')))

# Per-(cap, step) aggregate
agg = {}
for r in rows:
    agg.setdefault((r['cap'], r['step']), []).append(r)
print()
print('=== aggregate by (cap, step): q_std_mean mean ± std ===')
for k in sorted(agg.keys()):
    g = agg[k]
    qs = [r['q_std_mean'] for r in g if r['q_std_mean'] is not None]
    if qs:
        mean = sum(qs)/len(qs)
        std = st.stdev(qs) if len(qs) >= 2 else 0.0
        print(f'  cap={k[0]:<6} step={k[1]:<5}  q_std_mean = {mean:.4f} ± {std:.4f}  (n={len(qs)})')
    else:
        print(f'  cap={k[0]:<6} step={k[1]:<5}  no data')
PYEOF

echo "[$(date +%H:%M:%S)] probe sweep done"
