#!/bin/bash
#SBATCH --job-name=b0-jles01
#SBATCH --partition=jlessler
#SBATCH --qos=normal
#SBATCH --nodelist=g1803jles01
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=output/slurm/b0-jles01-%A_%a.log

# Private node g1803jles01: 4x L40 (46068 MiB = 44 GiB usable), 56 cores, 500GB RAM.
# The partition is hidden: `sinfo -a -n g1803jles01` shows it, `scontrol show
# partition jlessler` reports "not found". Do not take that as missing access.
#
# ONE ARRAY TASK = ONE GPU, holding LANES configurations at once. A lane owns a
# configuration and fits its three seeds in sequence, so seeds never overlap while
# LANES different configurations share the card:
#
#   GPU 0   lane A: task 6  s42 -> s43 -> s44
#           lane B: task 10 s42 -> s43 -> s44     (in parallel with A)
#           ...LANES lanes
#
# This is why concurrency cannot come from the array throttle here: Slurm has no
# `shard`/`mps` GRES (GresTypes=gpu,myJob, SelectTypeParameters=CR_CPU_MEMORY), so
# a card is allocated whole and a second job on a busy GPU pends on `Resources`.
# One job per card, many lanes inside it, is the only way to pack these cards.
#
#   sbatch --array=0-3 scripts/b0_jles01.sbatch b0-us-cross-3
#
# `manager status -e <experiment>` prints the remaining task numbers; put them in
# TASKS (default: the 6-59 that remain in b0-us-cross-3).
#
# Sizing, measured on this node during job 1218975. Per-fit GPU memory is NOT
# uniform: light `raw` configurations sit at ~1.8 GiB, but the conv family ranges
# over 2.8, 4.0 and up to 6.9 GiB. Sampling only the light ones suggests 10 lanes
# fit in ~18 GiB; that is wrong. A card holding several conv lanes is the binding
# case -- slot 2 measured 19.2 GiB with a single conv fit resident alongside five
# light ones, and 8 conv lanes would want ~55 GiB against the L40's 44.
#
# 6 lanes is the proven setting and already drives the card to 100% utilization,
# so more lanes buy little. Raise LANES only after checking the conv share of the
# remaining work (`manager status` + grep conv); 7-8 is the realistic ceiling on a
# mixed card, and only 10 if the remaining work is conv-free.
# Each fit pins one torch thread, so 6 lanes want 6 of the 56 cores; 4 jobs use 24.
# Walltime is 12h: a lane is now the sum of its seeds, and it holds several
# configurations in a row.
set -euo pipefail
experiment=${1:?usage: sbatch --array=0-3 scripts/b0_jles01.sbatch EXPERIMENT}
lanes=${LANES:-6}
gpus=${GPUS:-4}
# Stride over the task list so heavy conv configurations spread across cards
# instead of clumping onto whichever slice happens to hold them.
# Unquoted so it word-splits on spaces (a TASKS override) and on the newlines
# `seq` emits; `read -a <<<` would take only the first line and silently drop the rest.
# shellcheck disable=SC2206
all=(${TASKS:-$(seq 6 59)})
mine=()
for ((i = SLURM_ARRAY_TASK_ID; i < ${#all[@]}; i += gpus)); do mine+=("${all[i]}"); done
# One thread per fit: these fits are launch-latency-bound, so the second thread
# bought nothing and 10 lanes x 2 threads x 4 jobs (80) would exceed the 56 cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# Fragmentation guard: several lanes share one card and allocate in bursts.
export PYTORCH_ALLOC_CONF=expandable_segments:True
hostname
echo "Experiment ${experiment}, GPU slot ${SLURM_ARRAY_TASK_ID}, ${lanes} lanes, tasks: ${mine[*]}"
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GiB")'
.venv/bin/python -m tapestry.models.manager run -e "$experiment" --task "${mine[@]}" \
    --device cuda --keep-going --fit-workers "$lanes" "${@:2}"
