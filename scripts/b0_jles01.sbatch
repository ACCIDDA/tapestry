#!/bin/bash
#SBATCH --job-name=b0-jles01
#SBATCH --partition=jlessler
#SBATCH --qos=normal
#SBATCH --nodelist=g1803jles01
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
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
# Sizing, from job 1218975 measured on this node: a fit resides in ~1.8 GiB, not
# the ~6 GiB the heaviest configuration peaks at while training, so 10 lanes need
# only ~18 GiB of the L40's 44. Memory is not the limit here; cores are. The card
# reports 100% utilization at 24% memory, i.e. it is launch-latency-bound, so a fit
# gains nothing from a second BLAS thread -- OMP_NUM_THREADS=1 lets 10 lanes fit in
# 10 of the 56 cores (4 jobs use 40). Raise LANES further only with cores to spare.
# Walltime is 12h: a lane is now the sum of its seeds, and it holds several
# configurations in a row.
set -euo pipefail
experiment=${1:?usage: sbatch --array=0-3 scripts/b0_jles01.sbatch EXPERIMENT}
lanes=${LANES:-10}
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
