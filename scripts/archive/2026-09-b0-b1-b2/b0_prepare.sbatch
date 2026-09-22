#!/bin/bash
#SBATCH --job-name=b0-prepare
#SBATCH --partition=jlessler
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/proj/jlessler/projects/tapestry-all/tapestry
#SBATCH --output=/proj/jlessler/projects/tapestry-all/tapestry/output/slurm/b0-prepare-%j.log

set -euo pipefail
source /usr/share/lmod/lmod/init/bash
module load r/4.5.0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

hostname
git rev-parse HEAD
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
Rscript -e 'library(scoringutils); library(purrr)'
test -s data/processed/build_b_finalized.npz
test -s data/processed/build_b_finalized.json
printf '%s  %s\n' 80aaa24750044a837e063812a1d0a22339c8b3a92513f71a23dac613eaaa29e0 data/metadata/locations.csv | sha256sum --check
while read -r hub revision; do
    test "$(git --git-dir="data/mirrors/hub_${hub}_current.git" rev-parse HEAD)" = "$revision"
done <<'HUBS'
flusight db3d8a9b8f022404affe56166c63a9d224a11861
covid df7965dd1d4832a96fe6712f9de9f000abc01f3e
rsv 53baf7db03a2429f5a3710d859319b9029df76cd
HUBS

# Frozen support on the hub's 23-quantile grid; the five-quantile support cannot score new runs.
frozen=data/evaluation/b0_hub_comparison_q23
if [[ ! -s "$frozen/manifest.json" ]]; then
    # Retain incomplete support builds for inspection.
    if [[ -e "$frozen" ]]; then
        echo "Inspect and archive incomplete $frozen before resubmitting." >&2
        exit 1
    fi
    # Bootstrap fits supply evaluation dates and locations.
    bootstrap="data/experiments/b0-support-bootstrap-${SLURM_JOB_ID}"
    .venv/bin/python -m tapestry.models.season_cv \
        --dataset data/processed/build_b_finalized.npz \
        --device cuda --epochs 1 --eval-members 8 --output "$bootstrap"
    .venv/bin/python -m tapestry.evaluation.compare \
        --run "$bootstrap" --mirrors data/mirrors \
        --cache data/evaluation/hub_cache_q23 --output "$frozen"
fi

# Optional arguments: experiment to register (default b0) and suite (default essential).
.venv/bin/python -m tapestry.models.manager plan \
    -e "${1:-b0}" --suite "${2:-essential}" --frozen "$frozen" --device cuda
