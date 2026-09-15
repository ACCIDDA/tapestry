# Mac and Longleaf execution

For the current 22,785-parameter Build B0, the local Apple M2 Max with 32 GiB RAM
is sufficient. A short synchronized warm-up/timing check measured about 0.027
seconds per training step on CPU (two threads), versus 0.030 on MPS. The three
season experiment therefore runs locally on CPU, with no transfer or cluster job.
These timings are specific to batch size 8, eight stochastic training members,
eight history weeks, and 52 locations.

The initial MPS trial hit an Apple rank-5 sorting assertion in fair CRPS. Flattening
the independent task axes before sorting fixed that limitation without changing
the score; both CPU and MPS completed the repeated timing check afterward.
MPS offers no measured advantage for this small configuration. Larger B1 models
or larger searches may justify a Longleaf GPU.

## Connect to the patron node

The user supplied the hostname and node below. The neighboring InfluPaint repo's
`docs/getting-started/cluster.md` independently records partition **`jlessler`**
(without a hyphen) for the UNC-IDD patron allocation. The node name and its current
availability have not been checked remotely in this run.

From the Mac:

```bash
ssh chadi@longleaf.unc.edu
```

On the login node, request the GPU allocation:

```bash
srun -p jlessler --gres=gpu:1 --cpus-per-task=4 --mem=64G \
  --time=10:00:00 --nodelist="g1803jles02" --pty /bin/zsh
```

Wait for Slurm allocation before starting training. On the allocated node:

```bash
hostname
nvidia-smi
cd ~/Tapestry
```

These commands are documented for later use; no SSH session or allocation was
started for the local experiment. Authentication/MFA follows the account's usual
SSH setup. The remote checkout directory `~/Tapestry` is an explicit proposed
location, not an observed existing directory.

## Transfer just the code and frozen tensor

From the Mac, prepare the destination and transfer only the necessary files:

```bash
ssh chadi@longleaf.unc.edu 'mkdir -p ~/Tapestry/data/processed'
rsync -av /Users/chadi/Research/Tapestry/src \
  /Users/chadi/Research/Tapestry/pyproject.toml \
  /Users/chadi/Research/Tapestry/README.md \
  chadi@longleaf.unc.edu:~/Tapestry/
rsync -av /Users/chadi/Research/Tapestry/data/processed/build_b_finalized.npz \
  /Users/chadi/Research/Tapestry/data/processed/build_b_finalized.json \
  chadi@longleaf.unc.edu:~/Tapestry/data/processed/
```

This file list omits the local virtual environment, credentials, raw archives,
and older results. Raw data are unnecessary for fitting the already built tensor.

Use a CUDA-enabled PyTorch environment on Longleaf. The neighboring InfluPaint
job launcher records `/nas/longleaf/home/chadi/.conda/envs/diffusion_torch6/bin/python`;
confirm that environment still exists before using it. For example, on the
allocated node:

```bash
/nas/longleaf/home/chadi/.conda/envs/diffusion_torch6/bin/python -c \
  'import numpy, torch; print(torch.__version__, torch.cuda.is_available())'
PYTHONPATH=src /nas/longleaf/home/chadi/.conda/envs/diffusion_torch6/bin/python \
  -m tapestry.models.season_cv --device cuda \
  --output data/experiments/b0_season_cv_longleaf
```

NumPy and PyTorch are sufficient to train/query this prebuilt panel through
`PYTHONPATH=src`. Acquisition is not run on the cluster. If that recorded
environment is missing or incompatible, create/select a current CUDA PyTorch
environment before running; do not assume that a Mac environment can be copied.

Retrieve results from the Mac:

```bash
rsync -av chadi@longleaf.unc.edu:~/Tapestry/data/experiments/b0_season_cv_longleaf/ \
  /Users/chadi/Research/Tapestry/data/experiments/b0_season_cv_longleaf/
```

## Repeat the local experiment

From `/Users/chadi/Research/Tapestry`:

```bash
PYTHONPATH=src .venv/bin/python -m tapestry.models.season_cv \
  --device cpu --epochs 50 --eval-members 2048 \
  --output data/experiments/b0_season_cv_repeat
```

Use a new output directory; the runner refuses to overwrite a previous run.
It records three checkpoints, per-origin forecast quantiles and 100 whole sample
members, truth/masks, per-channel/horizon scores, training histories, code/data
hashes, and timing. The fixed seed is 42. See the season experiment report for
fold definitions and interpretation.
