# Mac and Longleaf execution

For the 22,785-parameter Build B0, an Apple M2 Max with 32 GiB RAM is sufficient.
A synchronized warm-up/timing check gives about 0.027 seconds per training step
on CPU (two threads), versus 0.030 on MPS, at batch size 8, eight stochastic
training members, eight history weeks, and 52 locations. The three-season
experiment therefore runs locally on CPU, with no transfer or cluster job.
MPS offers no advantage for this small configuration. Larger B1 models or larger
searches may justify a Longleaf GPU.

## Connect to the patron node

The UNC-IDD patron allocation uses partition **`jlessler`** (without a hyphen).

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

Authentication/MFA follows the account's usual SSH setup. `~/Tapestry` is the
suggested remote checkout directory.

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
and results. Raw data are unnecessary for fitting the already built tensor.

Use a CUDA-enabled PyTorch environment on Longleaf, for example
`/nas/longleaf/home/chadi/.conda/envs/diffusion_torch6/bin/python`; confirm that
environment exists before using it. On the allocated node:

```bash
/nas/longleaf/home/chadi/.conda/envs/diffusion_torch6/bin/python -c \
  'import numpy, torch; print(torch.__version__, torch.cuda.is_available())'
PYTHONPATH=src /nas/longleaf/home/chadi/.conda/envs/diffusion_torch6/bin/python \
  -m tapestry.models.season_cv --device cuda \
  --output data/experiments/b0_season_cv_longleaf
```

NumPy and PyTorch are sufficient to train/query this prebuilt panel through
`PYTHONPATH=src`. Acquisition is not run on the cluster. If that environment is
missing or incompatible, create/select a current CUDA PyTorch environment before
running; do not assume that a Mac environment can be copied.

Retrieve results from the Mac:

```bash
rsync -av chadi@longleaf.unc.edu:~/Tapestry/data/experiments/b0_season_cv_longleaf/ \
  /Users/chadi/Research/Tapestry/data/experiments/b0_season_cv_longleaf/
```

## Run locally

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
