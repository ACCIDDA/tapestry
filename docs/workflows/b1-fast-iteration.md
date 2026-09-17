# Fast B1 iteration

2026-09-17: the user requested a much shorter follow-up to the running decisive
experiment. Use the existing manager and shared scorer; full-budget snapshots
and the running jobs remain unchanged.

The first fast screen is `B1-fast-5ep`: A/B/C, seed42 only, epoch cap5,
patience2, training members32, validation members64, evaluation members64.
All three existing folds, target components, width64, latent16, masking .5,
scientific forecast selection, select-then-refit, known-final exclusions and
pinned retrospective inputs remain unchanged. Only the computational budget
changes. Each candidate is fitted anew under that same reduced budget.

This screen is an early-learning/regression diagnostic. Five epochs can favor
fast-learning configurations and may leave all models underfit; one seed and
64 evaluation draws have substantial fitting and Monte Carlo uncertainty.
Do not apply the decisive experiment's 5% promotion rule to this screen or pool
its scores with full-budget runs. Use its direction to guide the next edit,
then check that direction with the long experiment.

All six patron GPUs were occupied at launch preparation. The generic CPU
launcher uses 12 spare CPUs on g1803jles02, three independent configuration fits
with four threads each, through `manager run`. The initial 5–10 minute target
is an unverified timing estimate; Slurm has a 30-minute limit. Completion
notification is an afterany Slurm job, including failures or timeout.

```bash
for suite in B1-decisive-A B1-direct-finalflag B1-joint-aux025; do
  .venv/bin/python -m tapestry.models.manager plan -e B1-fast-5ep \
    --suite "$suite" --seeds 42 --epochs 5 --patience 2 \
    --members 32 --validation-members 64 --eval-members 64 \
    --retrospective --device cpu
done
sbatch --job-name=B1-fast-5ep scripts/cpu.sbatch B1-fast-5ep
.venv/bin/python -m tapestry.models.manager status -e B1-fast-5ep
.venv/bin/python -m tapestry.models.manager rank -e B1-fast-5ep
```

An even cheaper existing smoke run (`B1-decisive-smoke`, two epochs, one seed,
16 training / 32 validation / 32 evaluation members) already completed all
three candidates and folds. Its natural scores were A=1.51295, B=1.53128,
C=1.52681; those heavily undertrained fits did not establish an improvement.
