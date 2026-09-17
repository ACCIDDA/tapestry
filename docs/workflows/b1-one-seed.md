# GPU-only one-seed iteration

2026-09-17: the user cancelled the ten-seed daytime execution and its final
report, retaining the plans for an overnight run. The proposed CPU screen was
withdrawn before submission; its launcher, instructions and unused experiment
folder were removed. No CPU fits ran. The decisive-report inference default is
CUDA as well; GPU allocation failure must not trigger a CPU fallback.

Run seed42 only for all A/B/C candidates, with the full 100-epoch cap, patience30,
128 training members, 256 validation members and 2,048 evaluation members. Keep
all three folds and select-then-refit. Reuse A's verified complete seed42. B and
C run on separate L40 GPUs without the ten-seed contention. The rough 25–40
minute estimate comes from A's prior ~24-minute full seed runs, not a guarantee.

The dispatcher accepts `--seeds 42` to restrict this invocation without changing
the ten-seed plans. `manager status/rank --seeds 42` reports this subset. Omit
the seed filter when resuming the full plans overnight. Interrupted incomplete
attempts remain available for audit; the manager starts fresh attempts for
incomplete seeds. Completed seeds remain reusable.

This is a paired engineering screen, not a ten-seed conclusion. There is no
estimable across-seed uncertainty with one seed, and no ten-distribution mixture.
Use the shared natural/stress forecast score and calibration to guide iteration;
the original uncertainty and decision-rule report waits for the overnight run.


One-seed report command after the two new fits finish:

```bash
.venv/bin/python -m tapestry.models.manager decisive -e B1-onlymask-refit \
  --seeds 42 --device cuda --report-output data/experiments/B1-seed42-report
```

The same dependency report launcher takes an optional seed (`scripts/b1_decisive.sbatch 42`).
It uses its separately pinned reporting source. The report includes all three
pairwise differences, corrected stress scores, target/season/geography calibration,
C's recent mechanisms, and conditional temporal intervals. It does not manufacture
seed intervals or a ten-member-model mixture from a single fitted seed.

## Submitted jobs and exact commands

The full daytime jobs 1487036, 1487037, 1487038 and pending full report 1490284
were cancelled at the user's request. A seed42 is reused from its complete
attempt; B/C seed42 restart as attempt002. New GPU jobs: B=1491695,
C=1491696. The seed42 report is job1491906. The training source snapshot is
2f10c91; the independent report snapshot is 2b50923. Notifications remain enabled.

```bash
for e in B1-direct-finalflag B1-joint-aux025; do
  .venv/bin/python -m tapestry.models.manager plan -e "$e" \
    --suite "$e" --seeds 42 --eval-members 2048 --retrospective --device cuda
  LANES=1 GPUS=1 sbatch --job-name="$e-s42" --array=0 \
    scripts/jlessler.sbatch "$e" --seeds 42
done
for e in B1-onlymask-refit B1-direct-finalflag B1-joint-aux025; do
  .venv/bin/python -m tapestry.models.manager status -e "$e" --seeds 42
  .venv/bin/python -m tapestry.models.manager rank -e "$e" --seeds 42
done
sbatch --job-name=B1-seed42-report --dependency=afterok:1491695:1491696 \
  --kill-on-invalid-dep=yes scripts/b1_decisive.sbatch 42
```

For the overnight run, the original ten-seed plans remain intact. Once the
seed42 jobs have ended, omit the seed restriction to resume unfinished seeds:

```bash
LANES=7 GPUS=1 sbatch --job-name=B1-onlymask-refit --array=0 scripts/jlessler.sbatch B1-onlymask-refit
LANES=10 GPUS=1 sbatch --job-name=B1-direct-finalflag --array=0 scripts/jlessler.sbatch B1-direct-finalflag
LANES=5 GPUS=2 sbatch --job-name=B1-joint-aux025 --array=0-1 scripts/jlessler.sbatch B1-joint-aux025
```

Use the three newly returned job IDs in `--dependency=afterok:ID_A:ID_B:ID_C`
when submitting `scripts/b1_decisive.sbatch` without a seed argument. No overnight
restart has been scheduled automatically.
