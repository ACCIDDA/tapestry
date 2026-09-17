# GPU-only one-seed iteration

2026-09-17: the user cancelled the ten-seed daytime execution and its final
report, retaining the plans for an overnight run. The proposed CPU screen was
withdrawn before submission; its launcher, instructions and unused experiment
folder were removed. No CPU fits ran.

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
