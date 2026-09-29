# B1-to-B0 chain: operational record

Run from `/proj/jlessler/projects/tapestry-all/tapestry` on `ssh longleaf`.
Plans are complete. Do not rerun them or modify launched snapshots.

The exact manager `plan` calls for fitting are in [plan-chain.sh](plan-chain.sh).
The exact manager `plan` calls for forecast regeneration are recorded in each
`data/experiments/b1-b0-score-*/reforecast-protocol.json` under `plan_command`.
[plan_chain_scores.py](plan_chain_scores.py) constructs and prints those calls.

Historical plan invocations (completed):

```bash
bash analysis/b0-audit/plan-chain.sh 07 08
bash analysis/b0-audit/plan-chain.sh 00 01 02 03 09
.venv/bin/python analysis/b0-audit/plan_chain_scores.py
```

Historical fitting launch (completed):

```bash
for stage in 00 01 02 03 07 08 09; do
  LANES=3 sbatch --job-name="b1-b0-chain-$stage" --array=0-0 \
    --nodelist=g1803jles01 scripts/jlessler.sbatch "b1-b0-chain-$stage"
done
```

Forecast regeneration launch, after the final common-calendar pin:

```bash
bash analysis/b0-audit/launch-chain-scores.sh
```

That script contains each exact `sbatch` command and its source-fit dependency.
It regenerates forecasts from saved weights; it does not fit new weights.

Inspect fitting:

```bash
for stage in 00 01 02 03 07 08 09; do
  .venv/bin/python -m tapestry.experiment.planner status -e "b1-b0-chain-$stage"
  .venv/bin/python -m tapestry.experiment.planner rank -e "b1-b0-chain-$stage" --no-plots
done
```

Inspect the final scientifically comparable forecasts:

```bash
for stage in 00 01 02 03 03b 04 05 06 07 09; do
  .venv/bin/python -m tapestry.experiment.planner status -e "b1-b0-score-$stage"
  .venv/bin/python -m tapestry.experiment.planner rank -e "b1-b0-score-$stage" --no-plots
done
.venv/bin/python analysis/b0-audit/analyze_chain.py
```

Use `status`'s resubmission command only for an unfinished or failed task. The
final report uses the reforecasted `b1-b0-score-*` results, not the training jobs'
original evaluation streams.

Score stage08 was canceled before execution: every fitted parameter matched stage07 in all nine folds. Its duplicate forecast generation is unnecessary. The final chain moves from07 directly to09. The diagnostic fitting/equality evidence is retained.
