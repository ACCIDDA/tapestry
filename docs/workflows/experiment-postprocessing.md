# Postprocessing a finished experiment

What to do once the fits are done: confirm the experiment really finished, find
the artifacts a given run actually produced, rank it, and turn the ranking into
figures and a results page.

All commands run from the repository root. Substitute your experiment name for
`B0.1`; the worked example is the [B0.1 crosses](../results/b0-1-crosses/index.md).

For **B1**, these are the same commands running the same code. A completed
attempt contains a `b1/` directory instead of `cv/`, holding one
`eval_<season>/` per fold, `totals.csv` in B0's schema, and additionally
`nowcast-totals.csv` for offsets -2/-1. `rank` writes B0's usual
`configuration_ranking.csv` plus a `nowcast/` subfolder scored against
preliminary-value persistence. See the
[B1 output definitions](../design/b1.md#one-manager-one-scorer). The B0
publishing/calibration commands below do not yet accept B1's artifact layout.

## 1. Confirm it finished

Never trust a stale `runs.csv` or the scheduler's own bookkeeping. `status`
rescans the attempt folders on disk and rewrites `runs.csv`:

```bash
.venv/bin/python -m tapestry.models.manager status -e B0.1
```

It prints one line per scenario × seed and a summary object last:

```text
complete	171	B0.1/independent__…__epochs_300	s42	<scenario>/s42/attempt-005
{"complete": 516}
```

A single `{"complete": N}` with `N` equal to configurations × seeds means the
experiment is done. Anything else lists the pending Slurm array tasks and the
command to resubmit them. To see only what is not finished:

```bash
.venv/bin/python -m tapestry.models.manager status -e B0.1 | grep -v '^complete'
```

!!! warning "`running` may mean killed"
    An attempt is marked `running` until its `run.json` is finalized, so a job
    killed by the scheduler or a node failure keeps that status forever. Check
    `squeue -u $USER` before concluding a task is still alive, and before
    resubmitting it.

## 2. Find a run's attempts

This is the step that most often misleads. A scenario/seed directory holds **one
folder per attempt**, and failed or interrupted attempts are preserved rather
than overwritten:

```text
data/experiments/B0.1/<scenario>/s42/
  attempt-001/   interrupted by an earlier launch — kept for provenance
  attempt-002/
  attempt-005/   the one that counts
```

`run` never reuses an attempt folder: it skips the seed if a complete attempt
exists, otherwise it starts the next number. So **a run is not incomplete just
because `attempt-001` has no `cv/totals.csv`** — a later attempt usually has it.
Resolution follows `seed_state`: the *latest complete* attempt wins, falling back
to the latest attempt if none is complete. A seed counts as complete only when
its `run.json` says `complete` *and* `cv/` holds `manifest.json`, `scores.csv`,
`totals.csv`, and all three folds' `model.pt`, `forecasts.npz`, `training.json`
and `scores.csv`.

Let `runs.csv` do the resolution instead of globbing the tree yourself. After
`status`, it has an `attempt` column with the winning path and an `attempts`
column with how many were made:

```bash
.venv/bin/python -c "
import pandas as pd
r = pd.read_csv('data/experiments/B0.1/runs.csv')
print(r.status.value_counts().to_dict())
print(r[r.attempts > 1][['task', 'name', 'seed', 'attempts', 'attempt']].to_string(index=False))
"
```

Counting only `attempt-001` on disk reports 63 of B0.1's 516 runs as missing;
all 63 finished under `attempt-002` … `attempt-005`.

### Listing tasks and configurations

| Question | Where to look |
|---|---|
| What tasks exist, and their names/scenarios/seeds? | `jobs.csv` — one row per Slurm array task |
| What is each scenario × seed's status and winning attempt? | `runs.csv` — rebuilt by `status` |
| What settings was the experiment planned with? | `experiment.json` |
| What did one attempt actually run? | `<attempt>/run.json` — command, config, git commit/dirty, host, Slurm IDs, times |
| Why did an attempt stop? | `<attempt>/run.json` `error`/`status`, then `<attempt>/run.log` |

```bash
# Task number → readable name (one array task per row).
column -s, -t data/experiments/B0.1/jobs.csv | head

# What a suite would contain, without writing anything.
.venv/bin/python -m tapestry.models.manager list --suite B0.1
```

Task numbers in `jobs.csv` are stable: `plan` only appends, so an existing task
never changes number and a Slurm array index always means the same scenario.

## 3. Rank

`rank` reads each winning attempt's `cv/totals.csv` and writes a
`ranking-<hash>/` folder:

```bash
.venv/bin/python -m tapestry.models.manager rank -e B0.1
```

It prints the destination, which you pass to the plotting script. The hash
covers the scoring version and the exact run set, so partial and full rankings
can coexist and never mix. `rank` **refuses an incomplete experiment** unless you
pass `--allow-incomplete`, which ranks only the complete runs — use it to preview
a suite still in flight, not to publish results.

Two things it will complain about, both deliberate:

- *"totals.csv lacks locations"* — old runs scored before per-location totals.
  Regenerate from the saved forecasts, no refit required:

  ```bash
  .venv/bin/python -m tapestry.evaluation.totals score \
      --run 'data/experiments/B0.1/<scenario>/s42/attempt-001/cv' \
      --frozen data/evaluation/b0_hub_comparison_q23
  ```

- *"Runs have different frozen target/season/location/horizon support"* — the
  runs were scored against different frozen tasks and are not comparable.
  Rescore them all on one frozen folder.

A warning that runs come from several commit/dirty states is informational:
experiments are not locked to a code version, and every attempt records its own
provenance. Read it before publishing, since it tells you whether the suite is
internally consistent.

See [Ranking](experiment-manager.md#ranking) for what each output column means.

## 4. Figures and a results page

`scripts/plot_b01_crosses.py` reads the ranking artifacts plus each run's saved
`forecasts.npz` and renders every figure the results page embeds — no refit and
no EpiBench run:

```bash
.venv/bin/python scripts/plot_b01_crosses.py -e B0.1 -r ranking-509b07b0d243
```

It writes to `docs/results/b0-1-crosses/figures/` by default (`--output` to
change) and takes `--skip-fans`, which skips the slow step while you are
iterating on the other panels. Fan plots come from `forecasts.npz` through
`tapestry.evaluation.hubs.export_b0` against the frozen truth.

Copy the script for a new experiment rather than generalizing it: the figures
worth drawing depend on what the experiment varied. Keep `--output` pointing at
the page's own `figures/` folder so MkDocs copies the images with the page.

Finally, add the page to the `nav` in `mkdocs.yml` — a file in `docs/` with no
`nav` entry builds and is reachable by URL but appears in no menu — and check it:

```bash
.venv/bin/python -m mkdocs build --strict
```

`--strict` turns broken internal links and missing images into errors, and the
build lists any page not included in `nav`.

## Optional: full EpiBench comparison

`rank` is enough for a leaderboard. For a shortlist that needs the complete
scoring pipeline, diagnostic plots and fans:

```bash
.venv/bin/python -m tapestry.models.manager compare -e B0.1
.venv/bin/python scripts/publish_evaluation_docs.py \
    --comparison data/experiments/B0.1/comparison-<hash>
```

`compare` suits tens of runs, not a full sweep — see
[Comparison](experiment-manager.md#comparison).

## Checklist

1. `status` → expect one `{"complete": N}` line; investigate anything else.
2. Check `runs.csv` `attempts`/`attempt` rather than assuming `attempt-001`.
3. `rank` → note the `ranking-<hash>` folder it prints.
4. Plot with that hash, `--skip-fans` while iterating.
5. Write the page, add it to `nav`, `mkdocs build --strict`.
