"""Summarize a completed B0 season run without refitting or resampling."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--report', type=Path, default=Path('docs/results/b0-season-cv.md'))
    args = parser.parse_args()
    manifest = json.loads((args.run / 'manifest.json').read_text())
    scores = list(csv.DictReader((args.run / 'scores.csv').open()))
    primary = [r for r in scores if r['channel'] == 'nhsn_flu_admissions' and r['geography'] == 'states_dc']
    args.report.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    colors = ['#2870a0', '#bb591e', '#278269']
    for i, fold in enumerate(manifest['folds']):
        label = fold['eval_season']
        axes[0, 0].plot(np.arange(1, len(fold['history']) + 1), fold['history'], color=colors[i], label=label)
        rows = [r for r in primary if r['eval_season'] == label and r['horizon'] != 'all']
        axes[0, 1].plot([int(r['horizon']) for r in rows], [float(r['wis']) for r in rows], 'o-', color=colors[i])
        axes[0, 2].plot([int(r['horizon']) for r in rows], [100 * float(r['coverage80']) for r in rows], 'o-', color=colors[i])
        with np.load(args.run / f'eval_{label}' / 'forecasts.npz', allow_pickle=False) as data:
            loc = data['locations'].tolist().index('NC')
            dates = data['target_dates'][:, 3].astype('datetime64[D]')
            valid = data['mask'][:, 3, 0, loc]
            q = data['quantiles'][:, :, 3, 0, loc][:, valid]
            ax = axes[1, i]
            ax.fill_between(dates[valid], q[3], q[-4], color=colors[i], alpha=.20, label='80% interval')
            ax.plot(dates[valid], q[11], color=colors[i], label='4-week forecast median')
            ax.plot(dates[valid], data['truth'][:, 3, 0, loc][valid], color='#252525', linewidth=1, label='Finalized observation')
            ax.set(title=f'NC influenza admissions · {label}', ylabel='Admissions', xlabel='Target week')
            ax.tick_params(axis='x', labelrotation=25)
            ax.set_ylim(bottom=0)
    axes[0, 0].set(title='Training loss · fixed 50 epochs', xlabel='Epoch', ylabel='Weighted scaled fair CRPS')
    axes[0, 0].legend(title='Held-out season', fontsize=8)
    axes[0, 1].set(title='State/DC influenza WIS', xlabel='Weeks ahead', ylabel='WIS (lower is better)', xticks=[1, 2, 3, 4])
    axes[0, 2].set(title='State/DC influenza interval coverage', xlabel='Weeks ahead', ylabel='80% interval coverage (%)', xticks=[1, 2, 3, 4], ylim=(0, 100))
    axes[0, 2].axhline(80, color='gray', linestyle='--', linewidth=1)
    axes[1, 0].legend(fontsize=7)
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.15)
    figure = args.report.with_suffix('.png')
    fig.savefig(figure, dpi=150)
    plt.close(fig)
    table = ['| Train seasons | Evaluate season | Origins | WIS | MAE | 80% coverage | 95% coverage | WIS / persistence* |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for fold in manifest['folds']:
        row = next(r for r in primary if r['eval_season'] == fold['eval_season'] and r['horizon'] == 'all')
        table.append(f"| {' + '.join(fold['train_seasons'])} | {fold['eval_season']} | {fold['eval_origins']} | {float(row['wis']):.2f} | {float(row['mae']):.2f} | {100*float(row['coverage80']):.1f}% | {100*float(row['coverage95']):.1f}% | {float(row['wis_ratio_to_persistence']):.3f} |")
    timings = '\n'.join(f"- Evaluate {f['eval_season']}: {f['train_episodes']} training windows; fit {f['train_seconds']:.1f}s, forecast/save/score {f['evaluation_seconds']:.1f}s; loss {f['history'][0]:.4f} → {f['history'][-1]:.4f}." for f in manifest['folds'])
    text = f'''# Build B0: three-season cross-validation

Completed locally on the Apple M2 Max (32 GiB), using two CPU threads and
PyTorch {manifest['torch_version']}. All three training/evaluation combinations finished in
**{manifest['elapsed_seconds']:.1f} seconds**. Longleaf was unnecessary for this model;
[connection and rsync instructions](../development-longleaf.md) are documented.

## Results

Influenza admissions, 50 states + DC, pooled over four horizons and all eligible
weekly origins in each respiratory season. US national scores are separate in the
CSV. These are year-round season scores, not official FluSight challenge scores.

{chr(10).join(table)}

*Persistence is a **deterministic last-observed-value distribution**, a simple
sanity-check baseline, not a strong probabilistic forecasting comparator. Ratios
use only common valid tasks; 10 of 10,030 model-scored cells in season 2024–25 lack
a baseline anchor. The table's WIS/MAE/coverage use all available model labels.
The model's WIS beats this baseline in every fold. Its 80% and 95% intervals
under-cover in every fold. This is an uncalibrated pilot, not evidence of
competitiveness with GBQR, the FluSight ensemble, or other operational models.

![Training, horizon scores, and rolling four-week NC forecasts]({figure.name})

NC is a fixed illustrative location, chosen without searching for a good or bad
example. Each plotted forecast was issued four weeks before its target date;
these are rolling marginal forecasts, not one full-season sampled trajectory.
The top panels summarize all states/DC. Inspect the saved forecasts for any
other location, channel, or horizon.

## Exact experiment

- Season 1: 2023–24 (available data September 2, 2023–July 27, 2024).
- Season 2: 2024–25 (August 3, 2024–July 26, 2025).
- Season 3: 2025–26 (August 2, 2025–August 1, 2026).
- Season grouping is CDC epiweek 31–30, including the 53-week calendar effect.
  The four available weeks of 2026–27 are excluded entirely from this experiment.
- Each sample moves forward **one week**, with an **8-week context** and targets
  **1, 2, 3, and 4 weeks after context end**. Whole location panels share an origin.
- The held-out season is removed from every training context, every training
  target, and scaler fitting. Its absent training-context weeks stay masked,
  rather than being filled from the held-out data. Training origins belong only
  to the two fit seasons. Evaluation may condition on all already observed past
  context, including earlier weeks within the evaluation season.
- Labels outside the evaluation season are masked. Origins with no evaluable
  future labels are omitted. Final origins can have fewer than four scored
  horizons; all four forecasts are still generated. Leading contexts can be
  padded/masked at the September 2023 data boundary or a training-season gap.
- Fixed B0 settings: width 64, 16-dimensional global latent, 22,785 parameters,
  50 epochs, batch size 8, Adam learning rate 0.001, eight training members,
  seed 42. Per-channel training-only Q95 scales and loss weights
  `[1,.1,.1,.1,.1,.1]` are unchanged. No held-out tuning or early stopping.
- 2,048 independent draws per evaluation origin; quantiles on the 23-level grid.
  Admission quantiles are rounded half-up; ED proportions remain continuous.
  WIS, median MAE, interval coverage/width, and persistence comparisons are saved
  separately for each channel, horizon, and geography group.
- 100 complete sample members per origin are retained alongside quantiles
  computed from all 2,048 draws. Member identity is preserved over all locations,
  channels and horizons within that origin. Identity does not link separate origins.

Only **train seasons 1+2 → evaluate season 3** is chronological. The other two
folds intentionally train on later seasons and are retrospective leave-one-season-out
experiments. All folds use finalized snapshots, so even the chronological fold is
not a reconstruction of real-time data availability. Season 1 had already been
used for the skeleton's smoke test. All three seasons have now been examined;
none should subsequently be called an untouched final test set. No new settings
were selected from these scores.

## Timing and artifacts

{timings}

Run directory: `{args.run}` (relative to the repository root).
It contains `manifest.json`, pooled `scores.csv`, and one `eval_SEASON/` folder
per fold with `model.pt`, `training.json`, `forecasts.npz`, and `scores.csv`.
The manifest records source/code hashes, exact origins, fitted scales, seeds,
configuration and elapsed time. Raw forecasts are saved before scoring.

```bash
PYTHONPATH=src .venv/bin/python -m influpaintx.models.season_cv \\
  --device cpu --epochs 50 --eval-members 2048 \\
  --output data/experiments/b0_season_cv_repeat
PYTHONPATH=src .venv/bin/python experiments/b0/report_b0_cv.py \\
  data/experiments/b0_season_cv_repeat
```

To inspect the chronological fold without resampling:

```python
import numpy as np
p = np.load('data/experiments/b0_season_cv_20260913/eval_2025-2026/forecasts.npz',
            allow_pickle=False)
print(p['quantiles'].shape)  # (23, 52, 4, 6, 52)
print(p['samples'].shape)    # (100, 52, 4, 6, 52)
# Quantile/member × origin × horizon × channel × location.
print(p['context_end'], p['target_dates'], p['locations'], p['channels'])
```

Checks cover held-out-data perturbation invariance for training inputs/labels/scales,
weekly stride, horizon/season mask boundaries, WIS against an independent pinball
calculation, fair CRPS, masking, and stochastic gradients. Saved outputs are also
checked for finite values, monotone quantiles, valid units, and fold label boundaries.
No calibration correction or model architecture change was made after evaluation.
'''
    args.report.write_text(text)
    shutil.copyfile(args.report, args.run / 'report.md')
    shutil.copyfile(figure, args.run / figure.name)
    print(args.report)


if __name__ == '__main__':
    main()
