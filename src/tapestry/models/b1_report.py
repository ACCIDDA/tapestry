"""Numerical B1 audits and forward-comparison graphs; no implicit Hub matching."""
from datetime import date
import json
from pathlib import Path

import numpy as np
import torch

from tapestry.model_data.finalized import CHANNELS, season
from .b0 import fair_crps_cells
from .b1 import MASK_SCENARIOS
from .quantiles import LEVELS


def evaluate(models, episodes, ds, args, variant, seed, root, scenario_string=None):
    import pandas as pd
    from .b1_run import sample, task_weights
    from tapestry.evaluation.totals import quantile_scores
    rows = []
    direct = models[0].config['direct']
    hs = slice(2, 6) if direct else slice(None)
    weights = task_weights(episodes, range(6), direct)
    if not direct:
        weights *= 2  # Rank forecasting and nowcasting separately, each sums to one.
    scales = models[0].scale.detach().cpu().numpy()
    for scenario in MASK_SCENARIOS:
        dropouts, quantiles = [], []
        # One episode at a time bounds memory; seed is common across model variants.
        for i, episode in enumerate(episodes):
            samples, d = sample(models, [episode], members=args.evaluation_members,
                seed=seed + i * 101, device=args.device, scenario=scenario)
            dropouts.append(d[0])
            truth = episode['Y'][hs, :, 0]
            valid = episode['Y'][hs, :, 1].astype(bool)
            crps = fair_crps_cells(torch.tensor(samples), torch.tensor(truth[None]),
                                   torch.tensor(valid[None])).numpy()[0]
            q = np.quantile(samples[:, 0], LEVELS, axis=0)
            q[:, :, :3] = np.round(q[:, :, :3])
            quantiles.append(q)
            metrics = quantile_scores(q[:, valid].T, truth[valid]).to_dict('records')
            for (h, c, l), metric in zip(zip(*np.where(valid)), metrics):
                target_day = episode['target_dates'][h + (2 if direct else 0)]
                rows.append(dict(variant=variant, configuration=scenario_string, seed=seed, scenario=scenario,
                    issuance_date=episode['issuance_date'], target_date=target_day,
                    season=season(date.fromisoformat(target_day)), target=CHANNELS[c],
                    location=ds.locations[l], horizon=h - (0 if direct else 2),
                    task='forecast' if direct or h >= 2 else 'nowcast',
                    observed=float(truth[h, c, l]), loss_scale=float(scales[c, l]),
                    objective_weight=float(weights[i, h, c, l]),
                    crps=float(crps[h, c, l]), **metric,
                    coverage_50=float(q[6, h, c, l] <= truth[h, c, l] <= q[16, h, c, l]),
                    coverage_95=float(q[1, h, c, l] <= truth[h, c, l] <= q[21, h, c, l])))
            if i == len(episodes) // 2 and scenario == 'natural':
                path_graph(samples[:, 0], episode, ds, variant, seed, root, direct)
        np.savez_compressed(root / f'evaluation-masks-{variant}-s{seed}-{scenario}.npz',
            D=np.stack(dropouts), issuance_dates=[e['issuance_date'] for e in episodes])
        np.savez_compressed(root / f'forecasts-{variant}-s{seed}-{scenario}.npz',
            quantiles=np.stack(quantiles, axis=1), quantile_levels=LEVELS,
            truth=np.stack([e['Y'][hs, :, 0] for e in episodes]),
            mask=np.stack([e['Y'][hs, :, 1].astype(bool) for e in episodes]),
            target_dates=np.array([e['target_dates'][hs] for e in episodes]),
            issuance_dates=[e['issuance_date'] for e in episodes], locations=ds.locations,
            channels=CHANNELS, horizons=np.arange(0 if direct else -2, 4))
    frame = pd.DataFrame(rows)
    frame.to_parquet(root / f'scores-{variant}-s{seed}.parquet', index=False)
    return rows


def path_graph(samples, episode, ds, variant, seed, root, direct):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    l = ds.locations.index('US') if 'US' in ds.locations else 0
    h = np.arange(4) if direct else np.arange(-2, 4)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for c, ax in enumerate(axes.flat):
        for member in samples[:30]:
            ax.plot(h, member[:, c, l], alpha=.2, lw=.7, color='steelblue')
        valid = episode['Y'][:, c, 1, l].astype(bool)
        ax.scatter(np.arange(-2, 4)[valid], episode['Y'][valid, c, 0, l], s=18, color='black', label='Reference final')
        available = episode['X'][-2:, c, 1, l].astype(bool)
        ax.scatter(np.array([-2, -1])[available], episode['X'][-2:, c, 0, l][available], marker='x', color='orange', label='Wednesday')
        ax.axvline(-.5, color='gray', linestyle='--')
        ax.set_title(CHANNELS[c]);ax.set_xlabel('Offset from following Saturday')
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f'{variant}, seed {seed}, {episode["issuance_date"]}, {ds.locations[l]}: paired sampled paths')
    fig.tight_layout();fig.savefig(root / f'paths-{variant}-s{seed}.png', dpi=160);plt.close(fig)


def data_audit(ds, root):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root);root.mkdir(parents=True, exist_ok=True)
    a = ds.arrays
    sources = np.array([p['source'] for p in ds.metadata['provenance']])
    kinds = np.array(['hub' if s.startswith('hub_') else 'delphi' if s.startswith('delphi_') else 'none' for s in sources])
    rows = []
    for i, issuance in enumerate(a['issuance_dates']):
        for c, channel in enumerate(CHANNELS):
            valid = a['X_available'][i, :, c]
            provider = kinds[a['X_provenance'][i, :, c]]
            rows.append(dict(issuance_date=issuance, target=channel, available=int(valid.sum()), total=int(valid.size),
                hub=int((valid & (provider == 'hub')).sum()), delphi=int((valid & (provider == 'delphi')).sum()),
                missing=int((~valid).sum()), recent_labels=int(a['Y_recent_valid'][i, :, c].sum()),
                future_labels=int(a['Y_future_valid'][i, :, c].sum())))
    frame = pd.DataFrame(rows);frame.to_csv(root / 'source-coverage.csv', index=False)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for c, ax in enumerate(axes.flat):
        f = frame[frame.target == CHANNELS[c]]
        ax.stackplot(pd.to_datetime(f.issuance_date), f.hub/f.total, f.delphi/f.total, f.missing/f.total,
                     labels=['Hub', 'Delphi fallback', 'Missing'], colors=['steelblue', 'orange', 'lightgray'])
        ax.set_title(CHANNELS[c]);ax.set_ylim(0, 1);ax.tick_params(axis='x', rotation=30)
    axes.flat[0].legend(fontsize=8);fig.tight_layout();fig.savefig(root / 'source-coverage.png', dpi=160);plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    audits = []
    for c, ax in enumerate(axes.flat):
        valid = a['X_available'][:, -2:, c] & a['Y_recent_valid'][:, :, c]
        x, y = a['X_values'][:, -2:, c][valid], a['Y_recent'][:, :, c][valid]
        ax.scatter(x, y, s=2, alpha=.15)
        if len(x):
            bounds = [0, max(float(x.max()), float(y.max()))]
            ax.plot(bounds, bounds, color='black', lw=.7)
        ax.set_title(CHANNELS[c]);ax.set_xlabel('Wednesday preliminary');ax.set_ylabel('Reference final')
        audits.append(dict(target=CHANNELS[c], n=int(valid.sum()),
            mean_absolute_revision=float(np.mean(np.abs(x-y))) if len(x) else None))
    fig.tight_layout();fig.savefig(root / 'preliminary-vs-final.png', dpi=160);plt.close(fig)
    (root / 'audit.json').write_text(json.dumps(dict(truth_cutoff=ds.metadata['truth_cutoff'],
        usable_episodes=sum(1 for _ in ds.episodes()), revision_audit=audits,
        coverage_policy=ds.metadata['coverage_policy']), indent=2) + '\n')


def report(rows, ds, root):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    data_audit(ds, root)
    frame = pd.DataFrame(rows)
    # Compare future cells across all variants present for each seed; nowcasts
    # only among models that produce them. Partial manager reports can have
    # different numbers of finished configurations for each seed.
    keys = ['seed', 'scenario', 'issuance_date', 'target_date', 'target', 'location']
    for _, seed_frame in frame.groupby('seed'):
        for recent in (False, True):
            f = seed_frame[seed_frame.horizon.lt(0) if recent else seed_frame.horizon.ge(0)]
            if not (f.groupby(keys).variant.nunique() == f.variant.nunique()).all():
                raise ValueError('Comparison variants do not have identical evaluation support')
    frame['geography'] = np.where(frame.location == 'US', 'US', 'states_dc')
    summary = frame.groupby(['variant', 'scenario', 'target', 'season', 'geography', 'horizon']).agg(
        n=('crps', 'size'), crps=('crps', 'mean'), wis=('wis', 'mean'),
        coverage_50=('coverage_50', 'mean'), coverage_95=('coverage_95', 'mean')).reset_index()
    summary.to_csv(root / 'summary.csv', index=False)
    # Separate the four stress scenarios so a formulation grid does not create
    # dozens of overlapping curves/legend entries inside each target panel.
    for stress in summary.scenario.unique():
        for metric in ('crps', 'wis', 'coverage_50', 'coverage_95'):
            fig, axes = plt.subplots(2, 3, figsize=(14, 10), sharex=True)
            for c, ax in enumerate(axes.flat):
                subset = summary[(summary.target == CHANNELS[c]) & (summary.scenario == stress)]
                for variant, f in subset.groupby('variant'):
                    curve = f.groupby('horizon')[metric].mean()
                    label = variant.rsplit('-', 1)[0].replace('multiscale_conv', 'multi').replace('two_stage', 'two').replace('mask', 'm')
                    ax.plot(curve.index, curve.values, label=label, alpha=.8)
                ax.set_title(CHANNELS[c]);ax.set_xlabel('Week offset (negative = nowcast)');ax.set_ylabel(metric)
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=7)
            fig.suptitle(f'{metric}: {stress} inputs')
            fig.subplots_adjust(bottom=.25, top=.92, hspace=.38, wspace=.3)
            suffix = '' if stress == 'natural' else '-' + stress
            fig.savefig(root / f'{metric}{suffix}-by-horizon.png', dpi=160);plt.close(fig)
    (root / 'README.md').write_text('B1 forward chronological comparison\n\n'
        'See summary.csv and per-cell scores-*.parquet. Recent offsets -2/-1 are reference-final nowcasts; '
        'future offsets 0–3 are forecasts. All variants use identical eligible future cells and fixed stress masks. '
        'Scores are native-unit diagnostics, with states/DC and US reported separately. Plot curves average the '
        'reported season/geography groups; they are not the scientific selection objective. '
        'This native report is separate from the optional manager Hub benchmark and its narrower support. '
        'Later pinned training truth, when enabled explicitly, makes these retrospective development results.\n')


if __name__ == '__main__':
    import argparse
    from tapestry.model_data.wednesday import WednesdayDataset
    parser = argparse.ArgumentParser(description='Audit B1 source coverage and preliminary-to-final revisions')
    parser.add_argument('--dataset', default='data/processed/build_b1_wednesday.npz')
    parser.add_argument('--output', default='data/processed/b1-audit')
    args = parser.parse_args()
    data_audit(WednesdayDataset.load(args.dataset), args.output)
