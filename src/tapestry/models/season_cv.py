"""Three leave-one-season-out fits of the frozen-data B0 pilot; no tuning."""
from __future__ import annotations

import argparse
import csv
from datetime import date
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from tapestry.model_data import CHANNELS, FinalizedDataset
from tapestry.model_data.finalized import season
from .run import calendar, fit
from .experiments import add_experiment_args, LOSS_WEIGHTS

SEASONS = ('2023-2024', '2024-2025', '2025-2026')
LEVELS = np.array([.01, .025, .05, .10, .15, .20, .25, .30, .35, .40,
                   .45, .50, .55, .60, .65, .70, .75, .80, .85, .90, .95, .975, .99])


def fold_data(ds, held_out, lookback=8):
    """Exclude held-out observations from ALL fit inputs, labels and scales.

    Keep a weekly calendar, masking dates outside the two training seasons.
    Evaluate origins in the held-out season; allow observed past context, but
    score only labels in that season. This is finalized-data cross-validation.
    """
    labels = np.array([season(date.fromisoformat(day)) for day in ds.dates])
    train_seasons = [s for s in SEASONS if s != held_out]
    training = np.isin(labels, train_seasons)
    testing = labels == held_out
    panel = ds.panel.copy()
    panel[~training] = 0
    train_ds = FinalizedDataset(panel, ds.dates, ds.locations, ds.metadata)
    episodes = []
    for day in np.array(ds.dates)[training]:
        q = train_ds.query(day, lookback=lookback)
        if q['Y'][:, :, 1].any():
            episodes.append(q)
    evaluation = []
    for day in np.array(ds.dates)[testing]:
        q = ds.query(day, lookback=lookback)
        for h, target in enumerate(q['target_dates']):
            if season(date.fromisoformat(target)) != held_out:
                q['Y'][h] = 0
        if q['Y'][:, :, 1].any():
            evaluation.append(q)
    scales = []
    for c in range(6):
        values = ds.panel[training, c, 0]
        valid = values[ds.panel[training, c, 1].astype(bool)]
        scales.append(max(float(np.quantile(valid, .95)) if valid.size else 0, 1 if c < 3 else .001))
    if not episodes or not evaluation:
        raise ValueError(f'No training or evaluation episodes for {held_out}')
    return episodes, evaluation, scales


def wis(quantiles, truth):
    """23-quantile WIS, quantile axis first; returns scores for every task."""
    score = .5 * np.abs(truth - quantiles[11])
    for i, lower_level in enumerate(LEVELS[:11]):
        alpha = 2 * lower_level
        lower, upper = quantiles[i], quantiles[-i - 1]
        score += alpha / 2 * (upper - lower) + np.maximum(lower - truth, 0) + np.maximum(truth - upper, 0)
    return score / 11.5


def persistence(x):
    """Last observed value per channel/location; no-history cells excluded in comparison."""
    mask = x[:, :, 1].astype(bool)
    idx = (mask * np.arange(1, len(x) + 1)[:, None, None]).argmax(axis=0)
    values = np.take_along_axis(x[:, :, 0], idx[None], axis=0)[0]
    return values, mask.any(axis=0)


def evaluate(model, episodes, args, output):
    model.eval()
    quantiles, retained, truths, masks, baselines, baseline_masks = [], [], [], [], [], []
    for i, episode in enumerate(episodes):
        x = torch.tensor(episode['X'][None], device=args.device)
        cal = torch.tensor(calendar([episode['context_dates'][-1]], model.config['dynamics']), device=args.device)
        with torch.no_grad():
            samples = torch.cat([model(x, cal, min(32, args.eval_members - j), locations=episode['locations']).cpu()
                                 for j in range(0, args.eval_members, 32)], dim=0).numpy()[:, 0]
        if not np.isfinite(samples).all():
            raise ValueError('Nonfinite predictions')
        q = np.quantile(samples, LEVELS, axis=0)
        q[:, :, :3] = np.floor(q[:, :, :3] + .5)  # fixed count quantile rounding
        quantiles.append(q)
        retained.append(samples[:100])  # whole members, same IDs across all tasks
        truths.append(episode['Y'][:, :, 0])
        masks.append(episode['Y'][:, :, 1].astype(bool))
        last, valid = persistence(episode['X'])
        last[:3] = np.floor(last[:3] + .5)
        baselines.append(np.broadcast_to(last, truths[-1].shape))
        baseline_masks.append(np.broadcast_to(valid, truths[-1].shape))
        if (i + 1) % 10 == 0 or i == len(episodes) - 1:
            print(json.dumps({'evaluated': i + 1, 'total': len(episodes)}), flush=True)
    # Forecasts stored before scoring, so metrics can be recomputed independently.
    q = np.stack(quantiles, axis=1)  # Q,N,H,C,L
    y, mask = np.stack(truths), np.stack(masks)
    baseline, baseline_mask = np.stack(baselines), np.stack(baseline_masks)
    np.savez_compressed(output / 'forecasts.npz', quantiles=q, quantile_levels=LEVELS,
                        samples=np.stack(retained, axis=1), truth=y, mask=mask,
                        baseline=baseline, baseline_mask=baseline_mask,
                        context_end=[e['context_dates'][-1] for e in episodes],
                        target_dates=[e['target_dates'] for e in episodes],
                        locations=episodes[0]['locations'], channels=CHANNELS,
                        metadata=json.dumps({'eval_members': args.eval_members, 'saved_members': min(100, args.eval_members),
                                             'quantile_axes': ['quantile', 'origin', 'horizon', 'channel', 'location'],
                                             'sample_axes': ['member', 'origin', 'horizon', 'channel', 'location'],
                                             'member_identity': 'shared across horizon/channel/location within each origin',
                                             'count_quantiles': 'round-half-up', 'baseline': 'deterministic last observation'}))
    metrics = {
        'wis': wis(q, y), 'mae': np.abs(q[11] - y),
        'coverage80': ((y >= q[3]) & (y <= q[-4])).astype(float),
        'coverage95': ((y >= q[1]) & (y <= q[-2])).astype(float),
        'width80': q[-4] - q[3],
        'baseline_wis': wis(np.broadcast_to(baseline, q.shape), y),
    }
    common = mask & baseline_mask
    rows = []
    locations = np.array(episodes[0]['locations'])
    for c, channel in enumerate(CHANNELS):
        for group in ('states_dc', 'US'):
            locs = locations != 'US' if group == 'states_dc' else locations == 'US'
            for horizon in (0, 1, 2, 3, 4):  # 0 is the pooled four-week score
                hm = np.ones(4, dtype=bool) if horizon == 0 else np.arange(4) == horizon - 1
                selected = mask[:, :, c, :] & hm[None, :, None] & locs[None, None, :]
                matched = common[:, :, c, :] & selected
                row = {'channel': channel, 'geography': group, 'horizon': 'all' if horizon == 0 else horizon,
                       'n': int(selected.sum()), 'baseline_comparison_n': int(matched.sum())}
                for name, values in metrics.items():
                    valid = matched if name == 'baseline_wis' else selected
                    row[name] = float(values[:, :, c, :][valid].mean()) if valid.any() else None
                row['model_wis_common'] = float(metrics['wis'][:, :, c, :][matched].mean()) if matched.any() else None
                base = row['baseline_wis']
                row['wis_ratio_to_persistence'] = row['model_wis_common'] / base if base else None
                rows.append(row)
    with (output / 'scores.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run(args):
    torch.set_num_threads(2)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    ds = FinalizedDataset.load(args.dataset)
    started = time.perf_counter()
    code_paths = [Path(__file__), Path(__file__).with_name('b0.py'), Path(__file__).with_name('run.py'), Path(__file__).with_name('experiments.py'),
                  Path(__file__).parents[1] / 'model_data' / 'finalized.py']
    manifest = {'config': vars(args), 'platform': platform.platform(), 'torch_version': str(torch.__version__),
                'dataset_sha256': hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
                'code_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths},
                'protocol': 'fixed 50-epoch default, three leave-one-season-out finalized-data folds; no holdout tuning',
                'season_definition': 'CDC epiweek 31–30; season 1 begins at available September 2023 data',
                'stride_weeks': 1, 'horizons': [1, 2, 3, 4],
                'holdout_rule': 'excluded from fit context, targets, and scalers; evaluation labels stay within held-out season',
                'prior_exposure': 'Season 1 already used for skeleton smoke fitting; all seasons now used for requested CV, none reserved as untouched holdout.',
                'folds': []}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    all_scores = []
    for held_out in SEASONS:
        torch.manual_seed(args.seed)
        train_seasons = [s for s in SEASONS if s != held_out]
        folder = output / f'eval_{held_out}'
        folder.mkdir()
        training, evaluation, scales = fold_data(ds, held_out, args.lookback)
        info = {'train_seasons': train_seasons, 'eval_season': held_out,
                'train_episodes': len(training), 'eval_origins': len(evaluation),
                'training_context_ends': [e['context_dates'][-1] for e in training],
                'evaluation_context_ends': [e['context_dates'][-1] for e in evaluation],
                'scale': scales, 'chronological': held_out == SEASONS[-1]}
        print(json.dumps(info), flush=True)
        before = time.perf_counter()
        model, history = fit(training, scales, args)
        info['train_seconds'] = time.perf_counter() - before
        info['history'] = history
        info['experiment'] = model.config
        info['loss_weights'] = LOSS_WEIGHTS[args.loss_weights]
        info['parameters'] = sum(p.numel() for p in model.parameters())
        torch.save({'config': model.config, 'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                    'metadata': {**info, 'dataset_sha256': manifest['dataset_sha256'], 'channels': list(CHANNELS),
                                 'locations': list(ds.locations), 'seed': args.seed}}, folder / 'model.pt')
        torch.manual_seed(args.seed + 1000)
        before = time.perf_counter()
        scores = evaluate(model, evaluation, args, folder)
        info['evaluation_seconds'] = time.perf_counter() - before
        (folder / 'training.json').write_text(json.dumps(info, indent=2) + '\n')
        for row in scores:
            all_scores.append({'eval_season': held_out, 'train_seasons': '+'.join(train_seasons), **row})
        manifest['folds'].append(info)
        manifest['elapsed_seconds'] = time.perf_counter() - started
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with (output / 'scores.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_scores[0]))
        writer.writeheader()
        writer.writerows(all_scores)
    print(json.dumps({'output': str(output), 'elapsed_seconds': manifest['elapsed_seconds']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='data/processed/build_b_finalized.npz')
    parser.add_argument('--output', default='data/experiments/b0_season_cv')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lookback', type=int, default=8)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--members', type=int, default=8)
    parser.add_argument('--eval-members', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=42)
    add_experiment_args(parser)
    args = parser.parse_args()
    args.horizons = [1, 2, 3, 4]
    if min(args.epochs, args.lookback, args.width, args.batch_size, args.eval_members) < 1 or args.members < 2:
        parser.error('Positive dimensions/epochs required; training members >= 2')
    run(args)


if __name__ == '__main__':
    main()
