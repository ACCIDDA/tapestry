"""Three leave-one-season-out fits of the frozen-data B0 pilot, with optional early stopping."""
from __future__ import annotations

import argparse
import os
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
from .bundles import GROUPS, IndependentBundle, supervision, checkpoint
from .experiments import add_experiment_args, LOSS_WEIGHTS, model_options
from .objective import LOSS_DEFINITION, US_WEIGHT, loss_scales
from .provenance import git_state
from .quantiles import LEVELS

SEASONS = ('2023-2024', '2024-2025', '2025-2026')
# Early stopping hides weeks 4–6, 20–22, and 36–38 of each training season.
VALIDATION_WEEKS, VALIDATION_SPACING, VALIDATION_OFFSET = 3, 16, 4
COVERAGE = (50, 80, 90, 95)


def season_labels(ds):
    return np.array([season(date.fromisoformat(day)) for day in ds.dates])


def masked_episodes(ds, keep, origins, lookback):
    """Episodes ending at `origins` whose context and labels come only from weeks in `keep`."""
    panel = ds.panel.copy()
    panel[~keep] = 0
    masked = FinalizedDataset(panel, ds.dates, ds.locations, ds.metadata)
    episodes = []
    for day in origins:
        q = masked.query(day, lookback=lookback)
        if q['Y'][:, :, 1].any():
            episodes.append(q)
    return episodes


def channel_scales(ds, keep):
    """Native-unit Q95 per channel/location, using only permitted unique weeks."""
    return loss_scales(ds.panel[keep])


def fold_data(ds, held_out, lookback=8):
    """Exclude held-out observations from ALL fit inputs, labels and scales.

    Keep a weekly calendar, masking dates outside the two training seasons.
    Evaluate origins in the held-out season; allow observed past context, but
    score only labels in that season. This is finalized-data cross-validation.
    """
    labels = season_labels(ds)
    training = np.isin(labels, [s for s in SEASONS if s != held_out])
    testing = labels == held_out
    episodes = masked_episodes(ds, training, np.array(ds.dates)[training], lookback)
    evaluation = []
    for day in np.array(ds.dates)[testing]:
        q = ds.query(day, lookback=lookback)
        for h, target in enumerate(q['target_dates']):
            if season(date.fromisoformat(target)) != held_out:
                q['Y'][h] = 0
        if q['Y'][:, :, 1].any():
            evaluation.append(q)
    scales = channel_scales(ds, training)
    if not episodes or not evaluation:
        raise ValueError(f'No training or evaluation episodes for {held_out}')
    return episodes, evaluation, scales


def validation_split(ds, held_out, lookback=8, horizons=4):
    """Early-stopping episodes inside the two training seasons of a fold.

    Each training season hides three consecutive target weeks out of every sixteen
    (season start, winter, and spring). Hidden weeks are removed from the inner fit's
    context, labels and scales, as a held-out season is from a fold. Validation
    episodes are the origins with a hidden week among their targets; only hidden
    weeks are scored, so each is predicted at every horizon. Validation episodes may
    condition on earlier training-season weeks.
    """
    labels = season_labels(ds)
    training = np.isin(labels, [s for s in SEASONS if s != held_out])
    hidden = np.zeros(len(ds.dates), dtype=bool)
    for label in SEASONS:
        if label != held_out:
            weeks = np.flatnonzero(labels == label)
            position = np.arange(len(weeks)) % VALIDATION_SPACING
            hidden[weeks[(position >= VALIDATION_OFFSET) & (position < VALIDATION_OFFSET + VALIDATION_WEEKS)]] = True
    origins = np.zeros(len(ds.dates), dtype=bool)
    for i in np.flatnonzero(hidden):
        origins[max(i - horizons, 0):i] = True
    origins &= training
    inner = training & ~hidden
    dates = np.array(ds.dates)
    hidden_dates = set(dates[hidden])
    inner_episodes = masked_episodes(ds, inner, dates[inner], lookback)
    validation = []
    for episode in masked_episodes(ds, training, dates[origins], lookback):
        for h, target in enumerate(episode['target_dates']):
            if target not in hidden_dates:
                episode['Y'][h] = 0
        if episode['Y'][:, :, 1].any():
            validation.append(episode)
    if not inner_episodes or not validation:
        raise ValueError(f'No inner training or validation episodes for {held_out}')
    info = {'validation_weeks': sorted(hidden_dates), 'validation_origins': [e['context_dates'][-1] for e in validation],
            'validation_target_weeks': int(hidden.sum()),
            'inner_weeks': int(inner.sum()), 'training_weeks': int(training.sum()),
            'inner_episodes': len(inner_episodes), 'validation_episodes': len(validation)}
    return inner_episodes, validation, channel_scales(ds, inner), info


def wis(quantiles, truth):
    """Quantile WIS diagnostic (twice the mean pinball loss); rankings use `tapestry.evaluation.totals`."""
    if quantiles.shape[0] != len(LEVELS):
        raise ValueError(f'WIS expects the {len(LEVELS)} saved quantiles')
    levels = LEVELS.reshape((-1,) + (1,) * (quantiles.ndim - 1))
    error = truth - quantiles
    return 2 * np.maximum(levels * error, (levels - 1) * error).mean(axis=0)


def persistence(x):
    """Last observed value per channel/location; no-history cells excluded in comparison."""
    mask = x[:, :, 1].astype(bool)
    idx = (mask * np.arange(1, len(x) + 1)[:, None, None]).argmax(axis=0)
    values = np.take_along_axis(x[:, :, 0], idx[None], axis=0)[0]
    return values, mask.any(axis=0)


def level(value):
    return int(np.flatnonzero(np.isclose(LEVELS, value))[0])


def evaluate(model, episodes, args, output, name=''):
    """Save quantiles, retained members, and diagnostics as `<name>forecasts.npz` and `<name>scores.csv`."""
    model.eval()
    quantiles, retained, truths, masks, baselines, baseline_masks = [], [], [], [], [], []
    for i, episode in enumerate(episodes):
        x = torch.tensor(episode['X'][None], device=args.device)
        cal = torch.tensor(calendar([episode['context_dates'][-1]], model.config.get('annual_calendar', True)), device=args.device)
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
    np.savez_compressed(output / f'{name}forecasts.npz', quantiles=q, quantile_levels=LEVELS,
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
    metrics = {'wis': wis(q, y), 'mae': np.abs(q[level(.5)] - y), 'width50': q[level(.75)] - q[level(.25)],
               'baseline_wis': wis(np.broadcast_to(baseline, q.shape), y)}
    for coverage in COVERAGE:
        tail = (1 - coverage / 100) / 2
        metrics[f'coverage{coverage}'] = ((y >= q[level(tail)]) & (y <= q[level(1 - tail)])).astype(float)
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
                for metric, values in metrics.items():
                    valid = matched if metric == 'baseline_wis' else selected
                    row[metric] = float(values[:, :, c, :][valid].mean()) if valid.any() else None
                row['model_wis_common'] = float(metrics['wis'][:, :, c, :][matched].mean()) if matched.any() else None
                base = row['baseline_wis']
                row['wis_ratio_to_persistence'] = row['model_wis_common'] / base if base else None
                rows.append(row)
    with (output / f'{name}scores.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run(args):
    torch.set_num_threads(int(os.environ.get('TAPESTRY_TORCH_THREADS', '2')))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    ds = FinalizedDataset.load(args.dataset)
    started = time.perf_counter()
    code_paths = [Path(__file__), Path(__file__).with_name('b0.py'), Path(__file__).with_name('run.py'), Path(__file__).with_name('experiments.py'), Path(__file__).with_name('quantiles.py'), Path(__file__).with_name('objective.py'), Path(__file__).with_name('architecture.py'), Path(__file__).with_name('bundles.py'),
                  Path(__file__).parents[1] / 'model_data' / 'finalized.py']
    stopping = (f'early stopping on inner validation blocks (patience {args.patience}, cap {args.epochs} epochs), '
                'then a refit on all training weeks for the best epoch count' if args.patience else f'fixed {args.epochs} epochs')
    from .scenarios import TrainingScenario
    manifest = {'scenario_string': TrainingScenario.from_config(vars(args)).scenario_string, 'config': vars(args), 'platform': platform.platform(), 'torch_version': str(torch.__version__), 'torch_threads': torch.get_num_threads(),
                'dataset_sha256': hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
                'code_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths},
                'git': git_state(),
                'protocol': f'{stopping}; three leave-one-season-out finalized-data folds; no holdout tuning',
                'loss': LOSS_DEFINITION, 'us_weight': US_WEIGHT, 'loss_weights': LOSS_WEIGHTS[args.loss_weights],
                'season_definition': 'CDC epiweek 31–30; season 1 begins at available September 2023 data',
                'stride_weeks': 1, 'horizons': [1, 2, 3, 4],
                'holdout_rule': 'excluded from fit context, targets, and scalers; evaluation labels stay within held-out season',
                'prior_exposure': 'Season 1 already used for skeleton smoke fitting; all seasons now used for requested CV, none reserved as untouched holdout.',
                'quantile_levels': LEVELS.tolist(), 'folds': []}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    all_scores = []
    for held_out in SEASONS:
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
        groups = GROUPS[args.fit_partition]
        models, stopped_models, components = [], [], []
        if args.patience:
            inner, validation, inner_scales, split = validation_split(ds, held_out, args.lookback, len(args.horizons))
        for ci, channels in enumerate(groups):
            component_seed = args.seed + 10000 * ci
            fit_args = argparse.Namespace(**{**vars(args), 'seed': component_seed})
            detail = dict(channels=channels, seed=component_seed, epoch_cap=args.epochs)
            if args.device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            if args.patience:
                torch.manual_seed(component_seed)
                before = time.perf_counter()
                stopped, record = fit(supervision(inner, channels), inner_scales, fit_args,
                                      options=model_options(inner, fit_args), validation=supervision(validation, channels))
                detail['early_stopping'] = dict(split, **record, scale=inner_scales,
                    seconds=time.perf_counter() - before, local_noise_scale=stopped.local_noise_scales())
                fit_args = argparse.Namespace(**{**vars(fit_args), 'epochs': record['best_epoch']})
                stopped_models.append(stopped.cpu())
            torch.manual_seed(component_seed)
            before = time.perf_counter()
            model, record = fit(supervision(training, channels), scales, fit_args, options=model_options(training, fit_args))
            detail.update(train_seconds=time.perf_counter() - before, epochs=fit_args.epochs,
                          history=record['loss'], local_noise_scale=model.local_noise_scales(),
                          experiment=model.config, parameters=sum(p.numel() for p in model.parameters()),
                          peak_cuda_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else None)
            models.append(model.cpu())
            components.append(detail)
        model = models[0] if args.fit_partition == 'all' else IndependentBundle(models, groups)
        info.update(components=components, fit_partition=args.fit_partition,
                    train_seconds=sum(c['train_seconds'] for c in components),
                    epochs=[c['epochs'] for c in components] if len(components) > 1 else components[0]['epochs'],
                    parameters=sum(c['parameters'] for c in components), loss_weights=LOSS_WEIGHTS[args.loss_weights],
                    loss=LOSS_DEFINITION, us_weight=US_WEIGHT,
                    evaluation_seed=args.seed + 1000, evaluation_sample_batch=32)
        if len(components) == 1:
            info.update(components[0])
        metadata = {**info, 'dataset_sha256': manifest['dataset_sha256'], 'channels': list(CHANNELS),
                    'locations': list(ds.locations), 'seed': args.seed}
        torch.save(checkpoint(model, metadata), folder / 'model.pt')
        if args.patience:
            stopped = stopped_models[0] if args.fit_partition == 'all' else IndependentBundle(stopped_models, groups)
            torch.save(checkpoint(stopped, metadata), folder / 'validation_model.pt')
            torch.manual_seed(args.seed + 1000)
            evaluate(stopped.to(args.device), validation, args, folder, name='validation_')
            stopped.cpu()
            del stopped, stopped_models
        model.to(args.device)
        torch.manual_seed(args.seed + 1000)
        before = time.perf_counter()
        scores = evaluate(model, evaluation, args, folder)
        info['evaluation_seconds'] = time.perf_counter() - before
        (folder / 'training.json').write_text(json.dumps(info, indent=2) + '\n')
        for row in scores:
            all_scores.append({'eval_season': held_out, 'train_seasons': '+'.join(train_seasons), **row})
        manifest['folds'].append(info)
        model.cpu()
        del model, models
        manifest['elapsed_seconds'] = time.perf_counter() - started
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with (output / 'scores.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_scores[0]))
        writer.writeheader()
        writer.writerows(all_scores)
    print(json.dumps({'output': str(output), 'elapsed_seconds': manifest['elapsed_seconds']}), flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='data/processed/build_b_finalized.npz')
    parser.add_argument('--output', default='data/experiments/b0_season_cv')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--epochs', type=int, default=50, help='Fixed epochs, or the cap with --patience')
    parser.add_argument('--patience', type=int, default=0,
                        help='0 trains fixed epochs; otherwise stop on inner validation blocks, then refit')
    parser.add_argument('--lookback', type=int, default=8)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--members', type=int, default=128,
                        help='Training draws per episode; fair CRPS is unbiased at any m >= 2, so this trades time for gradient variance')
    parser.add_argument('--eval-members', type=int, default=256)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=42)
    add_experiment_args(parser)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.horizons = [1, 2, 3, 4]
    if min(args.epochs, args.lookback, args.width, args.batch_size, args.eval_members) < 1 or args.members < 2:
        parser.error('Positive dimensions/epochs required; training members >= 2')
    if args.validation_members < 2:
        parser.error('validation-members must be >=2')
    if args.patience < 0:
        parser.error('patience must be nonnegative')
    run(args)


if __name__ == '__main__':
    main()
