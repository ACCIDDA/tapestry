"""Paired scientific comparison of saved manager runs; uses the shared scorer."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tapestry.models.b1 import MASK_SCENARIOS
from tapestry.models.manager import completed_runs
from tapestry.models.provenance import SEASONS
from tapestry.models.quantiles import LEVELS
from .hubs import KEY
from .totals import (METRICS, TARGET_WEIGHTS, cells_totals, forecast_cells,
                     run_scores, score_run, season_scores)

CELL_KEYS = ['season', 'target', *KEY]

EXPERIMENTS = ('B1-onlymask-refit', 'B1-direct-finalflag', 'B1-joint-aux025')


def evaluation_members(run):
    """Read the actual fitted attempt's budget, never a subsequently edited plan."""
    meta = json.loads((run / 'manifest.json').read_text())
    if 'eval_members' in meta:
        return int(meta['eval_members'])
    return int(json.loads((run.parent / 'run.json').read_text())['settings']['eval_members'])


def regenerate_draws(run, dataset, device):
    """Recover raw members from compatible checkpoints, verifying archived quantiles/masks."""
    from tapestry.model_data.wednesday import WednesdayDataset
    from tapestry.models.b1_run import crop_episodes, load_models, sample
    from tapestry.models.b1_seasons import fold
    import torch
    torch.set_num_threads(1)
    ds = WednesdayDataset.load(dataset)
    meta = json.loads((run / 'manifest.json').read_text())
    seed, prefix = meta['seed'], f"{meta['run_id']}-s{meta['seed']}"
    for held in SEASONS:
        folder = run / f'eval_{held}'
        missing = [stress for stress in MASK_SCENARIOS if not (folder / f'draws-{prefix}-{stress}.npy').exists()]
        if not missing:
            continue
        models, _ = load_models(folder / 'model.pt', device)
        episodes = crop_episodes(fold(ds, held, direct=meta['configuration']['pipeline'] != 'two_stage')[3],
                                 models[0].config['lookback'])
        episodes = [e for e in episodes if e['X'][:, :, 1].any()]
        for stress in missing:
            archive = np.load(folder / f'forecasts-{prefix}-{stress}.npz')
            masks = np.load(folder / f'evaluation-masks-{prefix}-{stress}.npz')
            assert list(archive['issuance_dates']) == [e['issuance_date'] for e in episodes]
            members = evaluation_members(run)
            temporary = folder / f'draws-{prefix}-{stress}.tmp.npy'
            draws = None
            for i, episode in enumerate(episodes):
                samples, d = sample(models, [episode], members=members, seed=seed + i * 101,
                                    device=device, scenario=stress)
                q = np.quantile(samples[:, 0], LEVELS, axis=0)
                q[:, :, :3] = np.round(q[:, :, :3])
                np.testing.assert_allclose(q, archive['quantiles'][:, i], rtol=2e-5, atol=1e-5)
                np.testing.assert_array_equal(d[0], masks['D'][i])
                if draws is None:
                    draws = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32,
                        shape=(len(episodes), members, *samples.shape[2:]))
                draws[i] = samples[:, 0]
            draws.flush(); del draws
            temporary.replace(folder / f'draws-{prefix}-{stress}.npy')
            print(f'Recovered and verified draws: {run}, {held}, {stress}', flush=True)


def mixture(runs, destination, dataset, device):
    """Equal seed mass: pool all raw draws per origin, then calculate quantiles."""
    from tapestry.model_data.wednesday import WednesdayDataset
    from tapestry.models.b1_run import crop_episodes, load_models, sample
    from tapestry.models.b1_seasons import fold
    ds = WednesdayDataset.load(dataset)
    destination.mkdir(parents=True, exist_ok=True)
    metas = [json.loads((r / 'manifest.json').read_text()) for r in runs]
    meta = dict(metas[0], run_id='equal-mixture', seed=0)
    (destination / 'manifest.json').write_text(json.dumps(meta, indent=2))
    for held in SEASONS:
        output = destination / f'eval_{held}'
        output.mkdir(exist_ok=True)
        models = [load_models(run / f'eval_{held}/model.pt', device)[0] for run in runs]
        episodes = crop_episodes(fold(ds, held, direct=True)[3], models[0][0].config['lookback'])
        episodes = [e for e in episodes if e['X'][:, :, 1].any()]
        for stress in MASK_SCENARIOS:
            archives, draws = [], []
            for run, m in zip(runs, metas):
                prefix = f"{m['run_id']}-s{m['seed']}"
                archives.append(np.load(run / f'eval_{held}/forecasts-{prefix}-{stress}.npz'))
                draws.append(np.load(run / f'eval_{held}/draws-{prefix}-{stress}.npy', mmap_mode='r'))
            reference = archives[0]
            for other in archives[1:]:
                for key in ('issuance_dates', 'target_dates', 'locations', 'channels', 'horizons', 'truth'):
                    np.testing.assert_array_equal(reference[key], other[key])
            if len({d.shape for d in draws}) != 1:
                raise ValueError('Equal-mixture members and support differ across seeds')
            quantiles = []
            for i in range(len(reference['issuance_dates'])):
                if stress == 'natural':
                    pooled = np.concatenate([d[i] for d in draws])
                else:
                    pooled_parts, common_mask = [], None
                    for bundle, m in zip(models, metas):
                        values, mask = sample(bundle, [episodes[i]], members=draws[0].shape[1],
                            seed=m['seed'] + i * 101, mask_seed=42 + i * 101,
                            device=device, scenario=stress, sample_batch=256)
                        if common_mask is None:
                            common_mask = mask
                        else:
                            np.testing.assert_array_equal(mask, common_mask)
                        pooled_parts.append(values[:, 0])
                    pooled = np.concatenate(pooled_parts)
                q = np.quantile(pooled, LEVELS, axis=0)
                q[:, :, :3] = np.round(q[:, :, :3])
                quantiles.append(q)
            contents = {k: reference[k] for k in reference.files}
            contents['quantiles'] = np.stack(quantiles, axis=1)
            np.savez_compressed(output / f'forecasts-equal-mixture-s0-{stress}.npz', **contents)
            for archive in archives:
                archive.close()
            print(f'Mixture quantiles: {destination.name}, {held}, {stress}', flush=True)
    return destination


def scalar(cells):
    scores = run_scores(season_scores(cells_totals(cells).assign(config_id='candidate', seed=0)))
    return float(scores.loc[scores.geography == 'all', 'combined'].iloc[0])


def block_counts(origins, length, repetitions, rng):
    """Season-stratified moving blocks on weekly origins, no wrap at boundaries."""
    counts = np.zeros((repetitions, len(origins)), dtype=np.float64)
    for season in origins.season.unique():
        ids = np.flatnonzero(origins.season.eq(season))
        dates = pd.to_datetime(origins.iloc[ids].reference_date)
        if len(dates) > 1 and not (np.diff(dates.values).astype('timedelta64[D]').astype(int) == 7).all():
            raise ValueError('Temporal blocks require consecutive weekly origins within each season')
        width = min(length, len(ids))
        starts = rng.integers(0, len(ids) - width + 1, size=(repetitions, int(np.ceil(len(ids) / width))))
        sampled = (starts[..., None] + np.arange(width)).reshape(repetitions, -1)[:, :len(ids)]
        for b, row in enumerate(sampled):
            counts[b, ids] = np.bincount(row, minlength=len(ids))
    return counts


def temporal_intervals(frames, length, repetitions=2000):
    """Paired origins, all locations/targets/horizons together; recompute every ratio."""
    labels = list(frames)
    base = frames[labels[0]].sort_values(CELL_KEYS).reset_index(drop=True)
    keys = CELL_KEYS
    aligned = []
    for label in labels:
        frame = frames[label].sort_values(keys).reset_index(drop=True)
        if not base[keys].equals(frame[keys]):
            raise ValueError('Temporal comparison has unmatched task keys')
        np.testing.assert_allclose(base.ensemble_wis, frame.ensemble_wis)
        aligned.append(frame)
    origins = pd.concat([pd.DataFrame(dict(season=held,
        reference_date=pd.date_range(part.reference_date.min(), part.reference_date.max(), freq='7D').strftime('%Y-%m-%d')))
        for held, part in base.groupby('season')], ignore_index=True)
    groups = base[['season', 'target', 'location']].drop_duplicates().sort_values(['season', 'target', 'location']).reset_index(drop=True)
    oi = pd.MultiIndex.from_frame(origins).get_indexer(pd.MultiIndex.from_frame(base[['season', 'reference_date']]))
    gi = pd.MultiIndex.from_frame(groups).get_indexer(pd.MultiIndex.from_frame(base[['season', 'target', 'location']]))
    weights = np.zeros(len(groups))
    for held, part in groups.groupby('season'):
        targets = part.target.unique()
        tw = sum(TARGET_WEIGHTS[t] for t in targets)
        for target, local in part.groupby('target'):
            us = local.location.eq('US')
            lw = np.where(us, .2 / max(1, us.sum()), .8 / max(1, (~us).sum()))
            lw /= lw.sum()
            weights[local.index] = lw * TARGET_WEIGHTS[target] / tw / groups.season.nunique()
    shape = (len(origins), len(groups))
    denominator = np.zeros(shape)
    np.add.at(denominator, (oi, gi), base.ensemble_wis)
    counts = block_counts(origins, length, repetitions, np.random.default_rng(20260917 + length))
    den = counts @ denominator
    usable = (den > 0).all(1)
    if not usable.any():
        raise ValueError('No temporal block draws retain required location support')
    estimates = {}
    for label, frame in zip(labels, aligned):
        numerator = np.zeros(shape)
        np.add.at(numerator, (oi, gi), frame.model_wis)
        estimates[label] = ((counts[usable] @ numerator) / den[usable]) @ weights
        np.testing.assert_allclose((numerator.sum(0) / denominator.sum(0)) @ weights, scalar(frame), rtol=1e-12)
    rows = []
    pairs = [(candidate, labels[0]) for candidate in labels[1:]]
    if len(labels) == 3:
        pairs.append((labels[2], labels[1]))
    for candidate, baseline in pairs:
        diff = estimates[candidate] - estimates[baseline]
        rel = estimates[candidate] / estimates[baseline] - 1
        rows.append(dict(candidate=candidate, baseline=baseline, block_weeks=length,
            repetitions=repetitions, excluded_missing_support=int((~usable).sum()),
            difference_low=np.quantile(diff, .025), difference_high=np.quantile(diff, .975),
            relative_low=np.quantile(rel, .025), relative_high=np.quantile(rel, .975)))
    return rows


def compare(root, output, device='cuda', repetitions=2000):
    output.mkdir(parents=True, exist_ok=True)
    # Recover the three reused A seeds concurrently in isolated CUDA processes.
    # Process isolation preserves each archived generator stream exactly.
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    recovery = []
    member_counts = set()
    for experiment in EXPERIMENTS:
        folder = root / experiment
        settings = json.loads((folder / 'experiment.json').read_text())
        done, _ = completed_runs(folder, False)
        for row in done:
            run = folder / row['attempt'] / 'b1'
            member_counts.add(evaluation_members(run))
            meta = json.loads((run / 'manifest.json').read_text())
            prefix = f"{meta['run_id']}-s{meta['seed']}"
            if any(not (run / f'eval_{held}/draws-{prefix}-{stress}.npy').exists()
                   for held in SEASONS for stress in MASK_SCENARIOS):
                recovery.append((run, settings['dataset'], device))
    if len(member_counts) != 1:
        raise ValueError('Re-evaluate completed checkpoints to a common trajectory count before comparing seeds')
    members = member_counts.pop()
    if recovery:
        with ProcessPoolExecutor(max_workers=min(3, len(recovery)), mp_context=get_context('spawn')) as pool:
            futures = [pool.submit(regenerate_draws, *args) for args in recovery]
            for future in futures:
                future.result()
    all_runs, records, calibration, paired, temporal = {}, [], [], [], []
    natural_reference = None
    masks_by_seed = {}
    for label, experiment in zip('ABC', EXPERIMENTS):
        folder = root / experiment
        settings = json.loads((folder / 'experiment.json').read_text())
        done, _ = completed_runs(folder, False)
        if sorted(row['seed'] for row in done) != list(range(42, 52)):
            raise ValueError('Require exactly seeds 42 through 51 and one candidate per experiment')
        all_runs[label] = []
        for row in sorted(done, key=lambda r: r['seed']):
            run = folder / row['attempt'] / 'b1'
            meta = json.loads((run / 'manifest.json').read_text())
            prefix = f"{meta['run_id']}-s{meta['seed']}"
            for held in SEASONS:
                for stress in MASK_SCENARIOS:
                    with np.load(run / f'eval_{held}/evaluation-masks-{prefix}-{stress}.npz') as m:
                        key = (row['seed'], held, stress)
                        if key in masks_by_seed:
                            np.testing.assert_array_equal(m['D'], masks_by_seed[key][0])
                            np.testing.assert_array_equal(m['issuance_dates'], masks_by_seed[key][1])
                        else:
                            masks_by_seed[key] = (m['D'].copy(), m['issuance_dates'].copy())
            regenerate_draws(run, settings['dataset'], device)
            old = pd.read_csv(run / 'totals.csv').sort_values(['target', 'season', 'location', 'horizon']).reset_index(drop=True)
            scored = score_run(run, settings['frozen']).sort_values(['target', 'season', 'location', 'horizon']).reset_index(drop=True)
            pd.testing.assert_frame_equal(old.sort_index(axis=1), scored.sort_index(axis=1), check_dtype=False, rtol=1e-12, atol=1e-10)
            all_runs[label].append(run)
            for stress in MASK_SCENARIOS:
                frame = pd.read_parquet(run / f'forecast-cells-{stress}.parquet')
                support = frame[CELL_KEYS].sort_values(CELL_KEYS).reset_index(drop=True)
                if natural_reference is None:
                    natural_reference = support
                if not support.equals(natural_reference):
                    raise ValueError('Candidate/seed/stress frozen support differs')
                records.append(dict(candidate=label, seed=row['seed'], stress=stress, objective=scalar(frame), kind='seed'))
                calibration.append(season_scores(cells_totals(frame).assign(config_id=label, seed=row['seed'])).assign(stress=stress, kind='seed'))
    (output / 'verification.json').write_text(json.dumps(dict(natural_totals_reproduced=True,
        identical_frozen_keys=True, matched_masks_across_candidates=True, seeds=list(range(42, 52))), indent=2))
    seed_scores = pd.DataFrame(records)
    rng = np.random.default_rng(20260917)
    for stress, part in seed_scores.groupby('stress'):
        wide = part.pivot(index='seed', columns='candidate', values='objective')
        for label, baseline in (('B', 'A'), ('C', 'A'), ('C', 'B')):
            diffs = wide[label] - wide[baseline]
            relative = wide[label] / wide[baseline] - 1
            indices = rng.integers(0, len(wide), size=(10000, len(wide)))
            boot = diffs.to_numpy()[indices].mean(1)
            relboot = wide[label].to_numpy()[indices].mean(1) / wide[baseline].to_numpy()[indices].mean(1) - 1
            paired.append(dict(candidate=label, baseline=baseline, stress=stress, mean_difference=diffs.mean(), seed_sd=diffs.std(),
                relative_change=wide[label].mean()/wide[baseline].mean()-1, seeds_better=int((diffs < 0).sum()),
                seed_ci_low=np.quantile(boot, .025), seed_ci_high=np.quantile(boot, .975),
                relative_low=np.quantile(relboot, .025), relative_high=np.quantile(relboot, .975)))
            pd.DataFrame(dict(seed=wide.index, difference=diffs, relative_change=relative)).to_csv(output / f'paired-{label}-minus-{baseline}-{stress}.csv', index=False)
    pd.DataFrame(paired).to_csv(output / 'paired-summary.csv', index=False)
    mixtures = {label: mixture(runs, output / f'mixture-{label}', settings['dataset'], device) for label, runs in all_runs.items()}
    for stress in MASK_SCENARIOS:
        frames, seed_frames = {}, {}
        for label, run in mixtures.items():
            frame = forecast_cells(run, settings['frozen'], stress)
            frame.to_parquet(run / f'forecast-cells-{stress}.parquet', index=False)
            frames[label] = frame
            records.append(dict(candidate=label, seed='mixture', stress=stress, objective=scalar(frame), kind='mixture'))
            calibration.append(season_scores(cells_totals(frame).assign(config_id=label, seed=0)).assign(stress=stress, kind='mixture'))
        for length in (8, 4, 12):
            temporal.extend(dict(row, stress=stress, kind='mixture') for row in temporal_intervals(frames, length, repetitions))
        # Averaging per-cell WIS (not quantiles) is linear through the shared
        # denominators, and exactly equals the mean of individual seed objectives.
        seed_frames = {}
        for label, runs in all_runs.items():
            members = [pd.read_parquet(run / f'forecast-cells-{stress}.parquet').sort_values(CELL_KEYS).reset_index(drop=True) for run in runs]
            frame = members[0].copy()
            frame['model_wis'] = np.mean([member.model_wis.to_numpy() for member in members], axis=0)
            seed_frames[label] = frame
            expected = seed_scores[(seed_scores.candidate == label) & (seed_scores.stress == stress)].objective.mean()
            np.testing.assert_allclose(scalar(frame), expected, rtol=1e-12)
        for length in (8, 4, 12):
            temporal.extend(dict(row, stress=stress, kind='mean_seed')
                            for row in temporal_intervals(seed_frames, length, repetitions))
    scores = pd.DataFrame(records)
    scores.to_csv(output / 'scores.csv', index=False)
    pd.concat(calibration, ignore_index=True).to_csv(output / 'calibration.csv', index=False)
    pd.DataFrame(temporal).to_csv(output / 'temporal-uncertainty.csv', index=False)
    recent_diagnostics(all_runs['C'], output)
    write_recommendation(scores, pd.DataFrame(paired), pd.DataFrame(temporal), output, members)


def recent_diagnostics(runs, output):
    """Partition-weighted scaled recent loss by mechanism, distinct from forecast WIS."""
    parts = []
    for run in runs:
        for path in run.glob('eval_*/scores-*.parquet'):
            frame = pd.read_parquet(path)
            frame = frame[frame.task.eq('nowcast')].copy()
            if not frame.empty:
                frame['scaled_crps'] = frame.crps / frame.loss_scale
                parts.append(frame)
    frame = pd.concat(parts, ignore_index=True)
    keys = ['seed', 'stress', 'recent_kind', 'season', 'target', 'geography']
    frame.groupby(keys).agg(cells=('wis', 'size'), wis=('wis', 'mean'),
        scaled_crps=('scaled_crps', 'mean'), coverage_50=('coverage_50', 'mean'),
        coverage_95=('coverage_95', 'mean')).reset_index().to_csv(output / 'recent-mechanisms.csv', index=False)
    # Its persistence-relative natural score remains in the manager nowcast rank.
    rows = []
    for seed, part in frame.groupby('seed'):
        for stress, f in part.groupby('stress'):
            for kind, group in f.groupby('recent_kind'):
                rows.append(dict(seed=seed, stress=stress, recent_kind=kind,
                    weighted_scaled_crps=float((group.scaled_crps * group.objective_weight).sum()) / group.season.nunique(),
                    task_weight=float(group.objective_weight.sum()) / group.season.nunique()))
    pd.DataFrame(rows).to_csv(output / 'recent-loss-contributions.csv', index=False)


def write_recommendation(scores, paired, temporal, output, members=256):
    mix = scores[scores.kind.eq('mixture')].pivot(index='stress', columns='candidate', values='objective')
    eligible, reasons = [], []
    for label in 'BC':
        natural = mix.loc['natural', label] / mix.loc['natural', 'A'] - 1
        stresses = mix.loc[['recent', 'gap', 'outage'], label] / mix.loc[['recent', 'gap', 'outage'], 'A'] - 1
        pair = paired[paired.candidate.eq(label) & paired.baseline.eq('A')]
        primary = pair[pair.stress.eq('natural')].iloc[0]
        interval = temporal[(temporal.kind == 'mixture') & (temporal.candidate == label) & (temporal.baseline == 'A') &
                            (temporal.stress == 'natural') & (temporal.block_weeks == 8)].iloc[0]
        # Conservative operationalization of "supported" and "ambiguous": both
        # fitting-randomness and primary date intervals must exclude no improvement.
        passes = (natural <= -.05 and primary.relative_change <= -.05 and primary.relative_high < 0
                  and interval.relative_high < 0 and stresses.max() <= .05
                  and pair[~pair.stress.eq('natural')].relative_change.max() <= .05)
        if passes:
            eligible.append(label)
        reasons.append(f'{label}: mixture natural change {natural:+.2%}; worst mixture stress change {stresses.max():+.2%}; passes predeclared gate: {passes}.')
    chosen = eligible[0] if eligible else 'A'
    if eligible == ['B', 'C']:
        cb = paired[(paired.candidate == 'C') & (paired.baseline == 'B') & (paired.stress == 'natural')].iloc[0]
        ti = temporal[(temporal.candidate == 'C') & (temporal.baseline == 'B') &
                      (temporal.kind == 'mixture') & (temporal.stress == 'natural') & (temporal.block_weeks == 8)].iloc[0]
        if cb.relative_high < 0 and ti.relative_high < 0:
            chosen = 'C'
        reasons.append('When both beat A, C advances over B only if both paired-seed and primary temporal intervals support improvement; otherwise B is simpler.')
    text = f'# B1 decisive experiment\n\nRecommendation: **{chosen}**.\n\n' + '\n\n'.join(reasons)
    text += '\n\n' + mix.to_string() + '\n\nScores are location-relative forecast WIS, equal seasons. Lower is better. '
    text += (f'Mixtures pool {members:,} raw members from each of ten fitted distributions ({10 * members:,} draws), before quantiles. '
             'Seed intervals bootstrap paired fits; temporal intervals resample weekly origins in season-stratified moving blocks '
             '(8 weeks primary, 4/12 sensitivity; 2,000 draws). All locations, targets and horizons move together. '
             'Thresholds, block lengths, auxiliary weight .25 and the interval gate are engineering assumptions. '
             'All three seasons informed development; retrospective supplied finals prevent an operational backtest claim. '
             'Freeze the recommended candidate and evaluate future issuances prospectively, with authentic as-of inputs and fit-cutoff labels for operational claims.\n')
    (output / 'README.md').write_text(text)
    print(text, flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    mix.plot.bar(ax=ax)
    ax.set_ylabel('Equal-mixture ensemble-relative WIS (lower is better)')
    ax.tick_params(axis='x', rotation=0)
    fig.tight_layout(); fig.savefig(output / 'mixture-stress.png', dpi=160); plt.close(fig)


def screen(root, output, seed=42):
    """Full-budget single-seed comparison; no across-seed inference or mixture claim."""
    output.mkdir(parents=True, exist_ok=True)
    runs, settings, masks = {}, {}, {}
    for label, experiment in zip('ABC', EXPERIMENTS):
        folder = root / experiment
        done, _ = completed_runs(folder, False, seeds=[seed])
        if len(done) != 1:
            raise ValueError('Single-seed screen requires one configuration per experiment')
        settings[label] = json.loads((folder / 'experiment.json').read_text())
        runs[label] = folder / done[0]['attempt'] / 'b1'
        meta = json.loads((runs[label] / 'manifest.json').read_text())
        prefix = f"{meta['run_id']}-s{seed}"
        for held in SEASONS:
            for stress in MASK_SCENARIOS:
                with np.load(runs[label] / f'eval_{held}/evaluation-masks-{prefix}-{stress}.npz') as archive:
                    key = (held, stress)
                    if key in masks:
                        np.testing.assert_array_equal(masks[key][0], archive['D'])
                        np.testing.assert_array_equal(masks[key][1], archive['issuance_dates'])
                    else:
                        masks[key] = (archive['D'].copy(), archive['issuance_dates'].copy())
    if any(v['input_sha256'] != settings['A']['input_sha256'] for v in settings.values()):
        raise ValueError('Single-seed inputs or frozen support differ')
    from .totals import rank as rank_runs
    member_counts = {evaluation_members(run) for run in runs.values()}
    if len(member_counts) != 1:
        raise ValueError('Single-seed candidates must use a common evaluation trajectory count')
    members = member_counts.pop()
    rank_runs([dict(config_id=label, seed=seed, path=run) for label, run in runs.items()], output)
    records, calibration, uncertainty = [], [], []
    for stress in MASK_SCENARIOS:
        frames = {}
        for label, run in runs.items():
            frame = forecast_cells(run, settings[label]['frozen'], stress).sort_values(CELL_KEYS).reset_index(drop=True)
            if frames and not frame[CELL_KEYS].equals(frames['A'][CELL_KEYS]):
                raise ValueError('Single-seed task keys differ')
            frames[label] = frame
            records.append(dict(candidate=label, seed=seed, stress=stress, objective=scalar(frame)))
            totals = cells_totals(frame)
            calibration.append(season_scores(totals.assign(config_id=label, seed=seed)).assign(stress=stress))
            if stress == 'natural':
                keys = ['target', 'season', 'location', 'horizon']
                prior = pd.read_csv(run / 'totals.csv').sort_values(keys).reset_index(drop=True).sort_index(axis=1)
                current = totals.sort_values(keys).reset_index(drop=True).sort_index(axis=1)
                pd.testing.assert_frame_equal(prior, current, check_dtype=False, rtol=1e-12, atol=1e-10)
        for length in (8, 4, 12):
            uncertainty.extend(dict(row, stress=stress, seed=seed) for row in temporal_intervals(frames, length))
    scores = pd.DataFrame(records)
    scores.to_csv(output / 'stress-scores.csv', index=False)
    pd.concat(calibration).to_csv(output / 'calibration.csv', index=False)
    pd.DataFrame(uncertainty).to_csv(output / 'temporal-uncertainty.csv', index=False)
    recent_diagnostics([runs['C']], output)
    table = scores.pivot(index='stress', columns='candidate', values='objective')
    contrasts = []
    for stress, row in table.iterrows():
        for candidate, baseline in (('B', 'A'), ('C', 'A'), ('C', 'B')):
            contrasts.append(dict(stress=stress, candidate=candidate, baseline=baseline,
                                  difference=row[candidate]-row[baseline], relative_change=row[candidate]/row[baseline]-1))
    pd.DataFrame(contrasts).to_csv(output / 'paired-differences.csv', index=False)
    best = table.loc['natural'].idxmin()
    text = (f'# Full-budget seed {seed} screen\n\nLowest natural forecast point score: **{best}**.\n\n'
            + table.to_string() + f'\n\nOne seed per candidate, all three folds, full training and {members:,} evaluation draws. '
            'A is reused; no reduced-epoch or CPU fits are included. '
            'The natural totals reproduce the ranked scores, and stress masks and frozen task keys match. '
            'Paired differences are one fitted-seed contrast, not an estimate of seed uncertainty. '
            'There is no ten-distribution mixture. Temporal intervals condition on these fitted models, '
            'resampling 8-week blocks with 4/12-week sensitivity; excluded-support counts accompany the intervals. '
            'Use this screen for iteration. The ten-seed promotion decision remains deferred to the overnight run.\n')
    (output / 'README.md').write_text(text)
    print(text, flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    table.plot.bar(ax=ax)
    ax.set_ylabel('Forecast ensemble-relative WIS (lower is better)')
    ax.tick_params(axis='x', rotation=0)
    fig.tight_layout(); fig.savefig(output / 'seed-stress.png', dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/experiments'))
    parser.add_argument('--output', type=Path, default=Path('data/experiments/B1-decisive-report'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--repetitions', type=int, default=2000)
    args = parser.parse_args()
    compare(args.root, args.output, args.device, args.repetitions)


if __name__ == '__main__':
    main()
