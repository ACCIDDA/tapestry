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
from .season_cv import persistence


def evaluate(models, episodes, ds, args, config_id, seed, root, scenario_string=None):
    import pandas as pd
    from .b1_run import sample, task_weights, supervision_mask, known_finals
    from tapestry.evaluation.totals import quantile_scores
    rows = []
    direct = models[0].config['direct']
    hs = slice(2, 6) if direct else slice(None)
    scales = models[0].scale.detach().cpu().numpy()
    for stress in MASK_SCENARIOS:
        dropouts, quantiles, masks, stress_rows, baselines, baseline_masks = [], [], [], [], [], []
        draws = None
        # One episode at a time bounds memory; seed is common across configurations.
        for i, episode in enumerate(episodes):
            samples, d = sample(models, [episode], members=args.evaluation_members,
                seed=seed + i * 101, device=args.device, scenario=stress)
            if draws is None:
                draws = np.lib.format.open_memmap(root / f'draws-{config_id}-s{seed}-{stress}.tmp.npy',
                    mode='w+', dtype=np.float32, shape=(len(episodes), samples.shape[0], *samples.shape[2:]))
            draws[i] = samples[:, 0]
            dropouts.append(d[0])
            truth = episode['Y'][hs, :, 0]
            valid = supervision_mask(episode, d[0])[hs]
            masks.append(valid)
            crps = fair_crps_cells(torch.tensor(samples), torch.tensor(truth[None]),
                                   torch.tensor(valid[None])).numpy()[0]
            q = np.quantile(samples[:, 0], LEVELS, axis=0)
            q[:, :, :3] = np.round(q[:, :, :3])
            quantiles.append(q)
            metrics = quantile_scores(q[:, valid].T, truth[valid]).to_dict('records')
            natural = episode['X'][:, :, 1].astype(bool)
            known = known_finals([episode])[0]
            visible = natural & ~d[0]
            baseline_x = episode['X'].copy()
            baseline_x[:, :, 1] = visible
            base, base_valid = persistence(baseline_x)
            if getattr(args, 'revision_baseline', False):
                baselines.append(episode['X'][-2:, :, 0])
                baseline_masks.append(visible[-2:] & ~known[-2:])
            else:
                baselines.append(base)
                baseline_masks.append(base_valid)
            for (h, c, l), metric in zip(zip(*np.where(valid)), metrics):
                target_day = episode['target_dates'][h + (2 if direct else 0)]
                stress_rows.append(dict(_cell=(i, h, c, l), config_id=config_id, scenario_string=scenario_string, seed=seed, stress=stress,
                    issuance_date=episode['issuance_date'], target_date=target_day,
                    season=season(date.fromisoformat(target_day)), target=CHANNELS[c],
                    location=ds.locations[l], horizon=h - (0 if direct else 2),
                    task='forecast' if direct or h >= 2 else 'nowcast',
                    focal_history_available=bool(natural[:, c, l].any()),
                    visible_focal_history=bool(visible[:, c, l].any()),
                    recent_report_available=bool(natural[-2 + h, c, l] and not known[-2 + h, c, l]) if not direct and h < 2 else None,
                    recent_kind=('artificial_reconstruction' if d[0, -2 + h, c, l] else
                                 'revision' if natural[-2 + h, c, l] and not known[-2 + h, c, l] else
                                 'natural_missing') if not direct and h < 2 else None,
                    observed=float(truth[h, c, l]), loss_scale=float(scales[c, l]),
                    median_error=float(q[len(LEVELS)//2, h, c, l] - truth[h, c, l]),
                    report_ae=(float(abs(episode['X'][-2 + h, c, 0, l] - truth[h, c, l]))
                               if not direct and h < 2 and visible[-2+h, c, l] and not known[-2+h, c, l] else None),
                    crps=float(crps[h, c, l]), **metric,
                    coverage_50=float(q[6, h, c, l] <= truth[h, c, l] <= q[16, h, c, l]),
                    coverage_95=float(q[1, h, c, l] <= truth[h, c, l] <= q[21, h, c, l])))
            if i == len(episodes) // 2 and stress == 'natural':
                path_graph(samples[:, 0], episode, ds, config_id, seed, root, direct)
        draws.flush()
        del draws
        (root / f'draws-{config_id}-s{seed}-{stress}.tmp.npy').replace(root / f'draws-{config_id}-s{seed}-{stress}.npy')
        weights = task_weights(episodes, range(6), direct, np.stack(dropouts))
        if not direct:
            weights *= 2  # Report each task separately.
        for row in stress_rows:
            row['objective_weight'] = float(weights[row.pop('_cell')])
        rows.extend(stress_rows)
        np.savez_compressed(root / f'evaluation-masks-{config_id}-s{seed}-{stress}.npz',
            D=np.stack(dropouts), issuance_dates=[e['issuance_date'] for e in episodes])
        np.savez_compressed(root / f'forecasts-{config_id}-s{seed}-{stress}.npz',
            quantiles=np.stack(quantiles, axis=1), quantile_levels=LEVELS,
            truth=np.stack([e['Y'][hs, :, 0] for e in episodes]),
            mask=np.stack(masks), X_final=known_finals(episodes),
            history_mode=ds.metadata['history_mode'],
            target_dates=np.array([e['target_dates'][hs] for e in episodes]),
            issuance_dates=[e['issuance_date'] for e in episodes], locations=ds.locations,
            channels=CHANNELS, horizons=np.arange(0 if direct else -2, 4),
            baseline_kind='same-week genuine preliminary report' if getattr(args, 'revision_baseline', False) else 'latest-visible-input persistence (reports or supplied finals)',
            baseline=np.stack(baselines), baseline_mask=np.stack(baseline_masks))
    frame = pd.DataFrame(rows)
    frame['geography'] = np.where(frame.location == 'US', 'US', 'states_dc')
    frame.to_parquet(root / f'scores-{config_id}-s{seed}.parquet', index=False)
    frame.groupby(['config_id', 'seed', 'stress', 'task', 'target', 'season', 'geography',
                   'focal_history_available', 'visible_focal_history'], dropna=False).agg(
        cells=('wis', 'size'), wis=('wis', 'mean'), crps=('crps', 'mean'),
        coverage_50=('coverage_50', 'mean'), coverage_95=('coverage_95', 'mean')).reset_index().to_csv(
            root / f'history-stratified-scores-{config_id}-s{seed}.csv', index=False)
    recent = frame[frame.task == 'nowcast']
    if not recent.empty:
        recent.groupby(['config_id', 'seed', 'stress', 'recent_kind', 'target', 'season',
                        'geography', 'location', 'horizon'], dropna=False).agg(
            cells=('wis', 'size'), wis=('wis', 'mean'), crps=('crps', 'mean'),
            median_mae=('ae_median', 'mean'), median_bias=('median_error', 'mean'),
            unchanged_report_mae=('report_ae', 'mean'),
            coverage_50=('coverage_50', 'mean'), coverage_95=('coverage_95', 'mean')).reset_index().to_csv(
                root / f'recent-diagnostics-{config_id}-s{seed}.csv', index=False)
    return rows


def path_graph(samples, episode, ds, config_id, seed, root, direct):
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
        final = episode['X'][-2:, c, 2, l].astype(bool)
        for mask, marker, color, label in ((available & ~final, 'x', 'orange', 'Wednesday report'),
                                          (available & final, 's', 'green', 'Supplied final')):
            ax.scatter(np.array([-2, -1])[mask], episode['X'][-2:, c, 0, l][mask], marker=marker, color=color, label=label)
        ax.axvline(-.5, color='gray', linestyle='--')
        ax.set_title(CHANNELS[c]);ax.set_xlabel('Offset from following Saturday')
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f'{config_id}, seed {seed}, {episode["issuance_date"]}, {ds.locations[l]}: paired sampled paths')
    fig.tight_layout();fig.savefig(root / f'paths-{config_id}-s{seed}.png', dpi=160);plt.close(fig)


def data_audit(ds, root, archive=None):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root);root.mkdir(parents=True, exist_ok=True)
    ds = ds.model_view()
    a = ds.arrays
    sources = np.array([p['source'] for p in ds.metadata['provenance']])
    git_sources = np.array([s.endswith(':git') for s in sources])
    kinds = np.array(['hub' if s.startswith('hub_') else 'delphi' if s.startswith('delphi_') else 'none' for s in sources])
    rows = []
    for i, issuance in enumerate(a['issuance_dates']):
        for c, channel in enumerate(CHANNELS):
            valid = a['X_available'][i, :, c]
            final = a['X_final'][i, :, c]
            provider = kinds[a['X_provenance'][i, :, c]]
            git = git_sources[a['X_provenance'][i, :, c]]
            eligible = np.isin(a['context_dates'][i], ds.calendar_weeks)
            total = int(eligible.sum()) * len(ds.locations)
            rows.append(dict(issuance_date=issuance, target=channel, available=int(valid.sum()), total=total,
                hub=int((valid & ~final & ~git & (provider == 'hub')).sum()),
                git=int((valid & ~final & git).sum()), delphi=int((valid & ~final & (provider == 'delphi')).sum()),
                supplied_final=int(final.sum()), recent_supplied_final=int(final[-2:].sum()),
                missing=int(total - valid.sum()), recent_labels=int(a['Y_recent_valid'][i, :, c].sum()),
                future_labels=int(a['Y_future_valid'][i, :, c].sum())))
    frame = pd.DataFrame(rows);frame.to_csv(root / 'input-source-coverage.csv', index=False)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for c, ax in enumerate(axes.flat):
        f = frame[frame.target == CHANNELS[c]]
        ax.stackplot(pd.to_datetime(f.issuance_date), f.hub/f.total, f.git/f.total, f.delphi/f.total, f.supplied_final/f.total, f.missing/f.total,
                     labels=['Hub as_of', 'Hub Git', 'Delphi', 'Supplied final', 'Missing'], colors=['steelblue', 'purple', 'orange', 'seagreen', 'lightgray'])
        ax.set_title(CHANNELS[c]);ax.set_ylim(0, 1);ax.tick_params(axis='x', rotation=30)
    fig.supylabel('Fraction of context week × location cells')
    fig.suptitle(f'Selected model inputs: up to {a["X_available"].shape[1]} in-calendar weeks × {len(ds.locations)} locations')
    axes.flat[0].legend(fontsize=8);fig.tight_layout();fig.savefig(root / 'input-source-coverage.png', dpi=160);plt.close(fig)
    if archive is not None:
        source_coverage(ds, root, archive)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    audits = []
    for c, ax in enumerate(axes.flat):
        valid = a['X_available'][:, -2:, c] & ~a['X_final'][:, -2:, c] & a['Y_recent_valid'][:, :, c]
        x, y = a['X_values'][:, -2:, c][valid], a['Y_recent'][:, :, c][valid]
        ax.scatter(x, y, s=2, alpha=.15)
        if len(x):
            bounds = [0, max(float(x.max()), float(y.max()))]
            ax.plot(bounds, bounds, color='black', lw=.7)
        ax.set_title(CHANNELS[c]);ax.set_xlabel('Wednesday preliminary');ax.set_ylabel('Reference final')
        audits.append(dict(target=CHANNELS[c], n=int(valid.sum()),
            mean_absolute_revision=float(np.mean(np.abs(x-y))) if len(x) else None))
    fig.tight_layout();fig.savefig(root / 'preliminary-vs-final.png', dpi=160);plt.close(fig)
    support = history_support(ds)
    pd.DataFrame(support).to_csv(root / 'history-support.csv', index=False)
    (root / 'audit.json').write_text(json.dumps(dict(truth_cutoff=ds.metadata['truth_cutoff'],
        history_mode=ds.metadata['history_mode'], finality_assumption=ds.metadata['finality_assumption'],
        recent_supplied_finals=int(a['X_final'][:, -2:].sum()),
        usable_episodes=sum(1 for _ in ds.episodes()), revision_audit=audits,
        calendar_weeks=list(ds.calendar_weeks), history_support=support,
        archive_issuances=ds.metadata.get('archive_issuances'), model_issuances=len(a['issuance_dates']),
        coverage_policy=ds.metadata['coverage_policy']), indent=2) + '\n')


def source_coverage(ds, root, archive):
    """Source information states, before provider selection or final filling."""
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    from tapestry.model_data.wednesday import START, RSV_ED_START
    a = ds.arrays
    eligible = np.isin(a['context_dates'], ds.calendar_weeks)
    valid, final = a['X_available'], a['X_final']
    expected = {(p['source'], p['snapshot_id'], p['manifest_sha256']) for p in ds.metadata['source_manifests']}
    actual = {(p['source'], p['snapshot_id'], p['manifest_sha256']) for p in archive.manifests}
    if expected != actual:
        raise ValueError('Archive acquisitions differ from the dataset; rebuild or use its pinned sources')
    masks = {key: np.zeros_like(valid) for key in ('hub', 'git', 'delphi')}
    selected_available = np.zeros_like(valid)
    selected_values = np.zeros_like(a['X_values'])
    selected_provenance = np.zeros_like(a['X_provenance'])
    truth_state = archive.resolve(ds.metadata['truth_cutoff'])
    mismatched_input_cells = 0
    for i, issuance in enumerate(a['issuance_dates']):
        state = archive.resolve(str(issuance))
        values, available, provenance, _ = archive.panel(a['context_dates'][i], ds.locations, state)
        selected_values[i], selected_available[i], selected_provenance[i] = values, available, provenance
        finals, final_available, _, _ = archive.panel(a['context_dates'][i], ds.locations, truth_state)
        use_final = np.ones_like(available);use_final[-2:] = ~available[-2:]
        expected_values = np.where(use_final, finals, values)
        expected_available = np.where(use_final, final_available, available)
        expected_final = use_final & final_available
        expected_values[~eligible[i]] = 0
        expected_available[~eligible[i]] = False
        expected_final[~eligible[i]] = False
        mismatched_input_cells += int(np.count_nonzero(
            (expected_values != a['X_values'][i]) | (expected_available != valid[i]) | (expected_final != final[i])))
        for key, observations in zip(masks, (state[0], state[4], state[3])):
            for t, day in enumerate(a['context_dates'][i]):
                if not eligible[i, t]:
                    continue
                for c in range(6):
                    if day < START or (c == 5 and day < RSV_ED_START):
                        continue
                    for l, location in enumerate(ds.locations):
                        value, _ = observations.get((str(day), c, location), (None, 0))
                        masks[key][i, t, c, l] = value is not None
    # Verify the stored dataset against independently reconstructed eligible states.
    # Older context intentionally uses finals; recent existing versions must survive.
    recent = np.zeros_like(valid);recent[:, -2:] = eligible[:, -2:, None, None]
    selected_available &= eligible[:, :, None, None]
    stored_reports = valid & ~final
    expected_reports = selected_available & recent
    mismatched_masks = int(np.count_nonzero(stored_reports != expected_reports))
    mismatched_values = int(np.count_nonzero(stored_reports & expected_reports & (a['X_values'] != selected_values)))
    releases = np.array([p['release'] or '' for p in archive.provenance])
    released_days = np.char.partition(releases[selected_provenance], 'T')[..., 0]
    unchanged = expected_reports & (released_days < a['issuance_dates'][:, None, None, None])
    check = dict(semantics='Latest eligible information state persists without a new revision; supplied finals are separate.',
                 stored_recent_report_cells=int(stored_reports.sum()),
                 recent_report_cells_from_earlier_releases=int(unchanged.sum()),
                 mismatched_report_masks=mismatched_masks, mismatched_report_values=mismatched_values,
                 mismatched_input_cells=mismatched_input_cells)
    (root / 'version-state-dataset-check.json').write_text(json.dumps(check, indent=2) + '\n')
    if mismatched_masks or mismatched_values or mismatched_input_cells:
        raise ValueError('Stored dataset disagrees with reconstructed source versions; see version-state-dataset-check.json')
    source_available = np.logical_or.reduce(list(masks.values()))
    counts = {key: mask.sum(axis=-1) for key, mask in masks.items()}
    report_count = source_available.sum(axis=-1)
    available_count = valid.sum(axis=-1)
    weekly_rows, location_rows, geographic_audit = [], [], []
    for c, channel in enumerate(CHANNELS):
        for i, t in np.argwhere(eligible):
            weekly_rows.append(dict(issuance_date=a['issuance_dates'][i],
                context_date=a['context_dates'][i, t], context_offset=t-valid.shape[1], target=channel,
                locations=len(ds.locations), model_input_locations=int(available_count[i, t, c]),
                version_locations=int(report_count[i, t, c]),
                **{key: int(n[i, t, c]) for key, n in counts.items()}))
        for name, n in [('model_input', available_count[:, :, c]), ('available_source_version', report_count[:, :, c])]:
            present = eligible & (n > 0)
            geographic_audit.append(dict(target=channel, kind=name,
                present_issuance_weeks=int(present.sum()),
                partial_location_issuance_weeks=int((present & (n < len(ds.locations))).sum()),
                minimum_locations_when_present=int(n[present].min()) if present.any() else None))
        for i, t in np.argwhere(eligible):
            if t < valid.shape[1] - 2:
                continue
            for l, location in enumerate(ds.locations):
                location_rows.append(dict(issuance_date=a['issuance_dates'][i],
                    context_date=a['context_dates'][i, t], context_offset=t-valid.shape[1],
                    target=channel, location=location, model_supplied_final=int(final[i, t, c, l]),
                    version_available=int(source_available[i, t, c, l]),
                    **{key: int(mask[i, t, c, l]) for key, mask in masks.items()}))
    pd.DataFrame(weekly_rows).to_csv(root / 'source-coverage-weeks.csv', index=False)
    location_frame = pd.DataFrame(location_rows)
    location_frame.to_csv(root / 'source-coverage-locations.csv', index=False)
    (root / 'location-coverage-check.json').write_text(json.dumps(geographic_audit, indent=2) + '\n')
    # Each bit marks a source with an eligible version, unchanged or revised.
    # Source overlap is measured before selecting a provider or supplying finals.
    labels = ['No available version', 'Hub as_of', 'Hub Git',
              'Hub as_of + Git', 'Delphi', 'Hub as_of + Delphi',
              'Hub Git + Delphi', 'Hub as_of + Git + Delphi']
    colors = ['lightgray', 'steelblue', 'purple', 'teal', 'orange', 'crimson', 'olive', 'black']
    cmap = ListedColormap(colors);cmap.set_bad('white')
    norm = BoundaryNorm(np.arange(-.5, len(colors)), len(colors))
    report_codes = sum(bit * masks[key].astype(np.uint8)
                       for bit, key in ((1, 'hub'), (2, 'git'), (4, 'delphi')))
    week_codes = np.bitwise_or.reduce(report_codes, axis=-1)
    summary = []
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True)
    for c, ax in enumerate(axes.flat):
        total = eligible.sum(axis=1) * len(ds.locations)
        shares = [((report_codes[:, :, c] == k) & eligible[:, :, None]).sum(axis=(1, 2)) / total
                  for k in range(8)]
        ax.stackplot(pd.to_datetime(a['issuance_dates']), *shares, colors=colors, labels=labels)
        ax.set_title(CHANNELS[c]);ax.set_ylim(0, 1);ax.tick_params(axis='x', rotation=30)
        for i, issuance in enumerate(a['issuance_dates']):
            for k, label in enumerate(labels):
                summary.append(dict(issuance_date=issuance, target=CHANNELS[c], source_combination=label,
                                    total=int(total[i]), fraction=float(shares[k][i])))
    pd.DataFrame(summary).to_csv(root / 'source-coverage.csv', index=False)
    fig.supylabel('Fraction of eligible context week × location cells')
    fig.suptitle('Available source versions, including unchanged values; before model-input selection')
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc='lower center', ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, .1, 1, .95));fig.savefig(root / 'source-coverage.png', dpi=160);plt.close(fig)
    # Keep real elapsed time, including gaps between model-calendar issuances.
    dates = pd.date_range(str(a['issuance_dates'][0]), str(a['issuance_dates'][-1]), freq='7D')
    positions = dates.get_indexer(pd.to_datetime(a['issuance_dates']))
    ticks = np.unique(np.linspace(0, len(dates)-1, 7).astype(int))
    handles = [Patch(color=color, label=label) for color, label in zip(colors, labels)]
    handles.append(Patch(facecolor='white', edgecolor='gray', label='Outside model calendar'))
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True)
    for c, ax in enumerate(axes.flat):
        states = np.ma.masked_all((valid.shape[1], len(dates)))
        states[:, positions] = np.ma.array(week_codes[:, :, c].T, mask=~eligible.T)
        ax.imshow(states, aspect='auto', interpolation='nearest', cmap=cmap, norm=norm)
        ax.set_title(CHANNELS[c]);ax.set_xticks(ticks, dates[ticks].strftime('%Y-%m-%d'), rotation=30, ha='right')
        ax.set_yticks(np.arange(valid.shape[1]), np.arange(-valid.shape[1], 0))
    fig.supylabel('Context week offset from following Saturday')
    fig.supxlabel('Wednesday issuance', y=.10)
    fig.suptitle('Available source versions by week (unchanged versions remain available)\nGray = no eligible source value; distinct colors show source overlap before model-input selection')
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, .13, 1, .93));fig.savefig(root / 'source-coverage-weeks.png', dpi=160);plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(18, 15), sharex=True, sharey=True)
    for c, ax in enumerate(axes.flat):
        states = np.ma.masked_all((len(ds.locations) * 2, len(dates)))
        # Each location has separate rows for -2 and -1, avoiding any averaging
        # or apparent source overlap between different recent weeks.
        recent = report_codes[:, -2:, c].transpose(2, 1, 0).reshape(len(ds.locations) * 2, -1)
        excluded = np.broadcast_to(~eligible[:, -2:].T[None],
                                   (len(ds.locations), 2, len(positions))).reshape(recent.shape)
        states[:, positions] = np.ma.array(recent, mask=excluded)
        ax.imshow(states, aspect='auto', interpolation='nearest', cmap=cmap, norm=norm)
        ax.set_title(CHANNELS[c]);ax.set_xticks(ticks, dates[ticks].strftime('%Y-%m-%d'), rotation=30, ha='right')
        ax.set_yticks(np.arange(len(ds.locations)) * 2 + .5, ds.locations, fontsize=7)
        ax.set_yticks(np.arange(len(ds.locations) + 1) * 2 - .5, minor=True)
        ax.grid(axis='y', which='minor', color='white', linewidth=.3)
        ax.tick_params(axis='y', which='minor', length=0, labelleft=True)
        ax.tick_params(axis='y', which='major', labelleft=True)
    fig.supylabel('Location (two rows each: older recent week −2 above, latest week −1 below)')
    fig.supxlabel('Wednesday issuance', y=.065)
    fig.suptitle('Available source versions by location over time — recent weeks −2 and −1\nUnchanged versions persist; gray = no eligible source value; white = outside model calendar')
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, .085, 1, .95));fig.savefig(root / 'source-coverage-locations.png', dpi=160);plt.close(fig)


def history_support(ds):
    """Distinguish reference labels, supplied answers, and supervised cells."""
    from .provenance import SEASONS
    ds = ds.model_view()
    a = ds.arrays
    x = a['X_available']
    reference = np.concatenate((a['Y_recent_valid'], a['Y_future_valid']), axis=1)
    y = reference.copy()
    y[:, :2] &= ~a['X_final'][:, -2:]
    seasons = np.array([[season(date.fromisoformat(str(d))) for d in row] for row in a['target_dates']])
    has_input = x.any((1, 2, 3))
    rows = []
    for label in SEASONS:
        for c, channel in enumerate(CHANNELS):
            for task, hs in (('nowcast', slice(0, 2)), ('forecast', slice(2, 6))):
                in_season = (seasons[:, hs] == label)[:, :, None]
                valid = reference[:, hs, c] & in_season
                # Same episode eligibility as the model, before task-specific label filtering.
                eligible = y[:, hs, c] & in_season & has_input[:, None, None]
                focal = x[:, :, c].any(1)[:, None, :]
                recent = x[:, -2:, c] & ~a['X_final'][:, -2:, c] if task == 'nowcast' else np.zeros_like(eligible)
                observed = eligible & recent
                first = np.flatnonzero(x[:, :, c].any((1, 2)))
                rows.append(dict(season=label, target=channel, task=task,
                    reference_label_cells=int(valid.sum()), model_eligible_cells=int(eligible.sum()),
                    supplied_final_cells=int((valid & a['X_final'][:, -2:, c]).sum()) if task == 'nowcast' else 0,
                    with_focal_history=int((eligible & focal).sum()),
                    without_focal_history=int((eligible & ~focal).sum()),
                    observed_recent_report=int(observed.sum()) if task == 'nowcast' else None,
                    missing_recent_report_with_older_history=int((eligible & ~recent & focal).sum()) if task == 'nowcast' else None,
                    missing_history_percent=100 * float((eligible & ~focal).sum()) / int(eligible.sum()) if eligible.any() else None,
                    first_wednesday_with_focal_history=str(a['issuance_dates'][first[0]]) if len(first) else None))
    return rows


def report(rows, ds, root):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    data_audit(ds, root)
    frame = pd.DataFrame(rows)
    # Compare future cells across all configurations present for each seed; nowcasts
    # only among models that produce them. Partial manager reports can have
    # different numbers of finished configurations for each seed.
    keys = ['seed', 'stress', 'issuance_date', 'target_date', 'target', 'location']
    for _, seed_frame in frame.groupby('seed'):
        for recent in (False, True):
            f = seed_frame[seed_frame.horizon.lt(0) if recent else seed_frame.horizon.ge(0)]
            if not (f.groupby(keys).config_id.nunique() == f.config_id.nunique()).all():
                raise ValueError('Compared configurations do not have identical evaluation support')
    frame['geography'] = np.where(frame.location == 'US', 'US', 'states_dc')
    summary = frame.groupby(['config_id', 'stress', 'target', 'season', 'geography', 'horizon']).agg(
        n=('crps', 'size'), crps=('crps', 'mean'), wis=('wis', 'mean'),
        coverage_50=('coverage_50', 'mean'), coverage_95=('coverage_95', 'mean')).reset_index()
    summary.to_csv(root / 'summary.csv', index=False)
    # Separate the four stress scenarios so a formulation grid does not create
    # dozens of overlapping curves/legend entries inside each target panel.
    for stress in summary.stress.unique():
        for metric in ('crps', 'wis', 'coverage_50', 'coverage_95'):
            fig, axes = plt.subplots(2, 3, figsize=(14, 10), sharex=True)
            for c, ax in enumerate(axes.flat):
                subset = summary[(summary.target == CHANNELS[c]) & (summary.stress == stress)]
                for config_id, f in subset.groupby('config_id'):
                    curve = f.groupby('horizon')[metric].mean()
                    label = config_id.rsplit('-', 1)[0].replace('multiscale_conv', 'multi').replace('two_stage', 'two').replace('mask', 'm')
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
        'future offsets 0–3 are forecasts. All configurations use identical eligible future cells and fixed stress masks. '
        'Scores are native-unit diagnostics, with states/DC and US reported separately. Plot curves average the '
        'reported season/geography groups; they are not the scientific selection objective. '
        'This native report is separate from the optional manager Hub benchmark and its narrower support. '
        'Inputs include reference-final older history and flagged recent-final fallback. Visible supplied recent finals '
        'are excluded from nowcast scoring. Results measure retrospective conditional forecasting, not operational skill.\n')


if __name__ == '__main__':
    import argparse
    from tapestry.model_data.wednesday import DEFAULT_DATASET, WednesdayDataset, read_archive
    parser = argparse.ArgumentParser(description='Audit B1 source coverage and preliminary-to-final revisions')
    parser.add_argument('--dataset', default=DEFAULT_DATASET)
    parser.add_argument('--output', default='data/processed/b1-audit')
    parser.add_argument('--data-root', default='data', help='Pinned source archives for information-state coverage')
    args = parser.parse_args()
    data_audit(WednesdayDataset.load(args.dataset), args.output, archive=read_archive(args.data_root))
