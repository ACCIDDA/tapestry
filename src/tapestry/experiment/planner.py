"""Plan, run, status, rank and compare experiments for the unified `Scenario`.

Collapses `models/manager.py` + `models/backends.py`: with one `Scenario` and one
`Model`, there is no per-model branching left, only one leave-one-season-out
training path (`fit`) shared by every scenario. Keeps the CLI shape (`plan`,
`run`, `status`, `rank`) from the old manager; `compare` (EpiBench/R-only) has
no replacement -- `rank`'s WIS-vs-frozen-ensemble ratio is the only comparison.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import torch

from tapestry.dataset import splits
from tapestry.dataset.build import (load as load_dataset, covariate_names_for, COVARIATE_GROUPS,
                                     STATE_COVARIATE_NAMES, NATIONAL_COVARIATE_NAMES, FINALIZED_DATASET,
                                     VINTAGED_DATASET)
from tapestry.model.network import Model, fair_crps_cells, draw_dropout, IndependentBundle, checkpoint, load_model
from tapestry.model.objective import LOSS_WEIGHTS, loss_cell_weights, loss_scales, LOSS_DEFINITION, US_WEIGHT
from tapestry.model.scenario import Scenario
from .provenance import SEASONS, save, now, environment, git_state

GROUPS = {'all': [list(range(6))], 'pathogen': [[0, 3], [1, 4], [2, 5]], 'target': [[i] for i in range(6)]}
LEVELS = np.array([.01, .025, .05, .10, .15, .20, .25, .30, .35, .40, .45, .50,
                   .55, .60, .65, .70, .75, .80, .85, .90, .95, .975, .99])
JOB_FIELDS = ['task', 'name', 'scenario', 'seeds']
ARRAY_CHUNK = 1000
FROZEN = 'data/evaluation/b0_hub_comparison_q23'
LOCATIONS = 'data/metadata/locations.csv'


def calendar(days, dynamics=True):
    phase = np.array([date.fromisoformat(day).timetuple().tm_yday for day in days]) * (2 * np.pi / 365.25)
    columns = [np.sin(phase), np.cos(phase)]
    if dynamics:
        dates = [date.fromisoformat(day) for day in days]
        columns.append(np.array([(d - date(d.year if d.month >= 7 else d.year - 1, 12, 25)).days / (7 * 26) for d in dates]))
    return np.stack(columns, axis=-1).astype('float32')


def populations(path, locations):
    values = {}
    with open(path) as stream:
        for row in csv.DictReader(stream):
            loc, value = row.get('abbreviation') or row['location'], float(row['population'])
            if loc in values or not np.isfinite(value) or value <= 0:
                raise ValueError(f'Invalid or duplicate population for {loc}')
            values[loc] = value
    return {loc: values[loc] for loc in locations}


def assemble_covariates(arrays, covariate_names, indices=None):
    """Per-location [.., K, L] value/available panel for the requested covariate names.

    `covariates`/`covariate_mask` are already per-location; a national-only name
    (currently `kinsa_ili`) is broadcast from `covariates_national`, available
    only at the `US` column, exactly as the pre-restructuring B2 arrays did.
    """
    state_names = list(arrays['covariate_names'])
    national_names = list(arrays['covariate_national_names'])
    locations = list(arrays['locations'])
    us = locations.index('US') if 'US' in locations else None
    leading = arrays['covariates'].shape[:-2]
    values = np.zeros((*leading, len(covariate_names), len(locations)), np.float32)
    available = np.zeros_like(values, dtype=bool)
    for k, name in enumerate(covariate_names):
        if name in state_names:
            j = state_names.index(name)
            values[..., k, :] = arrays['covariates'][..., j]
            available[..., k, :] = arrays['covariate_mask'][..., j]
        elif name in national_names and us is not None:
            j = national_names.index(name)
            values[..., k, us] = arrays['covariates_national'][..., j]
            available[..., k, us] = ~np.isnan(arrays['covariates_national'][..., j])
        elif name in national_names:
            pass  # No US column in this location set: stays unavailable everywhere.
        else:
            raise ValueError(f'Unknown covariate: {name}')
    return values, available


def _channel_first(panel):
    """`[.., L, C]` (the array-on-disk convention) -> `[.., C, L]` (Model's convention)."""
    return np.moveaxis(panel, -1, -2)


def episodes_from_finalized(arrays, lookback, horizons=(1, 2, 3, 4), covariate_names=()):
    """One episode per Saturday origin in a flat `[T, L, 6]` truth panel."""
    dates = np.array([str(d) for d in arrays['dates']])
    targets = _channel_first(arrays['targets'])
    known_final = _channel_first(arrays['known_final'])
    cov_values, cov_available = (assemble_covariates(arrays, covariate_names) if covariate_names else (None, None))
    for end in range(lookback - 1, len(dates) - max(horizons)):
        context = list(range(end - lookback + 1, end + 1))
        future = [end + h for h in horizons]
        available = ~np.isnan(targets[context])
        if not available.any():
            continue
        target_available = ~np.isnan(targets[future])
        if not target_available.any():
            continue
        episode = dict(values=np.nan_to_num(targets[context]), available=available,
                       known_final=known_final[context],
                       target_values=np.nan_to_num(targets[future]), target_available=target_available,
                       context_dates=tuple(dates[context]), target_dates=tuple(dates[future]),
                       locations=tuple(arrays['locations']))
        episode['Y'] = np.stack((episode['target_values'], episode['target_available']), axis=2)
        if covariate_names:
            episode['covariates'] = np.stack((cov_values[context], cov_available[context]), axis=-2)
        yield episode


def episodes_from_vintaged(arrays, lookback, horizons=(1, 2, 3, 4), covariate_names=()):
    """One episode per historical Wednesday issuance; window already materialized.

    The array's context window is fixed at build time (`metadata['lookback']`).
    A scenario may ask for a shorter lookback than was built; it then reads the
    most recent `lookback` context weeks of that fixed window, keeping the
    horizon slice anchored at the built lookback so target weeks never shift.
    """
    built_lookback = json.loads(str(arrays['metadata']))['lookback']
    if lookback > built_lookback:
        raise ValueError(f'Scenario lookback {lookback} exceeds the built vintaged lookback {built_lookback}')
    targets = _channel_first(arrays['targets'])
    known_final = _channel_first(arrays['known_final'])
    cov_values, cov_available = (assemble_covariates(arrays, covariate_names) if covariate_names else (None, None))
    for i in range(len(arrays['issuance_dates'])):
        context = slice(built_lookback - lookback, built_lookback)
        future = slice(built_lookback, built_lookback + len(horizons))
        available = ~np.isnan(targets[i, context])
        if not available.any():
            continue
        target_available = ~np.isnan(targets[i, future])
        if not target_available.any():
            continue
        episode = dict(values=np.nan_to_num(targets[i, context]), available=available,
                       known_final=known_final[i, context],
                       target_values=np.nan_to_num(targets[i, future]), target_available=target_available,
                       context_dates=tuple(str(d) for d in arrays['dates'][i, context]),
                       target_dates=tuple(str(d) for d in arrays['dates'][i, future]),
                       locations=tuple(arrays['locations']))
        episode['Y'] = np.stack((episode['target_values'], episode['target_available']), axis=2)
        if covariate_names:
            episode['covariates'] = np.stack((cov_values[i, context], cov_available[i, context]), axis=-2)
        yield episode


def episodes(scenario, dataset_root='data/processed'):
    """All usable episodes for a scenario's `input_mode`, plus their calendar labels."""
    covariate_names = covariate_names_for(scenario.covariate_set)
    if scenario.input_mode == 'finalized':
        arrays = load_dataset(Path(dataset_root) / 'finalized.npz')
        eps = list(episodes_from_finalized(arrays, scenario.lookback, covariate_names=covariate_names))
    else:
        arrays = load_dataset(Path(dataset_root) / 'vintaged.npz')
        eps = list(episodes_from_vintaged(arrays, scenario.lookback, covariate_names=covariate_names))
    if not eps:
        raise ValueError(f'No usable episodes for input_mode={scenario.input_mode!r}')
    return eps


def to_tensors(batch, device):
    values = torch.as_tensor(np.stack([e['values'] for e in batch]), device=device)
    available = torch.as_tensor(np.stack([e['available'] for e in batch]), device=device)
    known_final = torch.as_tensor(np.stack([e['known_final'] for e in batch]), device=device)
    y = torch.as_tensor(np.stack([e['target_values'] for e in batch]), device=device)
    y_mask = torch.as_tensor(np.stack([e['target_available'] for e in batch]), device=device)
    cal = torch.as_tensor(calendar([e['context_dates'][-1] for e in batch]), device=device)
    covariates = (torch.as_tensor(np.stack([e['covariates'] for e in batch]), device=device)
                  if 'covariates' in batch[0] else None)
    return values, available, known_final, y, y_mask, cal, covariates


def covariate_scales(batch):
    """Per-covariate/location mean and SD from observed fitting cells only."""
    c = np.stack([e['covariates'] for e in batch])
    values, available = c[..., 0, :], c[..., 1, :].astype(bool)
    support = available.sum(axis=(0, 1))
    total = np.where(available, values, 0.).sum(axis=(0, 1))
    offset = np.divide(total, support, out=np.zeros_like(total), where=support > 0)
    squared = np.where(available, (values - offset[None, None]) ** 2, 0.).sum(axis=(0, 1))
    variance = np.divide(squared, support, out=np.zeros_like(squared), where=support > 0)
    scale = np.sqrt(variance)
    scale = np.where(scale > 1e-6, scale, 1.)
    return offset.tolist(), scale.tolist(), (support > 0).tolist()


def model_options(batch, scenario, pop):
    covariate_names = covariate_names_for(scenario.covariate_set)
    options = dict(populations=pop, location_ids=list(batch[0]['locations']), lookback=scenario.lookback,
                   width=scenario.width, latent=scenario.latent, covariate_names=list(covariate_names),
                   **scenario.model_options())
    if covariate_names:
        offset, scale, trained = covariate_scales(batch)
        options.update(covariate_offset=offset, covariate_scale=scale, covariate_trained=trained)
    return options


def fit_component(train, validation, channels, component, options, scenario, seed, device, epochs=None):
    seed = seed + 10000 * component
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    selecting = validation is not None
    budget = scenario.epochs if epochs is None else epochs
    model = Model(horizons=(1, 2, 3, 4), scale=loss_scales(unique_truth(train)), **options).to(device)
    weights_by_channel = LOSS_WEIGHTS[scenario.loss_weights]
    values, available, known_final, y, y_mask, cal, cov = to_tensors(train, device)
    weights = torch.as_tensor(loss_cell_weights(train, weights_by_channel), device=device)[:, :, channels]
    if selecting:
        vvalues, vavailable, vknown_final, vy, vy_mask, vcal, vcov = to_tensors(validation, device)
        vweights = torch.as_tensor(loss_cell_weights(validation, weights_by_channel), device=device)[:, :, channels]
    optimizer = torch.optim.Adam(model.parameters(), lr=scenario.lr, weight_decay=scenario.weight_decay)
    best, best_state, best_epoch = float('inf'), None, 0
    history = []
    for epoch in range(budget):
        dropout = None
        a = available.cpu().numpy()
        if scenario.mask_rate:
            dropout = torch.as_tensor(draw_dropout(a, rng, scenario.mask_probabilities), device=device)
        model.train()
        total = 0.
        for ids in torch.randperm(len(train), device=device).split(scenario.batch_size):
            optimizer.zero_grad()
            visible = available[ids] if dropout is None else available[ids] & ~dropout[ids]
            samples = model(values=values[ids], available=visible, calendar=cal[ids], members=scenario.members,
                            known_final=known_final[ids], covariates=None if cov is None else cov[ids],
                            locations=list(train[0]['locations']), vintaged=scenario.input_mode == 'vintaged')
            score = fair_crps_cells(samples[:, :, :, channels], y[ids][:, :, channels], y_mask[ids][:, :, channels])
            loss = (weights[ids] * score / model.scale[channels]).sum() * len(train) / len(ids)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += float(loss.detach()) * len(ids) / len(train)
        val = None
        if selecting:
            model.eval()
            val = 0.
            with torch.no_grad():
                for ids in torch.arange(len(validation), device=device).split(scenario.batch_size):
                    samples = model(values=vvalues[ids], available=vavailable[ids], calendar=vcal[ids],
                                    members=scenario.validation_members, known_final=vknown_final[ids],
                                    covariates=None if vcov is None else vcov[ids],
                                    locations=list(train[0]['locations']), vintaged=scenario.input_mode == 'vintaged')
                    score = fair_crps_cells(samples[:, :, :, channels], vy[ids][:, :, channels], vy_mask[ids][:, :, channels])
                    val += float((vweights[ids] * score / model.scale[channels]).sum())
            if val < best:
                best, best_epoch, best_state = val, epoch + 1, {k: v.detach().clone() for k, v in model.state_dict().items()}
        history.append(dict(epoch=epoch + 1, loss=total, validation_loss=val))
        print(json.dumps(dict(component=component, phase='select' if selecting else 'refit', **history[-1])), flush=True)
        if selecting and scenario.patience and epoch + 1 - best_epoch >= scenario.patience:
            break
    if selecting and scenario.patience and best_state is not None:
        model.load_state_dict(best_state)
    record = dict(channels=channels, best_epoch=best_epoch,
                 selected_epoch=(best_epoch if scenario.patience else scenario.epochs) if selecting else budget,
                 phase='select' if selecting else 'refit', epochs=budget, history=history,
                 fitting_episodes=len(train), validation_episodes=len(validation) if selecting else 0)
    return model.cpu(), record


def unique_truth(episodes_):
    """One observation per calendar date, pooled across overlapping episodes."""
    by_date = {}
    for e in episodes_:
        for values, avail, day in zip(e['values'], e['available'], e['context_dates']):
            if day not in by_date:
                by_date[day] = (values, avail)
        for values, avail, day in zip(e['target_values'], e['target_available'], e['target_dates']):
            if day not in by_date:
                by_date[day] = (values, avail)
    panel = np.zeros((len(by_date), 6, 2, len(episodes_[0]['locations'])), np.float32)
    for i, (values, avail) in enumerate(by_date.values()):
        panel[i, :, 0] = values
        panel[i, :, 1] = avail
    return panel


def fit(scenario, seed, held_out_season, eval_members, device, output, dataset_root='data/processed'):
    """One leave-one-season-out fold: fit, evaluate the held-out season, save."""
    eps = episodes(scenario, dataset_root)
    dates = [e['context_dates'][-1] for e in eps]
    split = splits.season_split(dates, held_out_season)
    train_eps = [e for e, keep in zip(eps, split.train) if keep]
    val_eps = [e for e, keep in zip(eps, split.val) if keep] or None
    score_eps = [e for e, keep in zip(eps, split.score) if keep]
    if not train_eps or not score_eps:
        raise ValueError(f'No usable training/scoring episodes for held-out {held_out_season}')
    pop = populations(LOCATIONS, train_eps[0]['locations'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    groups = GROUPS[scenario.fit_partition]
    models, records = [], []
    for i, channels in enumerate(groups):
        options = model_options(train_eps, scenario, pop)
        selected = scenario.epochs
        if scenario.patience and val_eps:
            _, record = fit_component(train_eps, val_eps, channels, i, options, scenario, seed, device)
            records.append(record)
            selected = record['selected_epoch']
        model, record = fit_component(train_eps, None, channels, i, options, scenario, seed, device, epochs=selected)
        models.append(model)
        records.append(record)
    model = models[0] if scenario.fit_partition == 'all' else IndependentBundle(models, groups)
    metadata = dict(scenario=scenario.scenario_string, run_id=scenario.run_id, config=asdict(scenario),
                    seed=seed, held_out_season=held_out_season, groups=groups,
                    protocol='season_split_refit_v1', records=records,
                    loss=LOSS_DEFINITION, us_weight=US_WEIGHT, channels=list(load_dataset(
                        Path(dataset_root) / ('vintaged.npz' if scenario.input_mode == 'vintaged' else 'finalized.npz'))['target_names']),
                    locations=list(train_eps[0]['locations']), eval_members=eval_members, **environment())
    torch.save(checkpoint(model, metadata), output / 'model.pt')
    save(output / 'manifest.json', metadata)
    model.to(device)
    scores = evaluate(model, score_eps, eval_members, device, output)
    metadata['evaluation_scores'] = len(scores)
    save(output / 'manifest.json', metadata)
    return output / 'model.pt'


def evaluate(model, eps, eval_members, device, output, name=''):
    model.eval()
    quantiles, truths, masks = [], [], []
    for e in eps:
        values = torch.as_tensor(e['values'][None], device=device)
        available = torch.as_tensor(e['available'][None], device=device)
        known_final = torch.as_tensor(e['known_final'][None], device=device)
        cal = torch.as_tensor(calendar([e['context_dates'][-1]]), device=device)
        cov = torch.as_tensor(e['covariates'][None], device=device) if 'covariates' in e else None
        with torch.no_grad():
            samples = torch.cat([model(values=values, available=available, calendar=cal,
                                       members=min(32, eval_members - j), known_final=known_final,
                                       covariates=cov, locations=list(e['locations']), vintaged=True).cpu()
                                 for j in range(0, eval_members, 32)], dim=0).numpy()[:, 0]
        q = np.quantile(samples, LEVELS, axis=0)
        q[:, :, :3] = np.floor(q[:, :, :3] + .5)
        quantiles.append(q)
        truths.append(e['target_values'])
        masks.append(e['target_available'])
    q = np.stack(quantiles, axis=1)
    y, mask = np.stack(truths), np.stack(masks)
    np.savez_compressed(output / f'{name}forecasts.npz', quantiles=q, quantile_levels=LEVELS,
                        truth=y, mask=mask, context_end=[e['context_dates'][-1] for e in eps],
                        target_dates=[e['target_dates'] for e in eps], locations=eps[0]['locations'])
    return [dict(context_end=e['context_dates'][-1]) for e in eps]


def scenario_directory(value):
    return Scenario.from_string(value).run_id


def read_jobs(folder):
    if not (folder / 'jobs.csv').is_file():
        raise ValueError(f'No jobs.csv in {folder}; run plan first')
    with (folder / 'jobs.csv').open() as stream:
        return [dict(row, task=int(row['task']), seeds=[int(s) for s in row['seeds'].split()])
                for row in csv.DictReader(stream)]


def write_csv(path, rows, fieldnames):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_jobs(folder, jobs):
    write_csv(folder / 'jobs.csv', [dict(job, seeds=' '.join(map(str, job['seeds']))) for job in jobs], JOB_FIELDS)


def plan(folder, scenarios, seeds, settings):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'experiment.json'
    previous = json.loads(path.read_text()) if path.exists() else {}
    changed = sorted(k for k, v in previous.items() if k in settings and settings[k] != v)
    protocol = set(changed) - {'device'}
    if protocol:
        raise ValueError(f'Experiment settings changed: {sorted(protocol)}; use a new experiment name')
    save(path, {**previous, **settings})
    jobs = read_jobs(folder) if (folder / 'jobs.csv').exists() else []
    by_scenario = {job['scenario']: job for job in jobs}
    for name, scenario in scenarios.items():
        key = scenario.scenario_string
        if key not in by_scenario:
            by_scenario[key] = dict(task=len(jobs), name=name, scenario=key, seeds=[])
            jobs.append(by_scenario[key])
        by_scenario[key]['seeds'] = sorted(set(by_scenario[key]['seeds']) | set(seeds))
    write_jobs(folder, jobs)
    return jobs


def attempts(folder, scenario, seed):
    return sorted((folder / scenario_directory(scenario) / f's{seed}').glob('attempt-*'))


def complete_artifacts(output):
    required = ['manifest.json'] + [f'eval_{s}/model.pt' for s in SEASONS] + [f'eval_{s}/forecasts.npz' for s in SEASONS]
    if not all((output / n).is_file() and (output / n).stat().st_size for n in required):
        return False
    try:
        manifest = json.loads((output / 'manifest.json').read_text())
        return sorted(manifest.get('folds', [])) == sorted(SEASONS)
    except (KeyError, ValueError, TypeError):
        return False


def seed_state(folder, scenario, seed):
    found = []
    for attempt in attempts(folder, scenario, seed):
        try:
            found.append((attempt, json.loads((attempt / 'run.json').read_text())))
        except (OSError, ValueError):
            found.append((attempt, dict(status='unknown')))
    for attempt, record in reversed(found):
        if record.get('status') == 'complete' and complete_artifacts(attempt):
            return attempt, record, True
    return found[-1] + (False,) if found else (None, dict(status='planned'), False)


def run_seed(folder, job, seed, settings):
    if seed_state(folder, job['scenario'], seed)[2]:
        print(f'Reusing {job["name"]}, seed {seed}', flush=True)
        return True
    number = len(attempts(folder, job['scenario'], seed)) + 1
    attempt = folder / scenario_directory(job['scenario']) / f's{seed}' / f'attempt-{number:03d}'
    attempt.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-m', 'tapestry.experiment.planner', 'fit', '--scenario', job['scenario'],
               '--seed', str(seed), '--device', settings['device'],
               '--eval-members', str(settings['eval_members']), '--output', str(attempt)]
    record = dict(status='running', name=job['name'], scenario=job['scenario'], seed=seed,
                 settings=settings, command=command, started=now(), **environment())
    save(attempt / 'run.json', record)
    print(f'Running {job["name"]}, seed {seed}: {attempt}', flush=True)
    try:
        with (attempt / 'run.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        if not complete_artifacts(attempt):
            raise RuntimeError('Run exited without complete fitted/scored artifacts')
    except (Exception, KeyboardInterrupt) as error:
        record.update(status='failed', error=str(error), finished=now())
        save(attempt / 'run.json', record)
        if isinstance(error, KeyboardInterrupt):
            raise
        print(f'Failed {job["name"]}, seed {seed}: {error}', flush=True)
        return False
    record.update(status='complete', finished=now())
    save(attempt / 'run.json', record)
    print(f'Completed {job["name"]}, seed {seed}', flush=True)
    return True


def run(folder, tasks=None, device=None, keep_going=False, fit_workers=1, seeds=None):
    settings = json.loads((folder / 'experiment.json').read_text())
    if device:
        settings['device'] = device
    jobs = read_jobs(folder)
    if tasks is not None:
        jobs = [job for job in jobs if job['task'] in tasks]
    if seeds is not None:
        jobs = [dict(job, seeds=[s for s in job['seeds'] if s in seeds]) for job in jobs]
        jobs = [job for job in jobs if job['seeds']]
    failures, stop = 0, threading.Event()

    def worker(job):
        failed = 0
        for seed in job['seeds']:
            if stop.is_set():
                break
            if not run_seed(folder, job, seed, settings):
                failed += 1
                if not keep_going:
                    stop.set()
                    break
        return failed

    with ThreadPoolExecutor(max_workers=fit_workers) as pool:
        for future in as_completed([pool.submit(worker, job) for job in jobs]):
            failures += future.result()
    return failures


def collect(folder):
    rows = []
    for job in read_jobs(folder):
        for seed in job['seeds']:
            attempt, record, done = seed_state(folder, job['scenario'], seed)
            status = 'complete' if done else 'incomplete' if record.get('status') == 'complete' else record.get('status')
            rows.append(dict(task=job['task'], name=job['name'], seed=seed, status=status,
                             attempt=attempt.relative_to(folder).as_posix() if attempt else '',
                             scenario=job['scenario']))
    if rows:
        write_csv(folder / 'runs.csv', rows, list(rows[0]))
    return rows


def completed_runs(folder, allow_incomplete=False, seeds=None):
    rows = collect(folder)
    if seeds is not None:
        rows = [row for row in rows if row['seed'] in seeds]
    done = [row for row in rows if row['status'] == 'complete']
    if not done or (len(done) < len(rows) and not allow_incomplete):
        raise ValueError(f'{len(rows) - len(done)} of {len(rows)} runs incomplete; finish them or pass --allow-incomplete')
    return done


def rank(folder, allow_incomplete=False, seeds=None):
    from tapestry.evaluation.totals import rank as rank_runs
    done = completed_runs(folder, allow_incomplete, seeds)
    destination = folder / f"ranking-{hashlib.sha256(json.dumps(sorted(r['attempt'] for r in done)).encode()).hexdigest()[:12]}"
    runs = [dict(config_id=row['scenario'], name=row['name'], seed=row['seed'], path=folder / row['attempt'])
            for row in sorted(done, key=lambda r: r['attempt'])]
    ranking = rank_runs(runs, destination)
    print(ranking.head(20).to_string(index=False), flush=True)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fit_parser = sub.add_parser('fit', help='Fit and evaluate one scenario/seed across all three seasons')
    fit_parser.add_argument('--scenario', required=True)
    fit_parser.add_argument('--seed', type=int, default=42)
    fit_parser.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])
    fit_parser.add_argument('--eval-members', type=int, default=256)
    fit_parser.add_argument('--output', required=True)
    for name in ('plan', 'run', 'status', 'rank'):
        p = sub.add_parser(name)
        p.add_argument('-e', '--experiment', required=True)
        p.add_argument('--root', default='data/experiments')
        if name == 'plan':
            p.add_argument('-s', '--scenario', nargs='+', required=True, help='Full scenario strings to plan')
            p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
            p.add_argument('--eval-members', type=int, default=256)
            p.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])
        if name == 'run':
            p.add_argument('-t', '--task', nargs='+', type=int, default=None)
            p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default=None)
            p.add_argument('--keep-going', action='store_true')
            p.add_argument('--fit-workers', type=int, default=1)
            p.add_argument('--seeds', nargs='+', type=int, default=None)
        if name in ('rank', 'status'):
            p.add_argument('--seeds', nargs='+', type=int, default=None)
        if name == 'rank':
            p.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'fit':
        scenario = Scenario.from_string(args.scenario)
        output = Path(args.output)
        for held_out in SEASONS:
            fit(scenario, args.seed, held_out, args.eval_members, args.device, output / f'eval_{held_out}')
        folds = {held: json.loads((output / f'eval_{held}' / 'manifest.json').read_text()) for held in SEASONS}
        save(output / 'manifest.json', dict(scenario=scenario.scenario_string, run_id=scenario.run_id,
                                            seed=args.seed, folds=list(SEASONS), fold_manifests=folds,
                                            eval_members=args.eval_members))
        from tapestry.evaluation.totals import score_run
        score_run(output, FROZEN)
        return
    folder = Path(args.root) / args.experiment
    if args.command == 'plan':
        scenarios = {Scenario.from_string(s).run_id: Scenario.from_string(s) for s in args.scenario}
        settings = dict(device=args.device, eval_members=args.eval_members, frozen=FROZEN)
        jobs = plan(folder, scenarios, args.seeds, settings)
        print(json.dumps(dict(experiment=str(folder), configurations=len(scenarios), seeds=len(args.seeds),
                              runs=len(jobs) and len(scenarios) * len(args.seeds))))
    elif args.command == 'run':
        if run(folder, args.task, args.device, args.keep_going, args.fit_workers, args.seeds):
            raise SystemExit(1)
    elif args.command == 'status':
        rows = collect(folder)
        if args.seeds is not None:
            rows = [row for row in rows if row['seed'] in args.seeds]
        for row in rows:
            print(f"{row['status']}\t{row['task']}\t{row['name']}\ts{row['seed']}\t{row['attempt']}")
    elif args.command == 'rank':
        print(rank(folder, args.allow_incomplete, args.seeds))


if __name__ == '__main__':
    main()
