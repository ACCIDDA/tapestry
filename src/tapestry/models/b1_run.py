"""Fit and predict one B1 leave-one-season-out fold.

Experiment management, scoring and ranking are the shared model-agnostic tools
(`tapestry.models.manager`, `tapestry.evaluation.totals`); this module only fits
a fold and writes its held-out-season forecasts.
"""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from tapestry.model_data.finalized import CHANNELS
from tapestry.model_data.wednesday import DEFAULT_DATASET, WednesdayDataset
from .b0 import fair_crps_cells
from .b1 import B1, MASK_SCENARIOS, draw_dropout
from .b1_scenarios import add_scenario_args, resolve
from .b1_seasons import fold
from .bundles import GROUPS
from .experiments import input_scales
from .objective import TARGET_WEIGHTS, loss_cell_weights, loss_scales
from .quantiles import LEVELS
from .run import calendar
from .season_cv import SEASONS

def known_finals(episodes):
    """Per-context-cell flag; two-field synthetic episodes have no supplied finals."""
    return np.stack([e['X'][:, :, 2].astype(bool) & e['X'][:, :, 1].astype(bool)
                     if e['X'].shape[2] > 2 else np.zeros_like(e['X'][:, :, 1], dtype=bool)
                     for e in episodes])


def supervision_mask(episode, dropout=None):
    """Supplied answers are unscored only while visible; labels remain in the archive."""
    valid = episode['Y'][:, :, 1].astype(bool).copy()
    if 'X' in episode:
        known = known_finals([episode])[0]
        if dropout is not None:
            known &= ~dropout
        valid[:2] &= ~known[-2:]
    return valid


def task_weights(episodes, target, direct=False, dropout=None):
    """Fixed scientific weights; recent and future each receive half the total.

    For an entirely absent task, its half remains zero (not reassigned). Visible
    supplied finals are excluded; hiding them restores recent supervision.
    Weights are computed across the partition, never within a minibatch.
    """
    weights = np.zeros((len(episodes), 6, 6, len(episodes[0]['locations'])), np.float32)
    targets = [target] if isinstance(target, int) else list(target)
    channel_weights = [TARGET_WEIGHTS[c] if c in targets else 0. for c in range(6)]
    supervised = []
    for i, e in enumerate(episodes):
        y = e['Y'].copy()
        y[:, :, 1] = supervision_mask(e, None if dropout is None else dropout[i])
        supervised.append({**e, 'Y': y})
    tasks = ((slice(2, 6), 1.),) if direct else ((slice(0, 2), .5), (slice(2, 6), .5))
    for interval, share in tasks:
        part = [{**e, 'Y': e['Y'][interval], 'target_dates': e['target_dates'][interval]} for e in supervised]
        if not any(e['Y'][:, targets, 1].any() for e in part):
            continue
        weights[:, interval] = share * loss_cell_weights(part, channel_weights)
    return weights[:, 2:] if direct else weights


def unique_truth(episodes):
    by_date = {}
    for e in episodes:
        for day, row in zip(e['target_dates'], e['Y']):
            if day not in by_date:
                by_date[day] = row.copy()
            else:
                observed = row[:, 1].astype(bool)
                old = by_date[day]
                if np.any(observed & old[:, 1].astype(bool) & (old[:, 0] != row[:, 0])):
                    raise ValueError('Inconsistent reference truth within fitting partition')
                by_date[day] = np.where(observed[:, None], row, old)
    return np.stack(list(by_date.values()))


def arrays(episodes, device):
    x = np.stack([e['X'] for e in episodes])
    y = np.stack([e['Y'] for e in episodes])
    cal = calendar([e['context_dates'][-1] for e in episodes], True)
    return tuple(torch.as_tensor(a, device=device) for a in (x[:, :, :, 0], x[:, :, :, 1].astype(bool), y, cal))


def populations(path, locations):
    values = {}
    with open(path) as stream:
        for row in csv.DictReader(stream):
            loc, value = row.get('abbreviation') or row['location'], float(row['population'])
            if loc in values or not np.isfinite(value) or value <= 0:
                raise ValueError(f'Invalid or duplicate population for {loc}')
            values[loc] = value
    return {loc: values[loc] for loc in locations}


def partitions(ds, args, direct=True):
    """B0's leave-one-season-out fold. B1 has no other split.

    The earlier chronological split validated on nine summer issuances, whose
    medians sat 10-13x below the fitting and evaluation data, so selection
    optimized a seasonal floor. It was removed rather than kept as an option.

    Leave-one-season-out over completed seasons uses labels pinned after the fold's
    own dates by construction, exactly as B0's finalized CV does, so these are
    retrospective development runs and `--retrospective` must be explicit.
    """
    if not args.retrospective:
        raise ValueError('Leave-one-season-out CV pins truth after each fold; pass --retrospective '
                         'to acknowledge these are retrospective development fits, not operational ones.')
    fitting, validation, refit, _, _ = fold(ds, args.held_out_season,
                                            min_availability=getattr(args, 'min_availability', 0.),
                                            direct=direct)
    return fitting, validation, refit


def crop_episodes(episodes, lookback):
    if any(len(e['context_dates']) < lookback for e in episodes):
        raise ValueError(f'Materialize at least {lookback} context weeks; do not pad an unavailable longer history')
    return [{**e, 'X': e['X'][-lookback:], 'context_dates': e['context_dates'][-lookback:]} for e in episodes]


def fit_component(train, validation, targets, component, options, args, scenario, epochs=None):
    """Fit one component. With `validation`, select the best epoch on it; without,
    fit `epochs` epochs on `train` with no early-stopping decision (B0's refit)."""
    direct = scenario.pipeline == 'direct'
    seed = args.seed + 10000 * component
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = B1(target=targets, direct=direct, **options).to(args.device)
    selecting = validation is not None
    budget = scenario.epochs if epochs is None else epochs
    # Each fitted component uses only episodes with at least one of its labels.
    hs = slice(2, 6) if direct else slice(0, 6)
    train = [e for e in train if e['Y'][hs][:, targets, 1].any()]
    if selecting:
        validation = [e for e in validation if e['Y'][hs][:, targets, 1].any()]
    if not train or (selecting and not validation):
        raise ValueError(f'No training/validation labels for channels {targets}')
    x, a, y, cal = arrays(train, args.device)
    known = torch.as_tensor(known_finals(train), device=args.device)
    y = y[:, hs][:, :, targets]
    weights = torch.as_tensor(task_weights(train, targets, direct)[:, :, targets], device=args.device)
    if selecting:
        vx, va, vy, vcal = arrays(validation, args.device)
        vknown = torch.as_tensor(known_finals(validation), device=args.device)
        vy = vy[:, hs][:, :, targets]
        vw = torch.as_tensor(task_weights(validation, targets, direct)[:, :, targets], device=args.device)
    for i, c in enumerate(targets):
        if not weights[:, :, i].sum() or (selecting and not vw[:, :, i].sum()):
            raise ValueError(f'No training/validation labels for {CHANNELS[c]}')
    probabilities = scenario.mask_probabilities
    if selecting:
        generator = torch.Generator().manual_seed(seed + 2000)
        fixed = {}
        for stage in (('future',) if direct else ('recent', 'future')):
            for name, value in model.draw_noise(scenario.validation_members, len(validation), generator).items():
                if value is not None:
                    fixed[f'{"z" if name == "z" else name}_{stage}'] = value
        vd = torch.as_tensor(draw_dropout(va.cpu().numpy(), np.random.default_rng(seed + 2000), probabilities), device=args.device)
        vw = torch.as_tensor(task_weights(validation, targets, direct, vd.cpu().numpy())[:, :, targets], device=args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=scenario.lr, weight_decay=scenario.weight_decay)
    best, best_state, best_epoch = float('inf'), None, 0
    history, audit = [], []
    for epoch in range(budget):
        d = draw_dropout(a.cpu().numpy(), rng, probabilities)
        dropout = torch.as_tensor(d, device=args.device)
        weights = torch.as_tensor(task_weights(train, targets, direct, d)[:, :, targets], device=args.device)
        audit.append(np.packbits(d.reshape(-1)))
        total = 0.
        model.train()
        for ids in torch.randperm(len(train), device=args.device).split(scenario.batch_size):
            optimizer.zero_grad()
            samples = model(x[ids], a[ids], cal[ids], scenario.members, dropout[ids], known_final=known[ids])
            score = fair_crps_cells(samples, y[ids, :, :, 0], y[ids, :, :, 1])
            loss = (weights[ids] * score / model.scale[targets]).sum() * len(train) / len(ids)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite B1 loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += float(loss.detach()) * len(ids) / len(train)
        val = None
        if selecting:
            model.eval()
            val = 0.
            with torch.no_grad():
                for ids in torch.arange(len(validation), device=args.device).split(scenario.batch_size):
                    samples = model(vx[ids], va[ids], vcal[ids], dropout=vd[ids],
                                    known_final=vknown[ids],
                                    **{k: v[:, ids] for k, v in fixed.items()})
                    score = fair_crps_cells(samples, vy[ids, :, :, 0], vy[ids, :, :, 1])
                    val += float((vw[ids] * score / model.scale[targets]).sum())
            if not np.isfinite(val):
                raise ValueError('Nonfinite B1 validation loss')
            if val < best:
                best, best_epoch = val, epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(dict(epoch=epoch + 1, loss=total, validation_loss=val, hidden=int(d.sum())))
        print(json.dumps(dict(run_id=scenario.run_id, targets=[CHANNELS[c] for c in targets], seed=args.seed,
                              phase='select' if selecting else 'refit', **history[-1])), flush=True)
        if selecting and scenario.patience and epoch + 1 - best_epoch >= scenario.patience:
            break
    # Selection returns its best checkpoint only as a diagnostic; the reported model
    # is the refit below. patience=0 is an explicit fixed-epoch run, as in B0.
    if selecting and scenario.patience:
        model.load_state_dict(best_state)
    record = dict(targets=targets, best_epoch=best_epoch,
        selected_epoch=(best_epoch if scenario.patience else scenario.epochs) if selecting else budget,
        phase='select' if selecting else 'refit', epochs=budget,
        fitting_episodes=len(train), validation_episodes=len(validation) if selecting else 0, history=history)
    audit_arrays = dict(dropout_packed=np.stack(audit), dropout_shape=np.array(a.shape),
        train_issuance_dates=[e['issuance_date'] for e in train])
    if selecting:
        audit_arrays.update(validation_dropout=vd.cpu().numpy(),
                            validation_issuance_dates=[e['issuance_date'] for e in validation])
    return model.cpu(), record, audit_arrays


def train(args, scenario=None):
    scenario = resolve(args) if scenario is None else scenario
    ds = WednesdayDataset.load(args.dataset)
    fitting, validation, refit = partitions(ds, args, direct=scenario.pipeline == 'direct')
    fitting = [e for e in crop_episodes(fitting, scenario.lookback) if e['X'][:, :, 1].any()]
    validation = [e for e in crop_episodes(validation, scenario.lookback) if e['X'][:, :, 1].any()]
    refit = [e for e in crop_episodes(refit, scenario.lookback) if e['X'][:, :, 1].any()]
    if not fitting or not validation or not refit:
        raise ValueError('No usable episodes within the selected lookback')
    pop = populations(args.population_file, ds.locations)

    def fit_options(episodes):
        """Normalizers, logit centers and native Q95 loss scales from one partition."""
        return dict(populations=pop, locations=list(ds.locations), lookback=scenario.lookback,
            width=scenario.width, latent=scenario.latent, scale=loss_scales(unique_truth(episodes)),
            **input_scales(episodes, scenario.count_transform, scenario.ed_transform, pop),
            **scenario.model_options())

    # Selection uses inner-fit statistics only; the refit recomputes its own from the
    # full training partition, as B0 does with `channel_scales` over training weeks.
    options, refit_options = fit_options(fitting), fit_options(refit)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    components, records = [], []
    groups = GROUPS[scenario.fit_partition]
    for i, targets in enumerate(groups):
        selected = scenario.epochs
        if scenario.patience:
            _, record, audit = fit_component(fitting, validation, targets, i, options, args, scenario)
            records.append(record)
            np.savez_compressed(out / f'masks-{i}.npz', **audit)
            selected = record['selected_epoch']
        # Fresh model and optimizer, reseeded by fit_component's seed + 10000 * i rule.
        model, record, audit = fit_component(refit, None, targets, i, refit_options, args, scenario,
                                             epochs=selected)
        components.append(dict(config=model.config, state_dict=model.state_dict()))
        records.append(record)
        np.savez_compressed(out / f'refit-masks-{i}.npz', **audit)
    metadata = dict(model='B1', schema_version=3, scenario=scenario.scenario_string,
        run_id=scenario.run_id, configuration=asdict(scenario), groups=groups,
        seed=args.seed, held_out_season=args.held_out_season, protocol='season_cv_refit_v1',
        training_procedure=('Epoch count selected per component on inner validation, then a fresh '
            'seeded model refitted on all permitted training weeks for exactly that count, as in B0 '
            'season_cv.run. Selection checkpoints are diagnostics and are never exported.'
            if scenario.patience else
            f'Fixed {scenario.epochs} epochs on all permitted training weeks; no epoch selection.'),
        selected_epochs=[r['selected_epoch'] for r in records if r['phase'] == 'select'],
        refit_epochs=[r['epochs'] for r in records if r['phase'] == 'refit'],
        retrospective=args.retrospective, dataset_metadata=ds.metadata,
        dataset_sha256=hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
        population_file_sha256=hashlib.sha256(Path(args.population_file).read_bytes()).hexdigest(),
        mask_probabilities=list(scenario.mask_probabilities), training_members=scenario.members,
        validation_members=scenario.validation_members, records=records,
        objective='.5 recent + .5 future native fair CRPS / fitting-only Q95; visible supplied finals excluded from recent loss. Partition-wide season/target/geography weights recomputed after dropout; absent task share stays zero. Direct: future only.',
        cross_target_dependence='Independent draws across fitted components. Shared components allow within-group dependence; marginal scores do not establish joint calibration.')
    torch.save(dict(components=components, metadata=metadata), out / 'model.pt')
    (out / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    # The fold predicts its own held-out season immediately, as B0 evaluates inside
    # season_cv, so the run's scoring step needs no second model load.
    from .b1_report import evaluate
    from .b1_seasons import fold as season_fold
    fitted = load_models(out / 'model.pt', args.device)[0]
    _, _, _, evaluation, info = season_fold(ds, args.held_out_season,
                                            direct=scenario.pipeline == 'direct')
    evaluation = [e for e in crop_episodes(evaluation, scenario.lookback) if e['X'][:, :, 1].any()]
    if not evaluation:
        raise ValueError(f'No usable evaluation episodes for held-out {args.held_out_season}')
    settings = argparse.Namespace(device=args.device, evaluation_members=args.eval_members)
    evaluate(fitted, evaluation, ds, settings, scenario.run_id, args.seed, out, scenario.scenario_string)
    metadata['fold'] = info
    (out / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return out / 'model.pt'


def load_models(checkpoint, device):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['metadata'].get('schema_version') != 3:
        raise ValueError('B1 checkpoint predates known-final conditioning; rebuild the dataset and refit')
    models = []
    for component in saved['components']:
        model = B1(**component['config']).to(device)
        model.load_state_dict(component['state_dict'])
        models.append(model.eval())
    return models, saved['metadata']


def sample(models, episodes, *, members, seed, device, scenario='natural', sample_batch=32):
    episodes = crop_episodes(episodes, models[0].config['lookback'])
    x, a, _, cal = arrays(episodes, device)
    known = torch.as_tensor(known_finals(episodes), device=device)
    d = torch.as_tensor(draw_dropout(a.cpu().numpy(), np.random.default_rng(seed + 3000), scenario=scenario), device=device)
    components = []
    with torch.no_grad():
        for c, model in enumerate(models):
            torch.manual_seed(seed + 10000 * c)
            components.append(torch.cat([model(x, a, cal, min(sample_batch, members - m), d, known_final=known).cpu()
                                        for m in range(0, members, sample_batch)]).numpy())
    order = [c for model in models for c in model.targets]
    if sorted(order) != list(range(6)):
        raise ValueError('B1 prediction bundle must cover every channel exactly once')
    samples = np.concatenate(components, axis=3)[:, :, :, [order.index(c) for c in range(6)]]
    return samples, d.cpu().numpy()


def predict(args):
    ds = WednesdayDataset.load(args.dataset)
    models, metadata = load_models(args.checkpoint, args.device)
    if list(ds.locations) != models[0].config['locations']:
        raise ValueError('Prediction locations/order must match checkpoint')
    episodes = list(ds.episodes(start=args.issuance, end=args.issuance, supervised=False))
    if len(episodes) != 1:
        raise ValueError('No usable Wednesday snapshot at requested issuance')
    episodes = crop_episodes(episodes, models[0].config['lookback'])
    if not episodes[0]['X'][:, :, 1].any():
        raise ValueError('No conditioning observations within the selected lookback')
    samples, d = sample(models, episodes, members=args.members, seed=args.seed, device=args.device,
                        scenario=args.stress_scenario, sample_batch=args.sample_batch)
    targets = episodes[0]['target_dates'][2:] if models[0].config['direct'] else episodes[0]['target_dates']
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, samples=samples[:, 0], quantiles=np.quantile(samples[:, 0], LEVELS, axis=0),
        quantile_levels=LEVELS, target_dates=targets, issuance_date=args.issuance, locations=ds.locations,
        channels=CHANNELS, context_dates=episodes[0]['context_dates'],
        X_values=episodes[0]['X'][:, :, 0], X_available=episodes[0]['X'][:, :, 1].astype(bool),
        X_final=known_finals(episodes)[0], D=d[0],
        X_provenance=ds.arrays['X_provenance'][episodes[0]['index'], -models[0].config['lookback']:],
        X_reason=ds.arrays['X_reason'][episodes[0]['index'], -models[0].config['lookback']:],
        metadata=json.dumps(dict(fit=metadata, scenario=args.stress_scenario, seed=args.seed,
            input_dataset_metadata=ds.metadata)))
    print(json.dumps(dict(output=str(path), shape=list(samples[:, 0].shape))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    identify = commands.add_parser('scenario', help='Print resolved canonical configuration without fitting')
    add_scenario_args(identify)
    for command in ('train', 'predict'):
        p = commands.add_parser(command)
        p.add_argument('--dataset', default=DEFAULT_DATASET)
        p.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cpu')
        p.add_argument('--seed', type=int, default=42)
        p.add_argument('--output', required=True)
        if command == 'predict':
            p.add_argument('--members', type=int, default=256)
            p.add_argument('--checkpoint', required=True)
            p.add_argument('--issuance', required=True)
            p.add_argument('--stress-scenario', dest='stress_scenario', choices=MASK_SCENARIOS, default='natural')
            p.add_argument('--sample-batch', type=int, default=32)
        else:
            add_scenario_args(p)
            p.add_argument('--population-file', default='data/metadata/locations.csv')
            p.add_argument('--held-out-season', choices=SEASONS, required=True,
                           help="B0's leave-one-season-out fold; the only B1 protocol")
            p.add_argument('--eval-members', type=int, default=256,
                           help='Draws for the held-out season forecasts this fold writes')
            p.add_argument('--retrospective', action='store_true',
                           help='Acknowledge that leave-one-season-out CV pins truth after each fold')
            p.add_argument('--min-availability', type=float, default=0.,
                           help='Drop fitting/validation episodes below this mean input '
                                'availability; evaluation is never filtered')
    args = parser.parse_args(argv)
    try:
        if args.command == 'scenario':
            scenario = resolve(args)
            print(json.dumps(dict(run_id=scenario.run_id, scenario=scenario.scenario_string,
                                  config=asdict(scenario), mask_probabilities=scenario.mask_probabilities), indent=2))
            return
        if args.command == 'train' and args.eval_members < 2:
            raise ValueError('At least two evaluation members required for fair CRPS')
        if args.command == 'predict' and min(args.members, args.sample_batch) < 1:
            raise ValueError('Prediction members and sample-batch must be positive')
        torch.set_num_threads(int(os.environ.get('TAPESTRY_TORCH_THREADS', '2')))
        globals()[args.command](args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
