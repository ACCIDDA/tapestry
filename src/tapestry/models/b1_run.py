"""Train/predict B1 and compare direct, two-stage, and masked two-stage models."""
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
from tapestry.model_data.wednesday import WednesdayDataset
from .b0 import fair_crps_cells
from .b1 import B1, MASK_SCENARIOS, draw_dropout
from .b1_scenarios import PRESETS, add_scenario_args, resolve, comparison_grid
from .bundles import GROUPS
from .experiments import input_scales
from .objective import TARGET_WEIGHTS, loss_cell_weights, loss_scales
from .quantiles import LEVELS
from .run import calendar

def task_weights(episodes, target, direct=False):
    """Fixed scientific weights; recent and future each receive half the total.

    For an entirely absent task, its half remains zero (not reassigned). Such a
    target is rejected by fitting; partial missing labels retain valid supervision.
    """
    weights = np.zeros((len(episodes), 6, 6, len(episodes[0]['locations'])), np.float32)
    targets = [target] if isinstance(target, int) else list(target)
    channel_weights = [TARGET_WEIGHTS[c] if c in targets else 0. for c in range(6)]
    tasks = ((slice(2, 6), 1.),) if direct else ((slice(0, 2), .5), (slice(2, 6), .5))
    for interval, share in tasks:
        part = [{**e, 'Y': e['Y'][interval], 'target_dates': e['target_dates'][interval]} for e in episodes]
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


def partitions(ds, args):
    if not args.train_end < args.validation_start <= args.validation_end:
        raise ValueError('Require train-end < validation-start <= validation-end')
    if ds.metadata['truth_cutoff'] > args.validation_end and not args.retrospective:
        raise ValueError('Later reference finals require --retrospective. For operational fits rebuild with truth-cutoff <= validation-end.')
    fit = list(ds.episodes(end=args.train_end, target_end=args.train_end))
    validation = list(ds.episodes(start=args.validation_start, end=args.validation_end,
                                  target_start=args.validation_start, target_end=args.validation_end))
    if not fit or not validation:
        raise ValueError('No usable fitting or validation episodes')
    return fit, validation


def crop_episodes(episodes, lookback):
    if any(len(e['context_dates']) < lookback for e in episodes):
        raise ValueError(f'Materialize at least {lookback} context weeks; do not pad an unavailable longer history')
    return [{**e, 'X': e['X'][-lookback:], 'context_dates': e['context_dates'][-lookback:]} for e in episodes]


def fit_component(train, validation, targets, component, options, args, scenario):
    direct = scenario.pipeline == 'direct'
    seed = args.seed + 10000 * component
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = B1(target=targets, direct=direct, **options).to(args.device)
    # Each fitted component uses only episodes with at least one of its labels.
    hs = slice(2, 6) if direct else slice(0, 6)
    train = [e for e in train if e['Y'][hs][:, targets, 1].any()]
    validation = [e for e in validation if e['Y'][hs][:, targets, 1].any()]
    if not train or not validation:
        raise ValueError(f'No training/validation labels for channels {targets}')
    x, a, y, cal = arrays(train, args.device)
    vx, va, vy, vcal = arrays(validation, args.device)
    y, vy = y[:, hs][:, :, targets], vy[:, hs][:, :, targets]
    weights = torch.as_tensor(task_weights(train, targets, direct)[:, :, targets], device=args.device)
    vw = torch.as_tensor(task_weights(validation, targets, direct)[:, :, targets], device=args.device)
    for i, c in enumerate(targets):
        if not weights[:, :, i].sum() or not vw[:, :, i].sum():
            raise ValueError(f'No training/validation labels for {CHANNELS[c]}')
        if not direct and (not weights[:, :2, i].sum() or not weights[:, 2:, i].sum()):
            raise ValueError(f'Both recent and future training labels required for {CHANNELS[c]}')
    generator = torch.Generator().manual_seed(seed + 2000)
    fixed = {}
    for stage in ('recent', 'future'):
        for name, value in model.draw_noise(scenario.validation_members, len(validation), generator).items():
            if value is not None:
                fixed[f'{"z" if name == "z" else name}_{stage}'] = value
    probabilities = scenario.mask_probabilities
    vd = torch.as_tensor(draw_dropout(va.cpu().numpy(), np.random.default_rng(seed + 2000), probabilities), device=args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=scenario.lr, weight_decay=scenario.weight_decay)
    best, best_state, best_epoch = float('inf'), None, 0
    history, audit = [], []
    for epoch in range(scenario.epochs):
        d = draw_dropout(a.cpu().numpy(), rng, probabilities)
        dropout = torch.as_tensor(d, device=args.device)
        audit.append(np.packbits(d.reshape(-1)))
        total = 0.
        model.train()
        for ids in torch.randperm(len(train), device=args.device).split(scenario.batch_size):
            optimizer.zero_grad()
            samples = model(x[ids], a[ids], cal[ids], scenario.members, dropout[ids])
            score = fair_crps_cells(samples, y[ids, :, :, 0], y[ids, :, :, 1])
            loss = (weights[ids] * score / model.scale[targets]).sum() * len(train) / len(ids)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite B1 loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += float(loss.detach()) * len(ids) / len(train)
        model.eval()
        val = 0.
        with torch.no_grad():
            for ids in torch.arange(len(validation), device=args.device).split(scenario.batch_size):
                samples = model(vx[ids], va[ids], vcal[ids], dropout=vd[ids],
                                **{k: v[:, ids] for k, v in fixed.items()})
                score = fair_crps_cells(samples, vy[ids, :, :, 0], vy[ids, :, :, 1])
                val += float((vw[ids] * score / model.scale[targets]).sum())
        if not np.isfinite(val):
            raise ValueError('Nonfinite B1 validation loss')
        if val < best:
            best, best_epoch = val, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(dict(epoch=epoch + 1, loss=total, validation_loss=val, hidden=int(d.sum())))
        print(json.dumps(dict(run_id=scenario.run_id, targets=[CHANNELS[c] for c in targets], seed=args.seed, **history[-1])), flush=True)
        if scenario.patience and epoch + 1 - best_epoch >= scenario.patience:
            break
    # patience=0 is an explicit fixed-epoch run, as in B0.
    if scenario.patience:
        model.load_state_dict(best_state)
    return model.cpu(), dict(targets=targets, best_epoch=best_epoch,
        selected_epoch=best_epoch if scenario.patience else scenario.epochs,
        fitting_episodes=len(train), validation_episodes=len(validation), history=history), dict(
        dropout_packed=np.stack(audit), dropout_shape=np.array(a.shape),
        validation_dropout=vd.cpu().numpy(), train_issuance_dates=[e['issuance_date'] for e in train],
        validation_issuance_dates=[e['issuance_date'] for e in validation])


def train(args, scenario=None):
    scenario = resolve(args) if scenario is None else scenario
    ds = WednesdayDataset.load(args.dataset)
    fitting, validation = partitions(ds, args)
    fitting = [e for e in crop_episodes(fitting, scenario.lookback) if e['X'][:, :, 1].any()]
    validation = [e for e in crop_episodes(validation, scenario.lookback) if e['X'][:, :, 1].any()]
    if not fitting or not validation:
        raise ValueError('No usable episodes within the selected lookback')
    pop = populations(args.population_file, ds.locations)
    scales = input_scales(fitting, scenario.count_transform, scenario.ed_transform, pop)
    options = dict(populations=pop, locations=list(ds.locations), lookback=scenario.lookback,
        width=scenario.width, latent=scenario.latent, scale=loss_scales(unique_truth(fitting)),
        **scales, **scenario.model_options())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    components, records = [], []
    groups = GROUPS[scenario.fit_partition]
    for i, targets in enumerate(groups):
        model, record, audit = fit_component(fitting, validation, targets, i, options, args, scenario)
        components.append(dict(config=model.config, state_dict=model.state_dict()))
        records.append(record)
        np.savez_compressed(out / f'masks-{i}.npz', **audit)
    metadata = dict(model='B1', schema_version=2, scenario=scenario.scenario_string,
        run_id=scenario.run_id, configuration=asdict(scenario), groups=groups,
        seed=args.seed, train_end=args.train_end, validation_start=args.validation_start,
        validation_end=args.validation_end, retrospective=args.retrospective, dataset_metadata=ds.metadata,
        dataset_sha256=hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
        population_file_sha256=hashlib.sha256(Path(args.population_file).read_bytes()).hexdigest(),
        mask_probabilities=list(scenario.mask_probabilities), training_members=scenario.members,
        validation_members=scenario.validation_members, records=records,
        objective='.5 recent + .5 future native fair CRPS / fitting-only Q95; fixed season/target/geography weights within each component. Direct: future only.',
        cross_target_dependence='Independent draws across fitted components. Shared components allow within-group dependence; marginal scores do not establish joint calibration.')
    torch.save(dict(components=components, metadata=metadata), out / 'model.pt')
    (out / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return out / 'model.pt'


def load_models(checkpoint, device):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['metadata'].get('schema_version') != 2:
        raise ValueError('B1 checkpoint predates configurable formulations; refit using the current code')
    models = []
    for component in saved['components']:
        model = B1(**component['config']).to(device)
        model.load_state_dict(component['state_dict'])
        models.append(model.eval())
    return models, saved['metadata']


def sample(models, episodes, *, members, seed, device, scenario='natural', sample_batch=32):
    episodes = crop_episodes(episodes, models[0].config['lookback'])
    x, a, _, cal = arrays(episodes, device)
    d = torch.as_tensor(draw_dropout(a.cpu().numpy(), np.random.default_rng(seed + 3000), scenario=scenario), device=device)
    components = []
    with torch.no_grad():
        for c, model in enumerate(models):
            torch.manual_seed(seed + 10000 * c)
            components.append(torch.cat([model(x, a, cal, min(sample_batch, members - m), d).cpu()
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
        X_values=episodes[0]['X'][:, :, 0], X_available=episodes[0]['X'][:, :, 1].astype(bool), D=d[0],
        X_provenance=ds.arrays['X_provenance'][episodes[0]['index'], -models[0].config['lookback']:],
        X_reason=ds.arrays['X_reason'][episodes[0]['index'], -models[0].config['lookback']:],
        metadata=json.dumps(dict(fit=metadata, scenario=args.stress_scenario, seed=args.seed,
            input_dataset_metadata=ds.metadata)))
    print(json.dumps(dict(output=str(path), shape=list(samples[:, 0].shape))))


def compare(args):
    from .b1_report import evaluate, report
    base = resolve(args)
    if args.presets and (args.preset or args.scenario or any(getattr(args, k) is not None
            for k in ('encoder', 'spatial', 'head_sharing', 'fit_partition'))):
        raise ValueError('--presets selects those architecture fields; use a single --preset/--scenario for custom architecture overrides')
    if args.mask_rates is not None and args.mask_rate is not None:
        raise ValueError('Use --mask-rates for a grid or --mask-rate for one comparison level')
    if args.pipeline is not None:
        raise ValueError('compare expands both pipelines; select --pipeline with train or scenario instead')
    rates = args.mask_rates if args.mask_rates is not None else ([0., base.mask_rate] if args.mask_rate is not None else [0., .25, .5])
    grid = comparison_grid(base, args.presets, rates)
    ds = WednesdayDataset.load(args.evaluation_dataset or args.dataset)
    fitting_ds = WednesdayDataset.load(args.dataset)
    if ds.locations != fitting_ds.locations:
        raise ValueError('Fitting and evaluation datasets must have identical locations')
    if args.evaluation_start <= args.validation_end:
        raise ValueError('Forward evaluation must begin after validation-end')
    episodes = list(ds.episodes(start=args.evaluation_start, end=args.evaluation_end,
        target_start=args.evaluation_start, target_end=args.evaluation_end))
    if not episodes:
        raise ValueError('No evaluation episodes')
    if max(s.lookback for s in grid) > min(ds.metadata['lookback'], fitting_ds.metadata['lookback']):
        raise ValueError('Materialize enough context weeks for every comparison scenario')
    episodes = [e for e in crop_episodes(episodes, base.lookback) if e['X'][:, :, 1].any()]
    if not episodes:
        raise ValueError('No evaluation episodes within the selected lookback')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    plan = dict(settings=vars(args), configurations=[dict(run_id=s.run_id, scenario=s.scenario_string,
        config=asdict(s), mask_probabilities=s.mask_probabilities) for s in grid],
        evaluation_dataset_metadata=ds.metadata,
        evaluation_dataset_sha256=hashlib.sha256(Path(args.evaluation_dataset or args.dataset).read_bytes()).hexdigest())
    (root / 'comparison.json').write_text(json.dumps(plan, indent=2) + '\n')
    if args.plan_only:
        print(json.dumps(dict(configurations=len(grid), seed_runs=len(grid) * len(args.seeds), manifest=str(root / 'comparison.json'))))
        return
    results = []
    for seed in args.seeds:
        for scenario in grid:
            options = argparse.Namespace(**{**vars(args), 'seed': seed,
                'output': str(root / scenario.run_id / f's{seed}')})
            checkpoint = train(options, scenario)
            models, _ = load_models(checkpoint, args.device)
            results.extend(evaluate(models, crop_episodes(episodes, scenario.lookback), ds,
                                   args, scenario.run_id, seed, root, scenario.scenario_string))
    report(results, ds, root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    identify = commands.add_parser('scenario', help='Print resolved canonical configuration without fitting')
    add_scenario_args(identify)
    for command in ('train', 'predict', 'compare'):
        p = commands.add_parser(command)
        p.add_argument('--dataset', default='data/processed/build_b1_wednesday.npz')
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
            p.add_argument('--population-file', default='data/metadata/b0_locations.csv')
            p.add_argument('--train-end', required=True)
            p.add_argument('--validation-start', required=True)
            p.add_argument('--validation-end', required=True)
            p.add_argument('--retrospective', action='store_true', help='Explicitly allow later pinned truth during development')
            if command == 'compare':
                p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
                p.add_argument('--presets', choices=tuple(PRESETS), nargs='+', help='Formulations to compare using common training settings')
                p.add_argument('--mask-rates', type=float, nargs='+', help='Two-stage masking rates; default 0 .25 .5')
                p.add_argument('--plan-only', action='store_true', help='Save the complete scenario grid without training')
                p.add_argument('--evaluation-dataset', help='Separate later pinned truth for operational evaluation')
                p.add_argument('--evaluation-start', required=True)
                p.add_argument('--evaluation-end', required=True)
                p.add_argument('--evaluation-members', type=int, default=256)
    args = parser.parse_args(argv)
    try:
        if args.command == 'scenario':
            scenario = resolve(args)
            print(json.dumps(dict(run_id=scenario.run_id, scenario=scenario.scenario_string,
                                  config=asdict(scenario), mask_probabilities=scenario.mask_probabilities), indent=2))
            return
        if args.command == 'compare' and args.evaluation_members < 2:
            raise ValueError('At least two evaluation members required for fair CRPS')
        if args.command == 'predict' and min(args.members, args.sample_batch) < 1:
            raise ValueError('Prediction members and sample-batch must be positive')
        torch.set_num_threads(int(os.environ.get('TAPESTRY_TORCH_THREADS', '2')))
        globals()[args.command](args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
