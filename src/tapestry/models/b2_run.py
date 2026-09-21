"""Fit and predict B2 covariate extensions of B1 formulation B."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from tapestry.model_data.finalized import CHANNELS
from .b1 import MASK_SCENARIOS, draw_dropout
from .b1_run import (arrays, covariate_array, crop_episodes,
                     fit_component, known_finals, populations, unique_truth)
from .b1_seasons import fold
from .b1_report import evaluate
from .b2 import B2
from .b2_scenarios import SOURCE_GROUPS, add_scenario_args, resolve
from .bundles import GROUPS
from .experiments import input_scales
from .objective import loss_scales
from .quantiles import LEVELS
from .season_cv import SEASONS


def _data_api():
    from tapestry.model_data.b2 import B2Dataset, COVARIATE_GROUPS
    try:
        from tapestry.model_data.b2 import DEFAULT_DATASET
    except ImportError:
        DEFAULT_DATASET = 'data/processed/build_b2_wednesday.npz'
    return B2Dataset, COVARIATE_GROUPS, DEFAULT_DATASET


def selected_names(ds, covariate_set):
    """Resolve scenario source groups to the dataset's ordered names."""
    _, groups, _ = _data_api()
    if covariate_set == 'none':
        return []
    names = list(getattr(ds, 'covariate_names', ds.metadata.get('covariate_names', ())))
    selected = []
    for group in covariate_set.split('+'):
        if group not in SOURCE_GROUPS:
            raise ValueError(f'Unknown covariate group {group}')
        for value in groups[group]:
            name = names[value] if isinstance(value, (int, np.integer)) else value
            if name not in names:
                raise ValueError(f'Dataset is missing covariate {name}')
            selected.append(name)
    return selected


def select_covariates(episodes, all_names, selected):
    if not selected:
        return [{k: v for k, v in e.items() if k not in ('C', 'covariate_names')} for e in episodes]
    indices = [all_names.index(name) for name in selected]
    return [{**e, 'C': e['C'][:, indices], 'covariate_names': list(selected)} for e in episodes]


def _partitions(ds, args, scenario):
    if not args.retrospective:
        raise ValueError('B2 season CV pins final outcomes after each fold; pass --retrospective '
                         'to acknowledge these are retrospective development fits.')
    parts = fold(ds, args.held_out_season, min_availability=args.min_availability,
                 direct=True, input_mode=scenario.input_mode, allow_empty_context=True)
    names = list(getattr(ds, 'covariate_names', ds.metadata.get('covariate_names', ())))
    selected = selected_names(ds, scenario.covariate_set)
    converted = [select_covariates(part, names, selected) for part in parts[:4]]
    return (*converted, parts[4], selected)


def train(args, scenario=None):
    scenario = resolve(args) if scenario is None else scenario
    Dataset, _, _ = _data_api()
    ds = Dataset.load(args.dataset)
    fitting, validation, refit, evaluation, info, names = _partitions(ds, args, scenario)
    fitting, validation, refit, evaluation = [crop_episodes(part, scenario.lookback)
                                               for part in (fitting, validation, refit, evaluation)]
    if not all((fitting, validation, refit, evaluation)):
        raise ValueError('No usable episodes within the selected lookback')
    pop = populations(args.population_file, ds.locations)

    def fit_options(episodes):
        options = dict(populations=pop, locations=list(ds.locations), lookback=scenario.lookback,
            width=scenario.width, latent=scenario.latent, scale=loss_scales(unique_truth(episodes)),
            **input_scales(episodes, scenario.count_transform, scenario.ed_transform, pop),
            **scenario.model_options(), covariate_names=names)
        return options

    options, refit_options = fit_options(fitting), fit_options(refit)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    components, records = [], []
    groups = GROUPS[scenario.fit_partition]
    for i, targets in enumerate(groups):
        selected_epoch = scenario.epochs
        if scenario.patience:
            _, record, audit = fit_component(fitting, validation, targets, i, options, args, scenario,
                                             model_class=B2)
            records.append(record)
            np.savez_compressed(out / f'masks-{i}.npz', **audit)
            selected_epoch = record['selected_epoch']
        model, record, audit = fit_component(refit, None, targets, i, refit_options, args, scenario,
                                             epochs=selected_epoch, model_class=B2)
        components.append(dict(config=model.config, state_dict=model.state_dict()))
        records.append(record)
        np.savez_compressed(out / f'refit-masks-{i}.npz', **audit)

    metadata = dict(model='B2', schema_version=1, scenario=scenario.scenario_string,
        run_id=scenario.run_id, configuration=asdict(scenario), groups=groups,
        covariate_names=names,
        covariate_normalization=('Per covariate and location over observed cells in each component-eligible '
                                 'fitting partition; zero-support cells remain masked at inference.'),
        input_mode=scenario.input_mode, seed=args.seed, held_out_season=args.held_out_season,
        protocol='season_cv_refit_v1', retrospective=args.retrospective,
        dataset_metadata=ds.metadata,
        dataset_sha256=hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
        population_file_sha256=hashlib.sha256(Path(args.population_file).read_bytes()).hexdigest(),
        mask_probabilities=list(scenario.mask_probabilities), training_members=scenario.members,
        validation_members=scenario.validation_members, eval_members=args.eval_members, records=records,
        selected_epochs=[r['selected_epoch'] for r in records if r['phase'] == 'select'],
        refit_epochs=[r['epochs'] for r in records if r['phase'] == 'refit'],
        training_procedure=('Epoch count selected per component on inner validation, then a fresh seeded model '
                            'refitted on all permitted training weeks for exactly that count.'
                            if scenario.patience else
                            f'Fixed {scenario.epochs} epochs on all permitted training weeks; no epoch selection.'),
        selection_objective='forecast',
        objective='Future native fair CRPS / fitting-only Q95 with partition-wide scientific weights.',
        cross_target_dependence='Independent draws across fitted components; shared components preserve within-group dependence.')
    torch.save(dict(components=components, metadata=metadata), out / 'model.pt')
    (out / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')

    models, _ = load_models(out / 'model.pt', args.device)
    settings = argparse.Namespace(device=args.device, evaluation_members=args.eval_members)
    evaluate(models, evaluation, ds, settings, scenario.run_id, args.seed, out,
             scenario.scenario_string, sample_fn=sample)
    metadata['fold'] = info
    (out / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return out / 'model.pt'


def load_models(checkpoint, device):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['metadata'].get('model') != 'B2' or saved['metadata'].get('schema_version') != 1:
        raise ValueError('Unsupported B2 checkpoint')
    models = []
    for component in saved['components']:
        model = B2(**component['config']).to(device)
        model.load_state_dict(component['state_dict'])
        models.append(model.eval())
    return models, saved['metadata']


def sample(models, episodes, *, members, seed, device, scenario='natural', sample_batch=32, mask_seed=None):
    episodes = crop_episodes(episodes, models[0].config['lookback'])
    x, a, _, cal = arrays(episodes, device)
    cov = covariate_array(episodes, device)
    known = torch.as_tensor(known_finals(episodes), device=device)
    d = torch.as_tensor(draw_dropout(a.cpu().numpy(),
        np.random.default_rng((seed if mask_seed is None else mask_seed) + 3000), scenario=scenario), device=device)
    components = []
    with torch.no_grad():
        for c, model in enumerate(models):
            torch.manual_seed(seed + 10000 * c)
            batches = []
            for m in range(0, members, sample_batch):
                batches.append(model(x, a, cal, min(sample_batch, members - m), d, known_final=known,
                                     covariates=cov).cpu())
            components.append(torch.cat(batches).numpy())
    order = [c for model in models for c in model.targets]
    if sorted(order) != list(range(6)):
        raise ValueError('B2 prediction bundle must cover every channel exactly once')
    return np.concatenate(components, axis=3)[:, :, :, [order.index(c) for c in range(6)]], d.cpu().numpy()


def predict(args):
    Dataset, _, _ = _data_api()
    ds = Dataset.load(args.dataset)
    models, metadata = load_models(args.checkpoint, args.device)
    if list(ds.locations) != models[0].config['locations']:
        raise ValueError('Prediction locations/order must match checkpoint')
    episodes = list(ds.episodes(start=args.issuance, end=args.issuance, supervised=False,
                                input_mode=metadata['input_mode']))
    if len(episodes) != 1:
        raise ValueError('No usable Wednesday snapshot at requested issuance')
    all_names = list(getattr(ds, 'covariate_names', ds.metadata.get('covariate_names', ())))
    episodes = select_covariates(episodes, all_names, metadata['covariate_names'])
    samples, d = sample(models, episodes, members=args.members, seed=args.seed, device=args.device,
                        scenario=args.stress_scenario, sample_batch=args.sample_batch)
    episode = crop_episodes(episodes, models[0].config['lookback'])[0]
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, samples=samples[:, 0], quantiles=np.quantile(samples[:, 0], LEVELS, axis=0),
        quantile_levels=LEVELS, target_dates=episode['target_dates'][2:], issuance_date=args.issuance,
        locations=ds.locations, channels=CHANNELS, context_dates=episode['context_dates'],
        X_values=episode['X'][:, :, 0], X_available=episode['X'][:, :, 1].astype(bool),
        X_final=known_finals([episode])[0], D=d[0], covariate_names=metadata['covariate_names'],
        C=episode.get('C', np.empty((len(episode['context_dates']), 0, 2, len(ds.locations)))),
        metadata=json.dumps(dict(fit=metadata, scenario=args.stress_scenario, seed=args.seed,
                                 input_dataset_metadata=ds.metadata)))
    print(json.dumps(dict(output=str(path), shape=list(samples[:, 0].shape))))


def main(argv=None):
    _, _, default_dataset = _data_api()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    identify = commands.add_parser('scenario')
    add_scenario_args(identify)
    for command in ('train', 'predict'):
        p = commands.add_parser(command)
        p.add_argument('--dataset', default=default_dataset)
        p.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cpu')
        p.add_argument('--seed', type=int, default=42)
        p.add_argument('--output', required=True)
        if command == 'predict':
            p.add_argument('--members', type=int, default=256)
            p.add_argument('--checkpoint', required=True)
            p.add_argument('--issuance', required=True)
            p.add_argument('--stress-scenario', choices=MASK_SCENARIOS, default='natural')
            p.add_argument('--sample-batch', type=int, default=32)
        else:
            add_scenario_args(p)
            p.add_argument('--population-file', default='data/metadata/locations.csv')
            p.add_argument('--held-out-season', choices=SEASONS, required=True)
            p.add_argument('--eval-members', type=int, default=256)
            p.add_argument('--retrospective', action='store_true')
            p.add_argument('--min-availability', type=float, default=0.)
    args = parser.parse_args(argv)
    try:
        if args.command == 'scenario':
            scenario = resolve(args)
            print(json.dumps(dict(run_id=scenario.run_id, scenario=scenario.scenario_string,
                                  config=asdict(scenario)), indent=2))
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
