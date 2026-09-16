"""B1's chronological fitting/scoring backend for the shared experiment manager."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

from .b1_scenarios import B1Scenario, PRESETS, comparison_grid

SCORE_VERSION = 'b1-task-normalized-crps-v2'
SCORE_DEFINITION = ('Rank forecasting and nowcasting separately by native fair CRPS / fitting-only '
                    'channel-location Q95; equal target-date seasons, admission target weights 1 and ED .5, '
                    '80% equally weighted states/DC and 20% US, renormalizing absent support. '
                    'Within a location average eligible origins/horizons. Mean and sample SD across seeds. '
                    'Natural inputs are primary; stress scenarios have separate rankings. Not Hub-relative skill. '
                    'Score columns use B0 naming: config_id identifies the configuration, stress the input scenario.')
DATES = ('train_end', 'validation_start', 'validation_end', 'evaluation_start', 'evaluation_end')


def add_plan_args(parser):
    group = parser.add_argument_group('B1 planning (--suite B1)')
    group.add_argument('--presets', choices=tuple(PRESETS), nargs='+', default=list(PRESETS))
    group.add_argument('--mask-rates', type=float, nargs='+', default=[0., .25, .5])
    for key, value in dict(epochs=300, patience=50, members=128, validation_members=256, width=64, batch_size=8).items():
        group.add_argument('--' + key.replace('_', '-'), type=int, default=value)
    for key in DATES:
        group.add_argument('--' + key.replace('_', '-'))
    group.add_argument('--evaluation-dataset')
    group.add_argument('--retrospective', action='store_true')


def scenarios(args):
    if args.scenario:
        return {s.run_id: s for s in map(B1Scenario.from_string, args.scenario)}
    base = B1Scenario(**{key: getattr(args, key) for key in
                        ('epochs', 'patience', 'members', 'validation_members', 'width', 'batch_size')})
    return {s.run_id: s for s in comparison_grid(base, args.presets, args.mask_rates)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def settings(args):
    if any(getattr(args, key) is None for key in DATES):
        raise ValueError('B1 plan requires explicit train/validation/evaluation dates')
    return dict(model='B1', **{key: getattr(args, key) for key in DATES},
                evaluation_dataset=args.evaluation_dataset or args.dataset,
                retrospective=args.retrospective)


def check_inputs(config):
    from tapestry.model_data.wednesday import WednesdayDataset
    from .b1_run import partitions, populations
    data = WednesdayDataset.load(config['dataset'])
    evaluation = WednesdayDataset.load(config['evaluation_dataset'])
    partitions(data, argparse.Namespace(**config))
    populations(config['population_file'], data.locations)
    if data.locations != evaluation.locations:
        raise ValueError('B1 fitting/evaluation locations must match in order')
    if not config['validation_end'] < config['evaluation_start'] <= config['evaluation_end']:
        raise ValueError('Evaluation must follow validation and end on/after its start')
    if config['eval_members'] < 2:
        raise ValueError('Fair CRPS requires at least two evaluation members')
    if config.get('frozen'):
        from .manager import check_frozen
        check_frozen(config['frozen'])
    for path, expected in config.get('input_sha256', {}).items():
        if sha(path) != expected:
            raise ValueError(f'B1 input changed since planning: {path}; use a new experiment')


def prepare(folder, config):
    """Save the same runnable source snapshot pattern used by B0.1."""
    from .manager import save, git_state
    root = Path(__file__).resolve().parents[3]
    destination = folder / 'code'
    files = list((root / 'src').rglob('*.py')) + list((root / 'src').rglob('*.R'))
    files += [root / 'scripts/b1_jlessler.sbatch', root / 'docs/design/b1.md', root / 'pyproject.toml']
    hashes = {}
    if destination.exists():
        shutil.rmtree(destination)
    for source in files:
        target = destination / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(source.relative_to(root))] = sha(source)
    inputs = {config[k] for k in ('dataset', 'evaluation_dataset', 'population_file')}
    if config.get('frozen'):
        from tapestry.evaluation.totals import frozen_cases
        frozen = Path(config['frozen'])
        inputs.add(str(frozen / 'manifest.json'))
        inputs |= {str(frozen / case['directory'] / name) for case in frozen_cases(frozen)
                   for name in ('units.parquet', 'quantiles.parquet')}
    config['input_sha256'] = {p: sha(p) for p in sorted(inputs)}
    config['source_snapshot_sha256'] = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    save(folder / 'experiment.json', config)
    save(folder / 'preflight.json', dict(source_sha256=hashes, input_sha256=config['input_sha256'], **git_state()))
    save(folder / 'design.json', dict(experiment='B1', protocol='chronological', score_definition=SCORE_DEFINITION))


def commands(scenario, seed, config, output):
    command = [sys.executable, '-m', 'tapestry.models', 'b1', 'train', *scenario.flags(),
               '--dataset', config['dataset'], '--population-file', config['population_file'],
               '--device', config['device'], '--seed', str(seed), '--output', str(output)]
    for key in DATES[:3]:
        command += ['--' + key.replace('_', '-'), config[key]]
    if config['retrospective']:
        command.append('--retrospective')
    score = [sys.executable, '-m', 'tapestry.models.b1_experiment', '--run', str(output),
             '--settings-json', json.dumps(config)]
    return command, score


def score_run(output, config):
    from tapestry.model_data.wednesday import WednesdayDataset
    from .b1_run import crop_episodes, load_models
    from .b1_report import evaluate, report
    from .manager import save
    check_inputs(config)
    models, metadata = load_models(output / 'model.pt', config['device'])
    ds = WednesdayDataset.load(config['evaluation_dataset'])
    episodes = list(ds.episodes(start=config['evaluation_start'], end=config['evaluation_end'],
                               target_start=config['evaluation_start'], target_end=config['evaluation_end']))
    episodes = [e for e in crop_episodes(episodes, metadata['configuration']['lookback']) if e['X'][:, :, 1].any()]
    if not episodes:
        raise ValueError('No usable B1 evaluation episodes')
    args = argparse.Namespace(device=config['device'], evaluation_members=config['eval_members'])
    rows = evaluate(models, episodes, ds, args, metadata['run_id'], metadata['seed'], output, metadata['scenario'])
    report(rows, ds, output)
    scoring = dict(score_version=SCORE_VERSION, definition=SCORE_DEFINITION,
                   evaluation_dataset_sha256=sha(config['evaluation_dataset']),
                   settings=config, rows=len(rows), scenario=metadata['scenario'], seed=metadata['seed'])
    if config.get('frozen'):
        from .b1_hubs import score_hubs
        scoring['hub_support'] = score_hubs(output, config['frozen'], metadata, config)
    save(output / 'scoring.json', scoring)


def complete_artifacts(output):
    try:
        metadata = json.loads((output / 'manifest.json').read_text())
        scoring = json.loads((output / 'scoring.json').read_text())
        prefix = f'{metadata["run_id"]}-s{metadata["seed"]}'
        files = ['model.pt', 'summary.csv', f'scores-{prefix}.parquet']
        files += [f'{kind}-{prefix}-{stress}.npz' for kind in ('forecasts', 'evaluation-masks')
                  for stress in ('natural', 'recent', 'gap', 'outage')]
        if scoring['settings'].get('frozen'):
            files += ['hub-support.json', 'hub-scores.parquet']
        return (metadata['model'] == 'B1' and scoring['rows'] > 0
                and scoring['scenario'] == metadata['scenario'] and scoring['seed'] == metadata['seed']
                and all((output / p).is_file() and (output / p).stat().st_size for p in files))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def validate_support(frame):
    """Match exact cells, labels, normalization and weights across seeds/configs."""
    keys = ['issuance_date', 'target_date', 'target', 'location', 'horizon']
    reference = ['observed', 'loss_scale', 'objective_weight']
    if frame.duplicated(['config_id', 'seed', 'stress', *keys]).any():
        raise ValueError('Duplicate B1 scoring cells')
    if not np.isfinite(frame[['crps', 'wis', *reference]].to_numpy()).all():
        raise ValueError('Nonfinite B1 scores, weights or truth')
    if (frame.loss_scale <= 0).any() or (frame.objective_weight < 0).any():
        raise ValueError('Invalid B1 normalization or weights')
    for task, part in frame.groupby('task'):
        baseline = None
        for _, run in part.groupby(['config_id', 'seed', 'stress']):
            values = run[keys + reference].sort_values(keys).reset_index(drop=True)
            if baseline is not None and not values.equals(baseline):
                raise ValueError(f'B1 {task} support/truth/scales/weights differ; use common evaluation and fitting support')
            baseline = values
            if not np.isclose(run.objective_weight.sum(), 1., atol=1e-5):
                raise ValueError(f'B1 {task} objective weights must sum to one')


def ranking_tables(frame):
    validate_support(frame)
    ids = ['config_id', 'scenario_string', 'seed', 'stress', 'task']
    weighted = frame.assign(normalized_crps=frame.crps / frame.loss_scale * frame.objective_weight,
                            normalized_wis=frame.wis / frame.loss_scale * frame.objective_weight)
    runs = weighted.groupby(ids)[['normalized_crps', 'normalized_wis']].sum().reset_index()
    ranks = runs.groupby([c for c in ids if c != 'seed']).agg(
        score_mean=('normalized_crps', 'mean'), score_sd=('normalized_crps', 'std'),
        wis_mean=('normalized_wis', 'mean'), wis_sd=('normalized_wis', 'std'), seeds=('seed', 'nunique')).reset_index()
    # Configurations are ranked against each other within one stress scenario and task.
    ranks['rank'] = ranks.groupby(['stress', 'task']).score_mean.rank(method='min')
    # Native per-target scores retain their units; never pool admissions with ED.
    target = frame.assign(geography=np.where(frame.location == 'US', 'US', 'states_dc')).groupby(
        [*ids, 'target', 'season', 'geography', 'horizon']).agg(
        n=('wis', 'size'), wis=('wis', 'mean'), crps=('crps', 'mean'),
        coverage_50=('coverage_50', 'mean'), coverage_95=('coverage_95', 'mean')).reset_index()
    return runs, ranks.sort_values(['stress', 'task', 'rank']), target


def postprocess(folder, allow_incomplete=False, plots=False, workers=2):
    from .manager import completed_runs, save
    from .b1_report import report
    from tapestry.model_data.wednesday import WednesdayDataset
    done, versions = completed_runs(folder, allow_incomplete)
    outputs = [folder / r['attempt'] / 'b1' for r in done]
    records = [json.loads((p / 'scoring.json').read_text()) for p in outputs]
    comparable = ['input_sha256', *DATES, 'eval_members', 'retrospective']
    baseline = {k: records[0]['settings'].get(k) for k in comparable}
    if any({k: r['settings'].get(k) for k in comparable} != baseline for r in records[1:]):
        raise ValueError('B1 attempts have different data, date splits or evaluation settings')
    files = [p / f'scores-{json.loads((p / "manifest.json").read_text())["run_id"]}-s{r["seed"]}.parquet'
             for p, r in zip(outputs, done)]
    fingerprint = dict(version=SCORE_VERSION, runs=[dict(path=str(p), sha256=sha(p)) for p in files],
                       scoring=[sha(p / 'scoring.json') for p in outputs])
    digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:12]
    destination = folder / f'{"comparison" if plots else "ranking"}-{digest}'
    destination.mkdir(parents=True, exist_ok=True)
    frame = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    expected = set()
    for row in done:
        scenario = B1Scenario.from_string(row['scenario'])
        tasks = ('forecast',) if scenario.pipeline == 'direct' else ('forecast', 'nowcast')
        expected.update((scenario.run_id, row['seed'], stress, task)
                        for stress in ('natural', 'recent', 'gap', 'outage') for task in tasks)
    actual = set(frame[['config_id', 'seed', 'stress', 'task']].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError('Missing or unexpected B1 run/task/stress-scenario scores')
    runs, ranks, targets = ranking_tables(frame)
    runs.to_csv(destination / 'run_scores.csv', index=False)
    ranks.to_csv(destination / 'configuration_ranking.csv', index=False)
    targets.to_csv(destination / 'target_scores.csv', index=False)
    if records[0]['settings'].get('frozen'):
        from .b1_hubs import rank_hubs, compare_hubs
        matched = rank_hubs(outputs, done, destination)
        if plots and matched:
            compare_hubs(outputs, done, destination, records[0]['settings'], workers)
    if plots:
        check_inputs(records[0]['settings'])
        ds = WednesdayDataset.load(records[0]['settings']['evaluation_dataset'])
        report(frame, ds, destination)
    save(destination / 'manifest.json', dict(**fingerprint, definition=SCORE_DEFINITION,
         attempts=[r['attempt'] for r in done], run_versions=sorted(map(list, versions), key=str)))
    (destination / 'REPORT.md').write_text(
        '# B1 comparison\n\n' + SCORE_DEFINITION + '\n\n'
        'See configuration_ranking.csv, run_scores.csv, target_scores.csv and summary.csv (compare only). '
        'Columns follow B0: config_id is the configuration (B1 run_id) and scenario_string its full b1:v2: string. '
        'The stress column is the evaluation input scenario (natural/recent/gap/outage); configurations are ranked '
        'against each other within one stress scenario and task. '
        'The task column separates nowcast (-2/-1) and forecast (0–3). Direct controls have no nowcast rank. '
        'Hub-relative results, when enabled, are under hub-ranking/ and use narrower matched support.\n')
    print(ranks[ranks.stress == 'natural'][['config_id', 'task', 'score_mean', 'score_sd', 'seeds', 'rank']].to_string(index=False))
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--settings-json', required=True)
    args = parser.parse_args()
    score_run(args.run, json.loads(args.settings_json))
