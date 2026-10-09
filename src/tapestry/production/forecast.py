"""Run a released model on one Wednesday's reported data: no fitting, no future labels.

A release (`production/releases/<name>.json`) lists the exact fitted checkpoints of each
recipe, with the SHA256 of each `model.pt`:

    {"name": "b7-20261007", "model_abbr": "Chromantis", "hub": "flusight", "rule": "mixture",
     "view": "corrected", "description": "training seasons, input treatments, labels ...",
     "recipes": {"A_blocks3": [{"checkpoint": "data/production-fits/.../eval_2026-2027",
                                "sha256": "..."}, ...]}}

`forecast_checkpoint` replays one checkpoint: the saved network, input scales and
correction trees on the issuance's reported inputs (`corrected` view: newest weeks
corrected by the saved trees; `raw`: as reported), 512 sampled members, seed + 1000.
Writes `forecasts.npz` (marginal quantiles) and `operational-manifest.json`.

History: scripts/forecast_b6.py and the B6/B7 replay scripts (2026-10-07), unchanged in
what they compute.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import pickle
from pathlib import Path

import torch

from tapestry.dataset.build import load, for_hub, covariate_names_for, context_end
from tapestry.dataset.episodes import episodes
from tapestry.experiment.fit import ForecastSlice
from tapestry.experiment.provenance import sha256
from tapestry.experiment.training import evaluate
from tapestry.model.network import load_model
from tapestry.model.scenario import Scenario

MEMBERS = 512


def read_release(path):
    release = json.loads(Path(path).read_text())
    for key in ('name', 'model_abbr', 'hub', 'rule', 'view', 'description', 'recipes'):
        if key not in release:
            raise ValueError(f'{path}: release lacks {key!r}')
    return release


def forecast_checkpoint(checkpoint, dataset, issuance, output, view='corrected', members=MEMBERS, device='cpu',
                        expected_sha256=None):
    checkpoint, output = Path(checkpoint), Path(output)
    if expected_sha256 and sha256(checkpoint / 'model.pt') != expected_sha256:
        raise ValueError(f'{checkpoint}/model.pt differs from the released checkpoint')
    metadata = json.loads((checkpoint / 'manifest.json').read_text())
    scenario = Scenario.from_string(metadata['scenario'])
    if scenario.evaluation_seasons != 'production' or metadata['held_out_season'] != '2026-2027':
        raise ValueError('Use a production fit trained through 2025-26')
    if scenario.correction_noise:
        raise ValueError('Operational replay does not implement noisy-history recipes')
    panel = for_hub(load(dataset), 'flusight')
    panel_meta = json.loads(str(panel['metadata']))
    if panel_meta.get('omitted_sources'):
        if scenario.pathogen_inputs != 'flu':
            raise ValueError('Source-restricted operational panel requires audited flu input scope')
        required = {'nhsn_flu_admissions', 'nssp_flu_proportion'}
        required.update('nwss' if n.startswith('nwss_') else n for n in covariate_names_for(scenario.covariate_set))
        missing = required - set(panel_meta['available_sources'])
        if missing:
            raise ValueError(f'Operational panel omits model inputs: {sorted(missing)}')
    if issuance not in map(str, panel['issuance_dates']):
        raise ValueError('Requested issuance absent from refreshed panel')
    eps = episodes(panel, scenario.lookback, 'reported', covariate_names_for(scenario.covariate_set),
                   horizons=(1, 2, 3, 4), require_labels=False)
    eps = [e for e in eps if e['issuance'] == issuance]
    if len(eps) != 1 or eps[0]['context_dates'][-1] != context_end(issuance):
        raise ValueError('Operational issuance/context alignment failed')
    if eps[0]['target_available'].any():
        raise ValueError('Operational future outcomes unexpectedly available; audit requested issuance')
    if scenario.forecast_view != view:
        raise ValueError(f'{checkpoint}: the release view {view!r} differs from the recipe forecast_view '
                         f'{scenario.forecast_view!r}, the view it was ranked in')
    if view == 'corrected':
        with (checkpoint / 'nowcaster.pkl').open('rb') as stream:
            eps = pickle.load(stream).apply_batch(eps)
    elif view != 'raw':
        raise ValueError('Operational views are raw or corrected (half is not implemented in production)')
    model = load_model(torch.load(checkpoint / 'model.pt', map_location='cpu', weights_only=False)).to(device)
    if scenario.reconstruction_labels:
        model = ForecastSlice(model, scenario.future)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(metadata['seed'] + 1000)
    evaluate(model, eps, members, device, output)
    record = dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha256(checkpoint / 'model.pt'),
                  operational_dataset=str(Path(dataset).resolve()), operational_dataset_sha256=sha256(dataset),
                  issuance=issuance, input_view=view, seed=metadata['seed'], scenario=metadata['scenario'],
                  training_fold=metadata['fold'], members=members, future_labels_used=False)
    (output / 'operational-manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    return output


def forecast_release(release, dataset, issuance, out, device='cpu', jobs=4):
    """Replay every checkpoint of a release into `out/<recipe>/s<seed>/`; finished members are reused."""
    tasks = []
    for recipe, members in release['recipes'].items():
        for member in members:
            seed = json.loads((Path(member['checkpoint']) / 'manifest.json').read_text())['seed']
            target = Path(out) / recipe / f's{seed}'
            if not (target / 'operational-manifest.json').exists():
                tasks.append((member['checkpoint'], target, member['sha256']))
    print(f'{sum(map(len, release["recipes"].values()))} members; {len(tasks)} to forecast', flush=True)
    run = lambda task: forecast_checkpoint(task[0], dataset, issuance, task[1], release['view'], device=device,
                                           expected_sha256=task[2])
    with ThreadPoolExecutor(jobs) as pool:
        list(pool.map(run, tasks))
    return {recipe: [Path(out) / recipe / f's{json.loads((Path(m["checkpoint"]) / "manifest.json").read_text())["seed"]}'
                     for m in members] for recipe, members in release['recipes'].items()}
