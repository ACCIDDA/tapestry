"""Frozen forward-development benchmark; reuse B0 target MLP and B1 gate."""
import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from tapestry.model_data.forward import CUTOFF, DATASET, POPULATIONS, TEST_START
from tapestry.model_data.wednesday import WednesdayDataset
from .b1 import B1
from .b1_run import arrays, known_finals, populations, task_weights, unique_truth
from .b0 import fair_crps_cells
from .experiments import input_scales
from .objective import loss_scales
from .quantiles import LEVELS
from .provenance import save


@dataclass(frozen=True)
class ForwardScenario:
    candidate: str
    epochs: int = 100
    width: int = 64
    members: int = 128
    batch_size: int = 8
    fit_partition: str = 'target'
    encoder: str = 'mlp'
    decoder: str = 'legacy'
    spatial: str = 'none'
    lookback: int = 12

    @property
    def scenario_string(self):
        return f'forward:{self.candidate}:{self.epochs}:{self.width}:{self.members}:{self.batch_size}'

    @classmethod
    def from_string(cls, value):
        _, candidate, *budget = value.split(':')
        if candidate not in ('direct', 'joint', 'separate'):
            raise ValueError(candidate)
        return cls(candidate, *map(int, budget))

    @property
    def run_id(self):
        return self.scenario_string.replace(':', '-')


class ForwardBackend:
    model = 'Forward'
    output_dir = 'forward'

    def scenarios(self, args):
        budget = {k: getattr(args, k) for k in ('epochs', 'width', 'members', 'batch_size') if getattr(args, k) is not None}
        return {c: ForwardScenario(c, **budget) for c in ('direct', 'joint', 'separate')}

    def settings(self, args):
        return dict(model=self.model, protocol='forward_2025_v1')

    def check_inputs(self, settings):
        from .backends import check_frozen, sha
        ds = WednesdayDataset.load(settings['dataset'])
        if ds.metadata['training_cutoff'] != CUTOFF or ds.metadata['protocol'] != 'forward_2025_v1':
            raise ValueError('Expected cutoff-pinned forward data')
        if settings['eval_members'] < 2:
            raise ValueError('Need at least two evaluation samples')
        check_frozen(settings['frozen'])
        for path, digest in settings.get('input_sha256', {}).items():
            if sha(path) != digest:
                raise ValueError(f'Pinned input changed: {path}')

    def prepare(self, folder, settings):
        from .backends import snapshot
        snapshot(folder, settings, extra=[Path(__file__).resolve().parents[3] / 'docs/workflows/forward-2025.md'])

    def fit_commands(self, scenario, seed, settings, output):
        return [[sys.executable, '-m', 'tapestry.models.forward', '--scenario', scenario.scenario_string,
                 '--seed', str(seed), '--dataset', settings['dataset'], '--population-file', settings['population_file'],
                 '--device', settings['device'], '--members', str(settings['eval_members']), '--output', str(output)]]

    def collect_manifest(self, output):
        pass

    def required_artifacts(self, output):
        return ['manifest.json', 'model.pt', 'forecasts.npz', 'totals.csv', 'forecast-cells.parquet',
                'full-forecast-cells.parquet', 'recent-cells.parquet', 'training.json']

    def complete(self, output):
        return json.loads((output / 'manifest.json').read_text()).get('protocol') == 'forward_2025_v1'


def episodes(ds, training, finalized=False):
    result = []
    for e in ds.episodes(end=CUTOFF if training else None, start=None if training else TEST_START, supervised=False):
        i = e['index']
        # Eligibility is separate from label availability and never changes forecast labels.
        e['Y'][:2, :, 1] *= ds.arrays['revision_eligible'][i] | ds.arrays['reconstruction_eligible'][i]
        if finalized:
            valid = ds.arrays['X_cutoff_valid'][i]
            e['X'] = np.stack((np.where(valid, ds.arrays['X_cutoff_final'][i], 0), valid, valid), axis=2)
        # Zero all masked storage before it enters transforms or losses.
        e['X'][:, :, 0] = np.where(e['X'][:, :, 1], e['X'][:, :, 0], 0)
        e['Y'][:, :, 0] = np.where(e['Y'][:, :, 1], e['Y'][:, :, 0], 0)
        result.append(e)
    return result


def fit(train, options, scenario, target, seed, device, stage):
    torch.manual_seed(seed + 10000 * target)
    recent = stage == 'nowcast'
    joint = stage == 'joint'
    # Identical gated recent architecture for standalone and joint nowcasters.
    model = B1(target=target, direct=not (recent or joint), parallel_recent=recent or joint,
               revision_bridge=recent or joint, supplied_final=True, **options).to(device)
    train = [e for e in train if e['Y'][:2 if recent else 6, target, 1].any()] if recent else [e for e in train if e['Y'][2:, target, 1].any()]
    x, a, y, cal = arrays(train, device)
    known = torch.as_tensor(known_finals(train), device=device)
    if recent:
        weights = task_weights(train, target)[:, :2, target:target+1] * 2
        y = y[:, :2, target:target+1]
    elif joint:
        weights = task_weights(train, target)[:, :, target:target+1]
        weights[:, :2] *= .5  # future + .25 recent
        weights[:, 2:] *= 2
        y = y[:, :, target:target+1]
    else:
        weights = task_weights(train, target, direct=True)[:, :, target:target+1]
        y = y[:, 2:, target:target+1]
    weights = torch.as_tensor(weights, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    history = []
    for epoch in range(scenario.epochs):
        total = 0.
        for ids in torch.randperm(len(train), device=device).split(scenario.batch_size):
            optimizer.zero_grad()
            samples = model(x[ids], a[ids], cal[ids], scenario.members, known_final=known[ids])
            if recent:
                samples = samples[:, :, :2]
            score = fair_crps_cells(samples, y[ids, :, :, 0], y[ids, :, :, 1])
            loss = (weights[ids] * score / model.scale[target:target+1]).sum() * len(train) / len(ids)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite forward loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += float(loss.detach()) * len(ids) / len(train)
        history.append(total)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(json.dumps(dict(stage=stage, target=target, epoch=epoch+1, loss=total)), flush=True)
    return model.cpu().eval(), dict(stage=stage, target=target, episodes=len(train), loss=history)


@torch.no_grad()
def predict(models, nowcasters, test, members, device, seed):
    torch.manual_seed(seed + 900000)
    n, l = len(test), len(test[0]['locations'])
    result = np.full((len(LEVELS), n, 6, 6, l), np.nan, np.float32)
    models = [m.to(device) for m in models]
    nowcasters = [m.to(device) for m in nowcasters]
    # One issuance at a time bounds the sampled pipeline's expanded batch.
    for i, e in enumerate(test):
        x, a, _, cal = arrays([e], device)
        known = torch.as_tensor(known_finals([e]), device=device)
        if nowcasters:
            recent = torch.cat([m(x, a, cal, members, known_final=known)[:, :, :2] for m in nowcasters], dim=3)
            px = x.expand(members, -1, -1, -1).clone()
            pa = a.expand_as(px).clone()
            pk = known.expand_as(px).clone()
            px[:, -2:] = recent[:, 0]
            pa[:, -2:] = True
            # These are sampled finalized-value hypotheses, not certified reports.
            # Setting the model's supplied-reference flags matches its training representation.
            pk[:, -2:] = True
            future = torch.cat([m(px, pa, cal.expand(members, -1), 1, known_final=pk)[0] for m in models], dim=2)
            samples = torch.cat((recent, future[:, None]), dim=2)
        else:
            samples = torch.cat([m(x, a, cal, members, known_final=known) for m in models], dim=3)
        q = np.quantile(samples.cpu().numpy(), LEVELS, axis=0)[:, 0]
        result[:, i, -q.shape[1]:] = q
    return result


def train(args):
    torch.set_num_threads(int(os.environ.get('TAPESTRY_TORCH_THREADS', '1')))
    scenario = ForwardScenario.from_string(args.scenario)
    ds = WednesdayDataset.load(args.dataset)
    natural = episodes(ds, True)
    final = episodes(ds, True, finalized=True)
    test = episodes(ds, False)
    pop = populations(args.population_file, ds.locations)
    # Common transformations use the same cutoff-final training contexts for all candidates.
    options = dict(populations=pop, locations=list(ds.locations), width=scenario.width,
        lookback=12, latent=16, count_transform='fourth_root', ed_transform='logit',
        geography=True, dynamics=True, annual_calendar=True,
        scale=loss_scales(unique_truth(natural)), **input_scales(final, 'fourth_root', 'logit', pop))
    models, nowcasters, records = [], [], []
    if scenario.candidate == 'separate':
        for c in range(6):
            model, record = fit(natural, options, scenario, c, args.seed, args.device, 'nowcast')
            nowcasters.append(model); records.append(record)
    for c in range(6):
        model, record = fit(final if scenario.candidate == 'separate' else natural, options,
                            scenario, c, args.seed, args.device, 'joint' if scenario.candidate == 'joint' else 'forecast')
        models.append(model); records.append(record)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    save(out / 'training.json', records)
    torch.save(dict(models=[dict(config=m.config, state_dict=m.state_dict()) for m in models],
                    nowcasters=[dict(config=m.config, state_dict=m.state_dict()) for m in nowcasters]), out / 'model.pt')
    quantiles = predict(models, nowcasters, test, args.members, args.device, args.seed)
    np.savez_compressed(out / 'forecasts.npz', quantiles=quantiles, quantile_levels=LEVELS,
                        indices=[e['index'] for e in test], scales=np.asarray(options['scale']))
    save(out / 'manifest.json', dict(model='Forward', protocol='forward_2025_v1', candidate=scenario.candidate,
        scenario=scenario.scenario_string, seed=args.seed, dataset=args.dataset, dataset_metadata=ds.metadata,
        epochs=scenario.epochs, eval_members=args.members, recent_weight=.25, calibration='none'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--dataset', default=DATASET)
    parser.add_argument('--population-file', default=POPULATIONS)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--members', type=int, default=256)
    parser.add_argument('--output', required=True)
    train(parser.parse_args())
