"""Direct train/predict commands for the finalized Build B0 research pilot."""
import argparse
from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from tapestry.model_data import CHANNELS, FinalizedDataset
from .b0 import B0, fair_crps
from .experiments import LOSS_WEIGHTS, add_experiment_args, model_options


def calendar(days, dynamics=False):
    phase = np.array([date.fromisoformat(day).timetuple().tm_yday for day in days]) * (2 * np.pi / 365.25)
    columns = [np.sin(phase), np.cos(phase)]
    if dynamics:
        # Christmas in the July–June winter containing the context date; units /26 weeks.
        dates = [date.fromisoformat(day) for day in days]
        columns.append(np.array([(d - date(d.year if d.month >= 7 else d.year - 1, 12, 25)).days / (7 * 26) for d in dates]))
    return np.stack(columns, axis=-1).astype('float32')


def fit(episodes, scales, args, options=None):
    """Fit fixed hyperparameters on an already selected set of weekly episodes."""
    model = B0(args.lookback, args.horizons, args.width, scale=scales, **(model_options(episodes, args) if options is None else options)).to(args.device)
    x = torch.tensor(np.stack([e['X'] for e in episodes]), device=args.device)
    y = torch.tensor(np.stack([e['Y'] for e in episodes]), device=args.device)
    cal = torch.tensor(calendar([e['context_dates'][-1] for e in episodes], model.config['dynamics']), device=args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    for epoch in range(args.epochs):
        order = torch.randperm(len(episodes), device=args.device)
        total = 0
        for ids in order.split(args.batch_size):
            optimizer.zero_grad()
            samples = model(x[ids], cal[ids], args.members, locations=episodes[0]['locations'])
            scores = fair_crps(samples, y[ids, :, :, 0, :], y[ids, :, :, 1, :])
            # Native-unit CRPS after inversion; fixed channel Q95 across weight ablations.
            weights = scores.new_tensor(LOSS_WEIGHTS[getattr(args, 'loss_weights', 'influenza_first')])
            loss = (weights * scores / model.scale).sum()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += loss.item() * len(ids)
        history.append(total / len(episodes))
        print(json.dumps({'epoch': epoch + 1, 'loss': history[-1]}), flush=True)
    return model, history


def train(args):
    torch.manual_seed(args.seed)
    ds = FinalizedDataset.load(args.dataset)
    # Origins, labels and scale fitting stop at train_end; earlier history may be context.
    episodes = list(ds.windows(start=args.train_start, end=args.train_end,
                              target_start=args.train_start, target_end=args.train_end,
                              lookback=args.lookback, horizons=tuple(args.horizons)))
    if not episodes:
        raise ValueError('No supervised windows in training interval')
    selected = ds.panel[[args.train_start <= day <= args.train_end for day in ds.dates]]
    scales = []
    for c in range(6):
        valid = selected[:, c, 0, :][selected[:, c, 1, :].astype(bool)]
        scales.append(max(float(np.quantile(valid, .95)) if valid.size else 0, 1 if c < 3 else .001))
    # Earlier dates may supply context, but input scalers use only the fit interval.
    # Scale fitting is separate from the actual context passed to the model.
    scale_episodes = []
    for e in episodes:
        e = {**e, 'X': e['X'].copy()}
        for i, day in enumerate(e['context_dates']):
            if not args.train_start <= day <= args.train_end:
                e['X'][i] = 0
        scale_episodes.append(e)
    options = model_options(scale_episodes, args)
    model, history = fit(episodes, scales, args, options=options)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {'train_start': args.train_start, 'train_end': args.train_end,
                'seed': args.seed, 'epochs': args.epochs, 'episodes': len(episodes),
                'dataset_sha256': hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
                'channels': list(CHANNELS), 'locations': list(ds.locations),
                'loss': 'fair CRPS in native units / training Q95',
                'loss_weights': LOSS_WEIGHTS[args.loss_weights], 'experiment': model.config,
                'history': history, 'torch_version': str(torch.__version__),
                'parameters': sum(p.numel() for p in model.parameters())}
    torch.save({'config': model.config, 'state_dict': model.cpu().state_dict(), 'metadata': metadata}, output)
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'checkpoint': str(output), 'parameters': metadata['parameters']}))


def predict(args):
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = B0(**checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'])
    model.to(args.device).eval()
    ds = FinalizedDataset.load(args.dataset)
    q = ds.query(args.context_end, lookback=model.config['lookback'],
                 horizons=tuple(model.config['horizons']), locations=args.locations)
    x = torch.tensor(q['X'][None], device=args.device)
    cal = torch.tensor(calendar([args.context_end], model.config['dynamics']), device=args.device)
    with torch.no_grad():
        samples = torch.cat([model(x, cal, min(args.sample_batch, args.members - i), locations=q['locations']).cpu()
                             for i in range(0, args.members, args.sample_batch)], dim=0).numpy()[:, 0]
    from .quantiles import LEVELS
    levels = LEVELS
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as stream:
        np.savez_compressed(stream, samples=samples, quantiles=np.quantile(samples, levels, axis=0),
                            quantile_levels=levels, target_dates=q['target_dates'],
                            channels=CHANNELS, locations=q['locations'], context_dates=q['context_dates'],
                            metadata=json.dumps({'checkpoint': str(Path(args.checkpoint).resolve()),
                                                 'fit': checkpoint['metadata'], 'seed': args.seed,
                                                 'axes': ['member', 'horizon', 'channel', 'location']}))
    print(json.dumps({'output': str(output), 'samples_shape': samples.shape,
                      'mean_sample_std': float(samples.std(axis=0).mean())}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('train', 'predict'):
        p = sub.add_parser(name)
        p.add_argument('--dataset', default='data/processed/build_b_finalized.npz')
        p.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])
        p.add_argument('--seed', type=int, default=42)
        p.add_argument('--members', type=int, default=8 if name == 'train' else 256)
        p.add_argument('--output', default='data/processed/b0.pt' if name == 'train' else 'data/processed/b0_predictions.npz')
        if name == 'train':
            add_experiment_args(p)
            p.add_argument('--train-start', default='2023-09-01')
            p.add_argument('--train-end', required=True)
            p.add_argument('--lookback', type=int, default=8)
            p.add_argument('--horizons', type=int, nargs='+', default=[1, 2, 3, 4])
            p.add_argument('--epochs', type=int, default=50)
            p.add_argument('--batch-size', type=int, default=8)
            p.add_argument('--width', type=int, default=64)
            p.add_argument('--lr', type=float, default=.001)
        else:
            p.add_argument('--checkpoint', default='data/processed/b0.pt')
            p.add_argument('--context-end', required=True)
            p.add_argument('--locations', nargs='+')
            p.add_argument('--sample-batch', type=int, default=32)
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    if args.members < (2 if args.command == 'train' else 1):
        parser.error('Need >=2 training members or >=1 prediction member')
    if args.command == 'train' and min(args.epochs, args.batch_size, args.width, args.lookback) < 1:
        parser.error('epochs, batch-size, width and lookback must be positive')
    if args.command == 'predict' and args.sample_batch < 1:
        parser.error('sample-batch must be positive')
    (train if args.command == 'train' else predict)(args)
