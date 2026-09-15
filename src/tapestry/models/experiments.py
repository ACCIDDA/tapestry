"""Independent B0 representation, supervision, and architecture switches."""
import csv
from pathlib import Path

import numpy as np

LOSS_WEIGHTS = {
    'influenza_first': [1, .1, .1, .1, .1, .1],
    'balanced_admissions': [1, 1, 1, .1, .1, .1],
    'flu_only': [1, 0, 0, 0, 0, 0],
    # Matches the selection score: each admissions target counts twice an ED target.
    'objective': [1, 1, 1, .5, .5, .5],
}
COUNT_TRANSFORMS = ('raw', 'rate', 'sqrt', 'fourth_root', 'log1p')
ED_TRANSFORMS = ('linear', 'logit', 'fourth_root')


def add_experiment_args(parser):
    parser.add_argument('--count-transform', choices=COUNT_TRANSFORMS, default='raw',
                        help='raw counts, or rate per 100,000 with no transform, square/fourth root, or log1p')
    parser.add_argument('--ed-transform', choices=ED_TRANSFORMS, default='linear',
                        help='linear inputs with a logit residual (original), logit, or fourth root')
    parser.add_argument('--geography', action='store_true', help='Include log population and native US flag')
    parser.add_argument('--dynamics', action='store_true', help='Include slopes, acceleration, observation age and Christmas timing')
    parser.add_argument('--population-file', default='data/metadata/b0_locations.csv')
    parser.add_argument('--loss-weights', choices=list(LOSS_WEIGHTS), default='influenza_first')
    parser.add_argument('--encoder', choices=['mlp', 'conv'], default='mlp')
    parser.add_argument('--spatial', choices=['none', 'attention'], default='none',
                        help='One attention block across locations at the same forecast date')
    parser.add_argument('--heads', choices=['shared', 'state_us'], default='shared')
    parser.add_argument('--decoder', choices=['legacy', 'residual2'], default='legacy')
    parser.add_argument('--noise', choices=['global', 'local'], default='global',
                        help='Global latent only, or global plus a per-location latent')
    parser.add_argument('--us-error', choices=['none', 'shared_factor'], default='none',
                        help='shared_factor adds a per-episode, per-channel common mode to every '
                             'location, so state errors correlate instead of cancelling into the US')
    parser.add_argument('--latent', type=int, default=16)


def model_options(episodes, args):
    transform = getattr(args, 'count_transform', 'raw')
    ed_transform = getattr(args, 'ed_transform', 'linear')
    geography = getattr(args, 'geography', False)
    options = dict(count_transform=transform, ed_transform=ed_transform, geography=geography,
                   dynamics=getattr(args, 'dynamics', False))
    options.update(encoder=getattr(args, 'encoder', 'mlp'), spatial=getattr(args, 'spatial', 'none'),
                   heads=getattr(args, 'heads', 'shared'), decoder=getattr(args, 'decoder', 'legacy'),
                   noise=getattr(args, 'noise', 'global'), us_error=getattr(args, 'us_error', 'none'),
                   latent=getattr(args, 'latent', 16))
    populations = None
    if transform != 'raw' or geography:
        with Path(args.population_file).open() as stream:
            rows = list(csv.DictReader(stream))
        populations = {}
        for row in rows:
            loc, value = row.get('abbreviation', row['location']), float(row['population'])
            if loc in populations or not np.isfinite(value) or value <= 0:
                raise ValueError(f'Invalid or duplicate population for {loc}')
            populations[loc] = value
        missing = set(episodes[0]['locations']) - populations.keys()
        if missing:
            raise ValueError(f'Missing population for locations: {sorted(missing)}')
        options['populations'] = {loc: populations[loc] for loc in episodes[0]['locations']}
    # Raw counts in linear ED space used to skip this and fall back to the pooled
    # native-unit scale, which is exactly the path that put the US ~40x out of range.
    # Every configuration now gets per-location, per-channel input scales.
    options.update(input_scales(episodes, transform, ed_transform, options.get('populations')))
    return options


def input_scales(episodes, transform, ed_transform, populations):
    """Training-only scales (and logit centers) in each channel's model space."""
    import torch
    from .b0 import transform_counts, transform_proportions
    locations = episodes[0]['locations']
    # Count each context date once; do not overweight dates occurring in more windows.
    # The caller has already masked all observations outside the fitting partition.
    by_date = {}
    for episode in episodes:
        if episode['locations'] != locations:
            raise ValueError('Training episodes must share location order')
        for day, row in zip(episode['context_dates'], episode['X']):
            by_date[day] = row
    panel = torch.as_tensor(np.stack(list(by_date.values())))  # date, channel, (value, mask), location
    valid = panel[:, :, 1].bool()
    values = torch.where(valid, panel[:, :, 0], 0)
    population = torch.tensor([populations[loc] for loc in locations]) if populations else None
    transformed = torch.cat((transform_counts(values[:, :3], population, transform),
                             transform_proportions(values[:, 3:], ed_transform)), 1).numpy()
    valid = valid.numpy()
    # One scale per channel AND location. Pooling locations set a single scale from
    # state-sized counts, so the US entered the model ~40x too large and its intervals
    # collapsed; each location is now normalized by its own history. Falls back to the
    # pooled value where a location has no observations of a channel.
    scales, offsets = [], []
    for c in range(6):
        pooled = transformed[:, c][valid[:, c]]
        logit_channel = c >= 3 and ed_transform == 'logit'
        floor = .1 if logit_channel else (1 if c < 3 and transform == 'raw' else .001)
        if logit_channel:
            fallback = max(float(pooled.std()) if pooled.size else 0, floor)
            pooled_offset = float(pooled.mean()) if pooled.size else 0.
        else:
            fallback = max(float(np.quantile(pooled, .95)) if pooled.size else 0, floor)
            pooled_offset = 0.
        channel_scales, channel_offsets = [], []
        for li in range(len(locations)):
            observed = transformed[:, c, li][valid[:, c, li]]
            if not observed.size:
                channel_scales.append(fallback)
                channel_offsets.append(pooled_offset)
            elif logit_channel:
                channel_offsets.append(float(observed.mean()))
                channel_scales.append(max(float(observed.std()), floor))
            else:
                channel_offsets.append(0.)
                channel_scales.append(max(float(np.quantile(observed, .95)), floor))
        scales.append(channel_scales)
        offsets.append(channel_offsets)
    result = dict(input_scale=scales)
    if ed_transform == 'logit':
        result['input_offset'] = offsets
    return result
