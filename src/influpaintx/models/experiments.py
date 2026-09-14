"""Small, independent switches for the first three B0 experiments."""
import csv
from pathlib import Path

import numpy as np

LOSS_WEIGHTS = {
    'influenza_first': [1, .1, .1, .1, .1, .1],
    'balanced_admissions': [1, 1, 1, .1, .1, .1],
    'flu_only': [1, 0, 0, 0, 0, 0],
}


def add_experiment_args(parser):
    parser.add_argument('--count-transform', choices=['raw', 'sqrt', 'fourth_root'], default='raw')
    parser.add_argument('--geography', action='store_true', help='Include log population and native US flag')
    parser.add_argument('--dynamics', action='store_true', help='Include slopes, acceleration, observation age and Christmas timing')
    parser.add_argument('--population-file', default='data/metadata/b0_locations.csv')
    parser.add_argument('--loss-weights', choices=list(LOSS_WEIGHTS), default='influenza_first')


def model_options(episodes, args):
    transform = getattr(args, 'count_transform', 'raw')
    geography = getattr(args, 'geography', False)
    options = dict(count_transform=transform, geography=geography, dynamics=getattr(args, 'dynamics', False))
    if transform == 'raw' and not geography:
        return options
    with Path(args.population_file).open() as stream:
        rows = list(csv.DictReader(stream))
    populations = {}
    for row in rows:
        loc, value = row.get('abbreviation', row['location']), float(row['population'])
        if loc in populations or not np.isfinite(value) or value <= 0:
            raise ValueError(f'Invalid or duplicate population for {loc}')
        populations[loc] = value
    locations = episodes[0]['locations']
    missing = set(locations) - populations.keys()
    if missing:
        raise ValueError(f'Missing population for locations: {sorted(missing)}')
    options['populations'] = {loc: populations[loc] for loc in locations}
    if transform == 'raw':
        return options
    # Count each context date once; do not overweight dates occurring in more windows.
    # The caller has already masked all observations outside the fitting partition.
    by_date = {}
    for episode in episodes:
        if episode['locations'] != locations:
            raise ValueError('Training episodes must share location order')
        for day, row in zip(episode['context_dates'], episode['X']):
            by_date[day] = row
    panel = np.stack(list(by_date.values()))
    pop = np.array([populations[loc] for loc in locations])
    power = .5 if transform == 'sqrt' else .25
    scales = []
    for c in range(6):
        values = panel[:, c, 0]
        if c < 3:
            values = np.maximum(values, 0) * 100000 / pop
            values = values ** power
        valid = values[panel[:, c, 1].astype(bool)]
        scales.append(max(float(np.quantile(valid, .95)) if valid.size else 0, .001))
    options['input_scale'] = scales
    return options
