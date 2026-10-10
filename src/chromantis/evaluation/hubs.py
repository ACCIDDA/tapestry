"""Export a run's saved quantiles (`forecasts.npz`) into hub task tables.

The frozen ensemble support these are scored against was prepared once
(`data/evaluation/b0_hub_comparison_q23`); the offline Hub-mirror extraction code
that built it (`extract_hub`, `GitHubSnapshot`, ...) was deleted 2026-09-22 as it
had no remaining caller (recover it from git history to rebuild the support).
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import json
import numpy as np
import pandas as pd

from chromantis.data.geography import STATE_FIPS
from .quantiles import LEVELS

KEY = ['reference_date', 'target_end_date', 'location', 'horizon']
QCOLS = [f'q{q:g}' for q in LEVELS]
def export(run):
    """Hub task tables for one saved run: exact mapping context_end+7d -> reference."""
    postal_to_fips = {v: k for k, v in STATE_FIPS.items()} | {'US': 'US'}
    frames = {}
    manifest = json.loads((Path(run) / 'manifest.json').read_text())
    from chromantis.model.scenario import Scenario
    from chromantis.problem import Problem
    problem=Problem.load(manifest['problem'])
    if 'trained_targets' in manifest:
        trained=set(map(int,manifest['trained_targets']))
    else:
        scenario=Scenario.from_string(manifest['scenario'])
        trained={c for c,w in enumerate(problem.target_weights(scenario.loss_weights)) if w}
    for held in manifest['folds']:
        with np.load(Path(run) / f'eval_{held}' / 'forecasts.npz', allow_pickle=False) as data:
            if not np.array_equal(data['quantile_levels'], LEVELS):  # training.evaluate saves exactly LEVELS
                raise ValueError(f'{run}/eval_{held}/forecasts.npz holds other quantile levels than LEVELS')
            selected = data['quantiles']
            for c, signal in enumerate(problem.target_signals):
                target=signal.hub_target
                if not target or c not in trained:
                    continue
                q = selected[:, :, :, c, :]
                n, h, l = q.shape[1:]
                reference = [(date.fromisoformat(d) + timedelta(weeks=1)).isoformat() for d in data['context_end']]
                frame = pd.DataFrame({
                    'reference_date': np.repeat(reference, h * l),
                    'target_end_date': np.repeat(data['target_dates'].reshape(-1), l),
                    'location': np.tile([postal_to_fips[v] for v in data['locations']], n * h),
                    'horizon': np.tile(np.repeat(np.arange(h), l), n),
                    'model_original_truth': data['truth'][:, :, c, :].reshape(-1),
                    'model_original_mask': data['mask'][:, :, c, :].reshape(-1),
                })
                frame[QCOLS] = q.reshape(len(LEVELS), -1).T
                keep = problem.fold_labels(frame.target_end_date) == held
                frames[(held, target)] = frame[keep].copy()
    return frames
