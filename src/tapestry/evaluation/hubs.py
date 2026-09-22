"""Export a run's saved quantiles (`forecasts.npz`) into hub task tables.

The frozen ensemble support these are scored against was prepared once
(`data/evaluation/b0_hub_comparison_q23`); the offline Hub-mirror extraction code
that built it (`extract_hub`, `GitHubSnapshot`, ...) was deleted 2026-09-22 as it
had no remaining caller (recover it from git history to rebuild the support).
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from tapestry.data.geography import STATE_FIPS
from tapestry.dataset.cv import SEASONS, season
from .quantiles import LEVELS, select_quantiles

KEY = ['reference_date', 'target_end_date', 'location', 'horizon']
QCOLS = [f'q{q:g}' for q in LEVELS]
CHANNEL = {'wk inc flu hosp': 0, 'wk inc covid hosp': 1, 'wk inc rsv hosp': 2,
           'wk inc flu prop ed visits': 3, 'wk inc covid prop ed visits': 4, 'wk inc rsv prop ed visits': 5}


def export(run):
    """Hub task tables for one saved run: exact mapping context_end+7d -> reference."""
    postal_to_fips = {v: k for k, v in STATE_FIPS.items()} | {'US': 'US'}
    frames = {}
    for held in SEASONS:
        with np.load(Path(run) / f'eval_{held}' / 'forecasts.npz', allow_pickle=False) as data:
            selected = select_quantiles(data['quantiles'], data['quantile_levels'])
            for target, c in CHANNEL.items():
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
                keep = frame.target_end_date.map(lambda d: season(d)) == held
                frames[(held, target)] = frame[keep].copy()
    return frames
