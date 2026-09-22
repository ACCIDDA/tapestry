"""Leave-one-season-out split, generalized to the array schema in `dataset.build`.

Replaces `models/season_cv.py`'s `fold_data`/`masked_episodes`/`validation_split`:
one function, reused by both `finalized.npz`/`vintaged.npz` and by
`rank`/`compare` in `experiment.planner`, instead of bespoke per-model tensors.
"""
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

SEASONS = ('2023-2024', '2024-2025', '2025-2026')


def season(day):
    """CDC epiweek 31-30, as used throughout the project (not challenge dates)."""
    def boundary(year):
        jan4 = date(year, 1, 4)
        return jan4 - timedelta(days=(jan4.weekday() + 1) % 7) + timedelta(weeks=30)
    day = day if isinstance(day, date) else date.fromisoformat(str(day)[:10])
    year = day.year if day >= boundary(day.year) else day.year - 1
    return f'{year}-{year + 1}'


@dataclass(frozen=True)
class Split:
    train: np.ndarray   # boolean mask over the array's leading time axis
    val: np.ndarray
    score: np.ndarray


def season_split(dates, held_out_season, val_weeks=3, val_spacing=16, val_offset=4):
    """`dates` is one calendar label per row of the array's leading time axis.

    `train` excludes the held-out season and, within each remaining season,
    hides `val_weeks` consecutive weeks out of every `val_spacing`, starting at
    `val_offset` (B0's early-stopping pattern: 3-in-16, offset 4). `val` is
    exactly those hidden weeks. `score` is the held-out season.
    """
    if held_out_season not in SEASONS:
        raise ValueError(f'Unknown season {held_out_season}; expected one of {SEASONS}')
    labels = np.array([season(d) for d in dates])
    score = labels == held_out_season
    train = np.isin(labels, [s for s in SEASONS if s != held_out_season])
    val = np.zeros_like(train)
    for label in SEASONS:
        if label == held_out_season:
            continue
        index = np.flatnonzero(labels == label)
        position = np.arange(len(index)) % val_spacing
        hidden = (position >= val_offset) & (position < val_offset + val_weeks)
        val[index[hidden]] = True
    train = train & ~val
    return Split(train=train, val=val, score=score)
