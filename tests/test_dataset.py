from datetime import date, timedelta

import numpy as np
import pytest

from tapestry.dataset.splits import SEASONS, season, season_split
from tapestry.dataset.build import covariate_names_for, COVARIATE_GROUPS, SOURCE_GROUPS


def test_season_boundary_and_53_week_year():
    assert season(date(2023, 8, 5)) == '2023-2024'
    assert season(date(2023, 7, 29)) == '2022-2023'
    assert season(date(2021, 1, 2)) == '2020-2021'


DAYS = tuple((date(2023, 9, 2) + timedelta(weeks=i)).isoformat() for i in range(157))


def test_season_split_excludes_held_out_and_hides_a_fixed_share_of_each_training_season():
    for held_out in SEASONS:
        split = season_split(DAYS, held_out)
        assert not (split.train & split.score).any()
        assert not (split.val & split.score).any()
        assert not (split.train & split.val).any()
        labels = np.array([season(d) for d in DAYS])
        assert (labels[split.score] == held_out).all()
        assert held_out not in set(labels[split.train]) | set(labels[split.val])
        for label in SEASONS:
            if label == held_out:
                continue
            share = split.val[labels == label].mean()
            assert .15 < share < .2


def test_season_split_hidden_weeks_come_in_runs_of_at_most_three():
    split = season_split(DAYS, SEASONS[1])
    hidden_days = sorted(date.fromisoformat(d) for d, keep in zip(DAYS, split.val) if keep)
    runs = [1]
    for a, b in zip(hidden_days, hidden_days[1:]):
        runs[-1:] = [runs[-1] + 1] if (b - a).days == 7 else [runs[-1], 1]
    assert max(runs) == 3


def test_season_split_rejects_unknown_season():
    with pytest.raises(ValueError):
        season_split(DAYS, 'not-a-season')


def test_covariate_names_for_expands_groups_in_fixed_order():
    assert covariate_names_for('') == ()
    assert covariate_names_for('inpatient') == COVARIATE_GROUPS['inpatient']
    # Order follows COVARIATE_GROUPS, not the order named in the string.
    assert covariate_names_for('kinsa+inpatient') == COVARIATE_GROUPS['inpatient'] + COVARIATE_GROUPS['kinsa']


def test_covariate_names_for_rejects_unknown_group():
    with pytest.raises(ValueError):
        covariate_names_for('not_a_group')


def test_source_groups_cover_every_covariate_name_once():
    names = [name for group in SOURCE_GROUPS for name in COVARIATE_GROUPS[group]]
    assert len(names) == len(set(names))
