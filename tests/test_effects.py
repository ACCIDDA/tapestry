"""Matched effects must not confound settings or count contexts as seed replicates."""
import numpy as np
import pandas as pd
from scipy.stats import t

from tapestry.evaluation.effects import matched_effects
from tapestry.model.scenario import Scenario


def test_effects_pair_settings_and_compute_uncertainty_across_seeds():
    rows = []
    for width, deltas in [(16, [.1, .3]), (32, [.3, .5])]:
        for mask in (0., .2):
            for seed, delta in zip((42, 43), deltas):
                rows.append(dict(config_id=Scenario(width=width, mask_rate=mask).scenario_string,
                                 seed=seed, geography='all', combined=1 + (delta if mask else 0)))
    pairs, effects = matched_effects(pd.DataFrame(rows))
    effect = effects[effects.factor.eq('mask_rate')].iloc[0]
    assert effect.contexts == 2
    assert effect.seeds == '42,43'
    assert np.isclose(effect.mean_delta, .3)
    assert np.isclose(effect.seed_ci_high, .3 + t.ppf(.975, 1) * .1)
    for row in pairs[pairs.factor.eq('mask_rate')].itertuples():
        assert Scenario.from_string(row.reference_config).width == Scenario.from_string(row.config_id).width


def test_effects_do_not_impute_unmatched_seeds():
    rows = [dict(config_id=Scenario(mask_rate=mask).scenario_string, seed=seed,
                 geography='all', combined=1.) for mask, seed in [(0., 42), (.2, 42), (.2, 43)]]
    pairs, effects = matched_effects(pd.DataFrame(rows))
    assert pairs.empty
    assert effects.empty
