"""Seed-paired effects on the canonical combined score; no independent context replicates."""
from dataclasses import asdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

PAIR_COLUMNS = ['factor', 'reference', 'value', 'reference_config', 'config_id', 'seed', 'delta']
EFFECT_COLUMNS = ['factor', 'reference', 'value', 'contexts', 'seeds', 'mean_delta',
                  'seed_ci_low', 'seed_ci_high', 'improved_contexts', 'improved_seeds']


def matched_effects(scores):
    """Compare every single-field intervention, keeping all other Scenario fields fixed.

    Values are ordered numerically when numeric and lexically otherwise. Positive
    deltas mean the second value is worse. Only identical complete seed sets are
    paired; summaries with different seed sets remain separate. Scenario defaults
    are expanded before matching so omitted and explicit defaults are equivalent.
    """
    from tapestry.model.scenario import Scenario

    scores = scores[scores.geography.eq('all')]
    if scores.duplicated(['config_id', 'seed']).any() or not np.isfinite(scores.combined).all():
        raise ValueError('Matched effects require unique, finite configuration/seed scores')
    settings = {c: asdict(Scenario.from_string(c)) for c in scores.config_id.unique()}
    values = {c: g.set_index('seed').combined.sort_index() for c, g in scores.groupby('config_id')}
    rows = []
    for factor in asdict(Scenario()):
        # Stage overrides are mappings of multiple fields, not single interventions.
        if factor in ('nowcast', 'forecast'):
            continue
        groups = {}
        for config, fields in settings.items():
            key = json.dumps({k: v for k, v in fields.items() if k != factor}, sort_keys=True)
            groups.setdefault(key, []).append(config)
        for configs in groups.values():
            configs.sort(key=lambda c: settings[c][factor])
            for ref, config in combinations(configs, 2):
                if settings[ref][factor] == settings[config][factor]:
                    continue
                if not values[ref].index.equals(values[config].index):
                    continue
                for seed, delta in (values[config] - values[ref]).items():
                    rows.append(dict(factor=factor, reference=str(settings[ref][factor]),
                                     value=str(settings[config][factor]), reference_config=ref,
                                     config_id=config, seed=int(seed), delta=float(delta)))
    pairs = pd.DataFrame(rows, columns=PAIR_COLUMNS)
    effects = []
    # Keep contexts with different available seeds out of the same seed average.
    for (factor, ref, value), group in pairs.groupby(['factor', 'reference', 'value']):
        contexts = {}
        for _, context in group.groupby(['reference_config', 'config_id']):
            seed_set = tuple(sorted(context.seed))
            contexts.setdefault(seed_set, []).append(context)
        for seed_set, parts in contexts.items():
            matched = pd.concat(parts)
            byseed = matched.groupby('seed').delta.mean()
            bycontext = matched.groupby(['reference_config', 'config_id']).delta.mean()
            mean = byseed.mean()
            half = t.ppf(.975, len(byseed) - 1) * byseed.std() / np.sqrt(len(byseed)) if len(byseed) > 1 else np.nan
            effects.append(dict(factor=factor, reference=ref, value=value, contexts=len(parts),
                                seeds=','.join(map(str, seed_set)), mean_delta=mean,
                                seed_ci_low=mean-half, seed_ci_high=mean+half,
                                improved_contexts=int((bycontext < 0).sum()),
                                improved_seeds=int((byseed < 0).sum())))
    return pairs, pd.DataFrame(effects, columns=EFFECT_COLUMNS)


def write_effects(ranking):
    """Analyze saved scores without loading models, retraining, or redefining scoring."""
    ranking = Path(ranking)
    scores = pd.read_csv(ranking / 'run_scores.csv', keep_default_na=False)
    pairs, effects = matched_effects(scores)
    pairs.to_csv(ranking / 'matched_pairs.csv', index=False)
    effects.to_csv(ranking / 'matched_effects.csv', index=False)
    return effects
