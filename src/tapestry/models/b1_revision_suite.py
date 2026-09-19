"""B1 revision experiment: matched objectives, validation and augmentation."""
from dataclasses import replace
from .b1_scenarios import B1Scenario, B0_TOP4_BASE


def scenarios(**budget):
    base = B1Scenario(**B0_TOP4_BASE, pipeline='direct_finalflag', epochs=300,
                      validation_mode='natural_forecast', mask_rate=.5)
    candidates = {}
    for architecture in ('pathogen', 'target'):
        anchor = replace(base, fit_partition=architecture)
        formulations = {'B': anchor}
        for pipeline, label in [('joint_aux', 'C'), ('two_stage', 'two'), ('gated_revision', 'gated')]:
            for weight in (.1, .2):
                formulations[f'{label}_nw{weight:g}'] = replace(anchor, pipeline=pipeline, nowcast_weight=weight)
        for label, candidate in formulations.items():
            for rate in (0., .5):
                candidates[f'{architecture}_{label}_rev{rate:g}'] = replace(candidate, revision_rate=rate)
        candidates[f'{architecture}_B_gap_only'] = replace(anchor, mask_recent=0., mask_gap=1., mask_outage=0.)
        candidates[f'{architecture}_B_no_mask'] = replace(anchor, mask_rate=0.)
    overrides = {key: value for key, value in budget.items() if value is not None}
    return {name: replace(candidate, **overrides) for name, candidate in candidates.items()}
