"""B0-informed B1 formulation screen; all candidates use the shared manager."""
from dataclasses import replace

from .b1_scenarios import B1Scenario, B0_TOP4_BASE


def scenarios(formulations_only=False, **budget):
    base = B1Scenario(**B0_TOP4_BASE, pipeline='direct_finalflag', mask_rate=.5)
    # B0.1 ranks 1, 4, 5, 6, retaining their original epoch caps.
    references = {
        'target_mlp': base,
        'pathogen_mlp': replace(base, fit_partition='pathogen'),
        'target_multiscale': replace(base, encoder='multiscale_conv'),
        'joint_mlp': replace(base, fit_partition='all', epochs=300,
                             spatial='joint_location_target', head_sharing='pathogen'),
    }
    candidates = {}
    for name, anchor in references.items():
        # Four formulations at the working masking rate, on every architecture.
        for pipeline in ('direct', 'direct_finalflag', 'joint_aux025', 'two_stage'):
            candidates[f'{name}__{pipeline}__mask0.5'] = replace(anchor, pipeline=pipeline)
        if formulations_only:
            continue
        # Rate and mechanism contrasts focus on the first successful B recipe.
        for rate in (0., .25):
            candidates[f'{name}__direct_finalflag__mask{rate:g}'] = replace(anchor, mask_rate=rate)
        for kind, mix in [('recent', (1., 0., 0.)), ('gap', (0., 1., 0.)),
                          ('outage', (0., 0., 1.))]:
            candidates[f'{name}__direct_finalflag__{kind}_only'] = replace(
                anchor, mask_recent=mix[0], mask_gap=mix[1], mask_outage=mix[2])
        # C helped most under outages: retain this matched B/C contrast.
        candidates[f'{name}__joint_aux025__outage_only'] = replace(
            anchor, pipeline='joint_aux025', mask_recent=0., mask_gap=0., mask_outage=1.)
    overrides = {key: value for key, value in budget.items() if value is not None}
    # Keep explicit budget overrides available through the ordinary manager.
    unique = {}
    for name, candidate in candidates.items():
        candidate = replace(candidate, **overrides)
        unique.setdefault(candidate, name)
    return {name: candidate for candidate, name in unique.items()}
