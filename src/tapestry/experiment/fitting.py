"""Fit one experiment fold: standalone model or independently fitted two-stage pipeline."""
from dataclasses import asdict
from pathlib import Path

import torch
from tapestry.dataset import cv
from tapestry.dataset.build import load as load_dataset, PANEL_DATASET
from tapestry.model.network import checkpoint
from tapestry.model.objective import LOSS_DEFINITION, US_WEIGHT
from .training import populations, fit_models, evaluate, GROUPS
from .provenance import save, environment, sha256
from .two_stage import fit_pipeline

LOCATIONS = 'data/metadata/locations.csv'


def fit(scenario, seed, held_out_season, eval_members, device, output, dataset=PANEL_DATASET):
    """One leave-one-season-out fold (`dataset.cv`): select epochs, refit, evaluate the held-out season."""
    if scenario.task == 'pipeline':
        return fit_pipeline(scenario, seed, held_out_season, eval_members, device, output, dataset, LOCATIONS)
    panel = load_dataset(dataset)
    full = cv.fold(panel, scenario, held_out_season)
    inner = cv.fold(panel, scenario, held_out_season, inner=True) if scenario.patience else None
    pop = populations(LOCATIONS, full.train[0]['locations'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    groups = GROUPS[scenario.fit_partition]
    model, records = fit_models(inner.train if inner else full.train, inner.validation if inner else None,
                               full.train, scenario, seed, device, pop)
    metadata = dict(scenario=scenario.scenario_string, run_id=scenario.run_id, config=asdict(scenario),
                    seed=seed, held_out_season=held_out_season, groups=groups,
                    parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    protocol=('complete_history_deadline_refit_v1' if scenario.training_inputs == 'finalized'
                              else 'masked_panel_season_cv_refit_v2'), evaluation_seed=seed + 1000, fold=full.info,
                    inner_fold=inner.info if inner else None, records=records,
                    loss=LOSS_DEFINITION, us_weight=US_WEIGHT, channels=[str(c) for c in panel['target_names']],
                    dataset=str(dataset), dataset_sha256=sha256(dataset),
                    locations=list(full.train[0]['locations']), eval_members=eval_members, **environment())
    torch.save(checkpoint(model, metadata), output / 'model.pt')
    save(output / 'manifest.json', metadata)
    model.to(device)
    torch.manual_seed(seed + 1000)
    scores = evaluate(model, full.score, eval_members, device, output)
    metadata['evaluation_scores'] = len(scores)
    save(output / 'manifest.json', metadata)
    return output / 'model.pt'

