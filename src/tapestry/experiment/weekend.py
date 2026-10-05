"""B2-sized revision experiments using the common planner, fitter and scorer."""
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch

from tapestry.dataset import cv
from tapestry.dataset.build import load
from tapestry.dataset.reporting_error import ReportingErrors
from tapestry.model.network import checkpoint
from tapestry.model.revision_regression import RevisionRegression
from . import training
from .provenance import save, sha256, environment


class CorrectedErrors:
    """Fresh synthetic reports each epoch, corrected by season-cross-fitted models."""
    transport_missingness = False
    def __init__(self, bank, models):
        self.bank, self.models = bank, models
    def batch(self, batch, rng):
        return [self.models[cv.season(e['context_dates'][-1])].apply(self.bank.draw(e, rng)) for e in batch]


def fit_correctors(train, bank, scenario, seed):
    # The empirical error distribution belongs to the outer training fold. Each
    # trajectory season is excluded from its correction regressor's label fit.
    rng = np.random.default_rng(seed + 410000)
    examples = [(bank.draw(e, rng), e) for e in train for _ in range(2)]
    def build(ex):
        return RevisionRegression(scenario.nowcast_weeks, scenario.correction_penalty,
                                  scenario.correction_strength, scenario.correction_features).fit(ex)
    labels = sorted({cv.season(e['context_dates'][-1]) for e in train})
    by_season = {}
    for label in labels:
        permitted = []
        for x, t in examples:
            if cv.season(t['context_dates'][-1]) == label:
                continue
            label_mask = np.array([cv.season(d) != label for d in t['context_dates']])[:, None, None]
            permitted.append((x, dict(t, available=t['available'] & label_mask)))
        by_season[label] = build(permitted)
    final = build(examples)
    return by_season, final


def subset_labels(e, indices):
    out = dict(e)
    for name in ('target_values', 'target_available', 'Y'):
        out[name] = e[name][indices]
    out['target_dates'] = tuple(np.asarray(e['target_dates'])[indices])
    return out


class ForecastSlice(torch.nn.Module):
    def __init__(self, model, indices):
        super().__init__()
        self.model, self.indices = model, indices
        self.config = dict(model.config, horizons=[model.config['horizons'][i] for i in indices])
        self.scale = model.models[0].scale if hasattr(model, 'models') else model.scale
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)[:, :, self.indices]


def fit(scenario, seed, held_out, members, device, output, dataset):
    panel = load(dataset)
    full = cv.fold(panel, scenario, held_out)
    inner = cv.fold(panel, scenario, held_out, inner=True) if scenario.patience else None
    roles = cv.week_roles(panel['dates'], scenario, held_out)
    keep = np.isin(roles, ['fit', 'validation'])
    outer = ReportingErrors(panel, scenario, held_out, keep)
    selection = ReportingErrors(panel, scenario, held_out, roles == 'fit') if inner else None
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    correct = None
    correction_records = None
    if scenario.weekend_family == 'two_stage':
        cross, final = fit_correctors(full.train, outer, scenario, seed)
        augmentation = CorrectedErrors(outer, cross)
        selection_augmentation = None
        if inner:
            inner_cross, inner_final = fit_correctors(inner.train, selection, scenario, seed)
            selection_augmentation = CorrectedErrors(selection, inner_cross)
            raw = selection.batch(inner.validation, np.random.default_rng(seed + 800000))
            inner.validation = [inner_final.apply(e) for e in raw]
        correct = lambda eps: [final.apply(e) for e in eps]
        correction_records = dict(full=final.record(), cross_fits={k:v.record() for k,v in cross.items()},
            inner=inner_final.record() if inner else None,
            error_library='Shared fold-permitted donor errors; regression labels exclude the corrected trajectory season')
    else:
        augmentation, selection_augmentation = outer, selection
        if inner:
            inner.validation = selection.batch(inner.validation, np.random.default_rng(seed + 800000))
    pop = training.populations('data/metadata/locations.csv', full.train[0]['locations'])
    model, records = training.fit_models(inner.train if inner else full.train,
        inner.validation if inner else None, full.train, scenario, seed, device, pop,
        augmenter=augmentation, selection_augmenter=selection_augmentation)
    metadata = dict(scenario=scenario.scenario_string, run_id=scenario.run_id, config=asdict(scenario),
        seed=seed, held_out_season=held_out, protocol='b2_weekend_scheduled_reports_v1',
        fold=full.info, inner_fold=inner.info if inner else None, records=records,
        reporting_errors=outer.metadata, selection_errors=selection.metadata if selection else None,
        corrections=correction_records,
        labels='Finalized future values; joint models additionally learn finalized recent context values',
        evaluation='Standard reported inputs (cv.score_episodes) at each Hub deadline',
        loss=training.LOSS_DEFINITION, joint_weight=scenario.joint_weight,
        dataset=str(dataset), dataset_sha256=sha256(dataset), eval_members=members,
        channels=panel['target_names'].tolist(), locations=list(full.train[0]['locations']), **environment())
    torch.save(checkpoint(model, metadata), out / 'model.pt')
    save(out / 'manifest.json', metadata)
    model.to(device)
    if scenario.weekend_family == 'joint':
        horizons = np.asarray(scenario.horizons)
        future = np.flatnonzero(horizons > 0).tolist()
        training.evaluate_hubs(ForecastSlice(model, future), panel, scenario, held_out, seed, members, device, out,
                               prepare=lambda eps: [subset_labels(e, future) for e in eps], first=full.score)
        indices = np.flatnonzero(horizons <= 0).tolist()
        recent = out / 'nowcast'; recent.mkdir(exist_ok=True)
        torch.manual_seed(seed + 1000)
        training.evaluate(ForecastSlice(model, indices), [subset_labels(e, indices) for e in full.score], members, device, recent, panel=panel)
    else:
        training.evaluate_hubs(model, panel, scenario, held_out, seed, members, device, out, prepare=correct, first=full.score)
    return out / 'model.pt'
