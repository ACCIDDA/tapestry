"""The training route: fit one leave-one-season-out fold, then evaluate it on standard inputs.

The training histories come from final values, actual archived reports, or final values
with artificial reporting errors (`history_source`; `dataset/reporting_error.py`); artificial
histories can be corrected by the cross-fitted correction model before training
(`history_correction`; `model/revision_tree.py`), and recent-week reconstruction labels can
be added (`reconstruction_labels`). See docs/reference/model-choices.md. Every fold saves
`model.pt`, the correction model `nowcaster.pkl` and forecasts for each input view: `raw` (the
evaluation inputs as they are), `corrected` (newest weeks corrected), `half` (50/50 predictive
mixture of raw and corrected), and optional `calibrated`, `sampled`, `delayed`, `nokinsa`.
The recipe's own view (`forecast_view`) is the one ranked and deployed; the others are
diagnostics.

A problem evaluated on finalized histories (`Problem.reporting_errors` false, e.g. a dataset
without archived report vintages) skips the whole reporting-error stage: no artificial errors,
no correction model (`nowcaster.pkl` holds None), no calibration, and only the `raw` view.

History: this was `experiment/pilot.py` (the B3 pilot). On 2026-10-08 it became the only
route; the plain, weekend, replay, nowcast, pipeline and finalization routes were deleted.
`ForecastSlice` and `subset_labels` came from the deleted `weekend.py`.
"""
from dataclasses import asdict, replace
from pathlib import Path
import pickle
import numpy as np
import torch
from chromantis.dataset import cv
from chromantis.dataset.build import load
from chromantis.dataset.episodes import episodes
from chromantis.dataset.reporting_error import ReportingErrors
from chromantis.model.revision_tree import TrajectoryNowcaster
from chromantis.model.network import checkpoint, load_model
from chromantis.model.scenario import Scenario
from chromantis.problem import Problem
from . import training
from .provenance import save, sha256, environment

LOCATIONS = 'data/metadata/locations.csv'  # populations; pinned by the planner


def subset_labels(e, indices):
    out = dict(e)
    for name in ('target_values', 'target_available', 'Y'):
        out[name] = e[name][indices]
    out['target_dates'] = tuple(np.asarray(e['target_dates'])[indices])
    return out


class ForecastSlice(torch.nn.Module):
    """The forecast weeks of a `joint` model, whose outputs also include reconstruction weeks."""
    def __init__(self, model, indices):
        super().__init__()
        self.model, self.indices = model, indices
        self.config = dict(model.config, horizons=[model.config['horizons'][i] for i in indices])
        self.scale = model.models[0].scale if hasattr(model, 'models') else model.scale
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)[:, :, self.indices]


def real_pairs(panel,problem,scenario,held_out,keep):
    """Genuine reported/mature pairs only; no finalized fills supervise revisions."""
    p=cv.masked(panel,keep)
    names=problem.covariate_names(scenario.covariate_set)
    reported=episodes(p,problem,problem.input_names(scenario.input_set),scenario.lookback,'reported',names)
    dates=panel['dates'].astype(str);issues=panel['issuance_dates'].astype(str)
    index={d:i for i,d in enumerate(dates)}
    allowed=set(dates[keep]);donor=max(cv.season(d) for d in allowed)
    # error_seasons=all (2026-10-06): genuine pairs from every permitted season, not only the latest.
    donors={cv.season(d) for d in allowed} if scenario.error_seasons=='all' else {donor}
    # In the forward fold all labels must be published before the holdout starts;
    # the reverse fold is explicitly retrospective and uses the latest training end.
    cutoff=min(str(np.datetime64(max(allowed))+np.timedelta64(1,'D')),str(panel.get('truth_day','9999-01-01')))
    pairs=[]
    for e in reported:
        if e['context_dates'][-1] not in allowed or cv.season(e['context_dates'][-1]) not in donors:continue
        v=np.zeros_like(e['values']);ok=np.zeros_like(e['available'])
        for a,d in enumerate(e['context_dates']):
            if d not in allowed or cv.season(d) not in donors:continue
            mature=str(np.datetime64(d)+np.timedelta64(12*7+4,'D'))
            w=np.searchsorted(issues,mature)
            if w>=len(issues) or issues[w]>=cutoff:continue
            panel_names=list(map(str,p['target_names']))
            indices=[panel_names.index(name) for name in e['input_names']]
            y=p['asof_targets'][w,index[d]][...,indices].T
            v[a]=np.nan_to_num(y)
            ok[a]=np.isfinite(y)&e['available'][a]&~e['filled'][a]
        pairs.append((e,dict(e,values=v,available=ok)))
    if not pairs:raise ValueError('No genuine mature report pairs before cutoff')
    return pairs


def fit_corrector(train,bank,problem,scenario,seed,pairs=None,excluded=None,residuals=False):
    rng=np.random.default_rng(seed+410000)
    synthetic=[(bank.draw(e,rng),e) for e in train]
    examples=synthetic if scenario.corrector_examples=='synthetic' else pairs
    if excluded:
        def purge(examples):
            return [(x,dict(y,available=y['available'] & np.array([d not in excluded for d in y['context_dates']])[:,None,None])) for x,y in examples]
        examples=purge(examples);synthetic=purge(synthetic)
    model=TrajectoryNowcaster(min(scenario.correction_weeks,scenario.lookback),scenario.correction_penalty,scenario.correction_strength,'phase')
    model.input_channels=list(range(len(train[0]['input_names'])))
    model.channels=[c for c,unit in zip(train[0]['target_input_indices'],train[0]['target_units'])
                    if unit=='count' or scenario.correction_ed]
    model.fit(examples,pretraining=synthetic if scenario.corrector_examples=='synthetic_then_real' else None,
              neural=scenario.corrector_model=='mlp',seed=seed)
    model.observed_donor_pairs_only=scenario.corrector_examples!='synthetic'
    if residuals:
        model.fit_residuals(examples,seed)
    return model


HISTORY_KEYS=('values','available','known_final','covariates','filled')


class CorrectedHistories:
    """Per-minibatch choice among stored correction realizations, optionally uncorrected or noisy."""
    transport_missingness=False
    def __init__(self,scenario,corrector):
        self.scenario,self.corrector=scenario,corrector
    def batch(self,episodes_,rng):
        out=[]
        for e in episodes_:
            corrected=True
            if 'realizations' in e:
                fixed,raw=e['realizations'][rng.integers(len(e['realizations']))]
                corrected=not(self.scenario.uncorrected_share and rng.random()<self.scenario.uncorrected_share)
                e=dict(e,**(fixed if corrected else raw))
            if corrected and self.scenario.correction_noise_train:
                e=self.corrector.perturb(e,rng,self.scenario.correction_noise)
            out.append(e)
        return out


def cross_correct(train,bank,problem,scenario,seed,pairs):
    # Four disjoint origin blocks; exclude their entire context-date union from
    # correction labels, even when those labels occur in other episodes.
    order=sorted({e['context_dates'][-1] for e in train})
    groups={d:(i//8)%4 for i,d in enumerate(order)}
    result=[]
    for g in range(4):
        selected=[e for e in train if groups[e['context_dates'][-1]]==g]
        excluded={d for e in selected for d in e['context_dates']}
        model=fit_corrector(train,bank,problem,scenario,seed,pairs,excluded)
        draws=[]
        for r in range(scenario.correction_realizations):
            # r=0 keeps the original B3/B4 random stream exactly.
            raw=bank.batch(selected,np.random.default_rng(seed+g+8700+100003*r))
            draws.append((raw,model.apply_batch(raw)))
        for i,e in enumerate(draws[0][1]):
            if scenario.correction_realizations>1 or scenario.uncorrected_share:
                pick=lambda x:{k:x[k] for k in HISTORY_KEYS if k in x}
                e=dict(e,realizations=[(pick(c[i]),pick(raw[i])) for raw,c in draws])
            result.append(e)
    return result


def evaluation_bank(panel,problem,scenario,held_out,keep):
    """Prescribed artificial evaluation histories (`evaluation_inputs=prescribed`), or None for actual reports.

    Identical empirical input process for every architecture, training treatment and seed."""
    if problem.evaluation_inputs!='prescribed':return None
    evaluation_scenario=replace(scenario,lookback=max(12,scenario.lookback),reporting_method='synchronous_phase_log',reporting_strength=1.,
        reporting_probability=1.,reporting_random_strength=False,reporting_recent=0,reporting_missingness=False,
        actual_share=0.,error_signals='targets')
    return ReportingErrors(panel,problem,evaluation_scenario,held_out,keep)


def forecast_labels(problem,scenario,eps):
    """Forecast-week labels only (a `joint` model also outputs reconstruction weeks)."""
    return [subset_labels(e,problem.future_indices(scenario)) for e in eps] if scenario.reconstruction_labels else eps


def vintage(eps,bank,scenario,draw):
    """Artificially preliminary copies of `eps` (draw `draw` of the prescribed process), cut to the lookback."""
    if bank is None:return eps
    result=[]
    for e in eps:
        # Calendar-derived seeds make the input draw independent of fitted-model seed.
        x=bank.draw(e,np.random.default_rng(20261007 + 10000019 * draw +
            int(np.datetime64(e['context_dates'][-1],'D').astype(int))))
        if len(x['context_dates'])>scenario.lookback:
            x=dict(x,context_dates=x['context_dates'][-scenario.lookback:],
                **{k:x[k][-scenario.lookback:] for k in HISTORY_KEYS if k in x})
        result.append(x)
    return result


def view_names(scenario,problem):
    if not problem.reporting_errors:return ['raw']
    names=['raw','half','corrected']
    if scenario.correction_noise:names.append('sampled')
    if scenario.stress_views:
        names.append('delayed')
        if scenario.covariate_set!='none':names.append('nokinsa')
    return names


def view_inputs(name,eps,corrector,problem,scenario,seed,members,direct):
    """Episodes for one input view; correctors act on episodes already cut at each Hub's own deadline."""
    if name=='raw':return eps
    if name=='delayed':
        # Stress view: the newest admission week of the (first) forecast pathogen was not reported by the deadline.
        k=problem.target_input_indices(scenario.input_set)[0]
        def late(e):
            available=e['available'].copy();available[-1,k]=False
            late_e=dict(e,available=available,values=np.where(available,e['values'],0).astype(np.float32))
            if 'filled' in e:
                filled=e['filled'].copy();filled[-1,k]=False;late_e['filled']=filled
            return late_e
        return corrector.apply_batch([late(e) for e in eps])
    changed=corrector.apply_batch(eps)
    if name=='corrected':return changed
    if name=='nokinsa':
        # Stress view: every covariate (Kinsa) feed missing at evaluation.
        return [dict(e,covariates=np.zeros_like(e['covariates'])) if 'covariates' in e else e for e in changed]
    if name=='sampled':
        # Plausible corrected histories: point correction plus sampled out-of-fold errors.
        rng=np.random.default_rng(seed+5150);k=16 if direct else members
        return [dict(c,history_samples=np.stack([corrector.perturb(c,rng,scenario.correction_noise)['values'] for _ in range(k)])) for c in changed]
    return [dict(e,mixture_episode=c) for e,c in zip(eps,changed)]  # half


def evaluate_views(model,corrector,problem,scenario,panel,held_out,seed,members,device,out,bank=None,first=None,resume=False):
    """Held-out forecasts of one fitted fold for every input view (and artificial draw), per Hub deadline."""
    model.to(device)
    if scenario.reconstruction_labels:
        model=ForecastSlice(model,problem.future_indices(scenario))
    direct=bool(model.config.get('direct_quantiles'))
    for draw,name in ((d,n) for d in range(problem.evaluation_draws if bank else 1) for n in view_names(scenario,problem)):
        root=out if draw==0 else out/f'draw-{draw}'
        destination=root if name=='raw' else root/name
        destination.mkdir(parents=True,exist_ok=True)
        if resume and (destination/'forecasts.npz').exists():
            try:
                with np.load(destination/'forecasts.npz') as prior:
                    for key in prior.files:prior[key]  # Reject an interrupted archive write.
                continue
            except (OSError, ValueError, EOFError):
                pass
        prepare=lambda eps,name=name,draw=draw:view_inputs(name,vintage(forecast_labels(problem,scenario,eps),bank,scenario,draw),
                                                          corrector,problem,scenario,seed,members,direct)
        training.evaluate_hubs(model,panel,problem,scenario,held_out,seed,members,device,destination,prepare=prepare,first=first)


def fit(problem,scenario,seed,held_out,members,device,output,dataset,resume=False):
    panel=load(dataset);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    problem.validate_scenario(scenario);problem.validate_panel(panel)
    full=cv.fold(panel,problem,scenario,held_out)
    inner=cv.fold(panel,problem,scenario,held_out,inner=True) if scenario.patience else None
    roles=cv.week_roles(panel['dates'],problem,scenario,held_out)
    keep=np.isin(roles,['fit','validation'])
    restored=resume and (out/'model.pt').exists() and (out/'nowcaster.pkl').exists() and (out/'manifest.json').exists()
    if resume and problem.evaluation_inputs!='prescribed':
        raise ValueError('Checkpoint resume currently requires prescribed evaluation vintages')
    if restored:
        saved=torch.load(out/'model.pt',map_location='cpu',weights_only=False)
        meta=saved['metadata']
        if (meta['seed'],meta['held_out_season'],meta['eval_members'],meta['dataset_sha256']) != (seed,held_out,members,sha256(dataset)) \
                or Scenario.from_string(meta['scenario']) != scenario or meta['problem_sha256'] != problem.hash:
            raise ValueError('Saved model does not match requested fold, inputs or evaluation members')
        if (out/'nowcast-diagnostics.csv').exists():
            return out/'model.pt'
        model=load_model(saved);selector=None
        with (out/'nowcaster.pkl').open('rb') as f:corrector=pickle.load(f)
    else:
        # Reported values are not finalized observations for native loss scales.
        if scenario.history_source=='reported':
            for e in full.train+(inner.train if inner else []):e['known_final']=e['filled'].copy()
        bank=selection=pairs=corrector=None
        if problem.reporting_errors:
            bank=ReportingErrors(panel,problem,scenario,held_out,keep)
            selection=ReportingErrors(panel,problem,scenario,held_out,roles=='fit') if inner else None
            pairs=real_pairs(panel,problem,scenario,held_out,keep) if scenario.corrector_examples!='synthetic' else None
            corrector_train=cv.fold(panel,problem,scenario,held_out,inputs='scheduled_final').train
            corrector=fit_corrector(corrector_train,bank,problem,scenario,seed,pairs,residuals=scenario.correction_noise>0)
        augmentation=selection_augmentation=None
        validation_corrector=inner_pairs=None
        if scenario.history_source=='artificial' and not scenario.history_correction:
            augmentation,selection_augmentation=bank,selection
            if inner:inner.validation=selection.batch(inner.validation,np.random.default_rng(seed+800000))
        if scenario.history_correction:
            if inner:
                inner_pairs=real_pairs(panel,problem,scenario,held_out,roles=='fit') if pairs is not None else None
                validation_corrector=fit_corrector(inner.train,selection,problem,scenario,seed,inner_pairs,residuals=scenario.correction_noise_train)
                inner.train=cross_correct(inner.train,selection,problem,scenario,seed,inner_pairs)
                inner.validation=validation_corrector.apply_batch(selection.batch(inner.validation,np.random.default_rng(seed+800000)))
            full.train=cross_correct(full.train,bank,problem,scenario,seed,pairs)
            if scenario.correction_realizations>1 or scenario.uncorrected_share or scenario.correction_noise_train:
                augmentation=CorrectedHistories(scenario,corrector)
                selection_augmentation=CorrectedHistories(scenario,validation_corrector) if inner else None
        pop=training.populations(LOCATIONS,full.train[0]['locations'])
        model,records,selector=training.fit_models(inner.train if inner else full.train,inner.validation if inner else None,
            full.train,problem,scenario,seed,device,pop,augmenter=augmentation,selection_augmenter=selection_augmentation,keep_selection=True)
    prescribed=evaluation_bank(panel,problem,scenario,held_out,keep)
    if not restored:
        meta=dict(scenario=scenario.scenario_string,config=asdict(scenario),problem_id=problem.id,
            problem=problem.reference,problem_sha256=problem.hash,dataset_spec=problem.dataset.reference,
            seed=seed,held_out_season=held_out,
            fold=full.info,inner_fold=inner.info if inner else None,records=records,eval_members=members,
            dataset=str(dataset),dataset_sha256=sha256(dataset),ili_sha256=sha256(scenario.ili_path) if scenario.ili_training!='none' else None,
            reporting_errors=bank.metadata if bank else None,corrector=corrector.record() if corrector else None,
            corrector_kind=f'{scenario.corrector_examples}_{scenario.corrector_model}' if corrector else None,
            evaluation_vintages=prescribed.metadata if prescribed else None,
            parameter_count=sum(p.numel() for p in model.parameters()),
            protocol='prescribed_2025_revision_process' if prescribed else 'b3_pilot_standard_hub_inputs',**environment())
        torch.save(checkpoint(model,meta),out/'model.pt')
        with (out/'nowcaster.pkl').open('wb') as f:pickle.dump(corrector,f)
        save(out/'manifest.json',meta)
    evaluate_views(model,corrector,problem,scenario,panel,held_out,seed,members,device,out,bank=prescribed,first=full.score,resume=resume)
    if not problem.reporting_errors:
        return out/'model.pt'
    if selector is not None and problem.evaluation_inputs!='prescribed':
        if validation_corrector is None:
            if inner_pairs is None and pairs is not None:
                inner_pairs=real_pairs(panel,problem,scenario,held_out,roles=='fit')
            inner_final=cv.fold(panel,problem,scenario,held_out,inner=True,inputs='scheduled_final').train
            validation_corrector=fit_corrector(inner_final,selection,problem,scenario,seed,inner_pairs)
        calibrate(selector,validation_corrector,inner,panel,problem,scenario,held_out,keep,seed,members,device,out)
    if prescribed is not None:
        # Same artificially preliminary inputs as the forecast evaluation; later histories are labels.
        import pandas as pd
        reference=forecast_labels(problem,scenario,full.score);raw=vintage(reference,prescribed,scenario,0);changed=corrector.apply_batch(raw)
        diagnostics=[]
        for target,k,unit in zip(problem.targets,problem.target_input_indices(scenario.input_set),problem.target_units):
            for age in range(scenario.lookback):
                for group in ('states_dc','US'):
                    before=[];after=[]
                    for t,r,c in zip(reference,raw,changed):
                        ids=np.array([loc=='US' if group=='US' else loc!='US' for loc in t['locations']])
                        ok=t['available'][-1-age,k]&ids
                        before.extend(abs(r['values'][-1-age,k]-t['values'][-1-age,k])[ok])
                        after.extend(abs(c['values'][-1-age,k]-t['values'][-1-age,k])[ok])
                    diagnostics.append(dict(target=target,unit=unit,age=age,geography=group,
                        cells=len(before),raw_mae=float(np.mean(before)),corrected_mae=float(np.mean(after))))
        pd.DataFrame(diagnostics).to_csv(out/'nowcast-diagnostics.csv',index=False)
        return out/'model.pt'
    # Direct history accuracy on the same genuine observed cells, independent of forecast score.
    evaluation_keep=problem.fold_labels(panel['dates'])==held_out
    # Maturity labels may be observed after forecast issuance; fitting above never uses them.
    try:
        eval_pairs=real_pairs(panel,problem,scenario,held_out,evaluation_keep)
    except ValueError:
        # Production (2026-27): no matured report pairs exist yet, so no direct nowcast diagnostics.
        save(out/'nowcast-diagnostics.json',dict(unavailable='no mature report pairs in the forecast season'))
        return out/'model.pt'
    raw=[e for e,_ in eval_pairs];changed=corrector.apply_batch(raw)
    diagnostics={}
    count_inputs=[i for i,unit in enumerate(problem.input_names(scenario.input_set))
                  if problem.dataset.by_name[unit].unit=='count']
    for k in count_inputs:
        metrics={name:[] for name in ('newest_week_mae','recent_four_week_level_mae','two_week_log_growth_mae')}
        for (e,t),c in zip(eval_pairs,changed):
            actual=t['values'][:,k];raw_value=e['values'][:,k];corrected_value=c['values'][:,k]
            newest=t['available'][-1,k]
            metrics['newest_week_mae'].extend(zip(abs(raw_value[-1]-actual[-1])[newest],abs(corrected_value[-1]-actual[-1])[newest]))
            ok=t['available'][-4:,k].all(0)
            functions={'recent_four_week_level_mae':lambda v:v[-4:].mean(0),
                       'two_week_log_growth_mae':lambda v:np.log1p(v[-2:].mean(0))-np.log1p(v[-4:-2].mean(0))}
            for name,func in functions.items():
                metrics[name].extend(zip(abs(func(raw_value)-func(actual))[ok],abs(func(corrected_value)-func(actual))[ok]))
        diagnostics[str(panel['target_names'][k])]={}
        for name,rows in metrics.items():
            a=np.asarray(rows)
            diagnostics[str(panel['target_names'][k])][name]=dict(cells=len(rows),raw=float(a[:,0].mean()),corrected=float(a[:,1].mean())) if len(rows) else dict(cells=0)
    save(out/'nowcast-diagnostics.json',diagnostics)
    return out/'model.pt'


CALIBRATION_SEASONS=('2023-2024','2024-2025','2025-2026')  # 2022-23 has no archived reports


def calibrate(selector,corrector,inner,panel,problem,scenario,held_out,keep,seed,members,device,out):
    """Fit spread factors on the inner validation weeks, apply them to the corrected view.

    The early-stopped inner model never trained on the validation weeks' labels, and the
    inner corrector excluded them. Calibration forecasts use the evaluation pipeline:
    FluSight-deadline Wednesday reports, cut from the fold's training-season panel only,
    with the newest admission weeks corrected. Origins in 2022-23 are skipped because no
    reports were archived then, so their inputs would all be finalized fills."""
    import json
    from chromantis.dataset.build import for_hub, HUBS
    from chromantis.dataset.episodes import restrict_labels
    from chromantis.evaluation import calibration
    dates=panel['dates'].astype(str);index={d:i for i,d in enumerate(dates)}
    hidden=set(inner.info['validation_weeks'])
    horizons=problem.model_horizons(scenario)
    origins={dates[index[d]-h] for d in hidden for h in horizons if h>0 and index[d]-h>=0}
    origins={d for d in origins if cv.season(d) in CALIBRATION_SEASONS and keep[index[d]]}
    hub=next((value for value in problem.target_hubs if value),HUBS[0])
    eps=episodes(for_hub(cv.masked(panel,keep),hub),problem,problem.input_names(scenario.input_set),
                 scenario.lookback,'reported',problem.covariate_names(scenario.covariate_set),horizons=horizons)
    eps=[e for e in eps if e['context_dates'][-1] in origins]
    eps=[e for e in (restrict_labels(e,hidden) for e in eps) if e is not None]
    eps=corrector.apply_batch(forecast_labels(problem,scenario,eps))
    model=ForecastSlice(selector,problem.future_indices(scenario)) if scenario.reconstruction_labels else selector
    model.to(device)
    folder=out/'calibration';folder.mkdir(exist_ok=True)
    torch.manual_seed(seed+1000)
    training.evaluate(model,eps,members,device,folder)
    with np.load(folder/'forecasts.npz') as f:
        factors,record=calibration.fit(f['quantiles'],f['truth'],f['mask'],f['locations'],problem.target_units,
            [c for c,w in enumerate(problem.target_weights(scenario.loss_weights)) if w])
    with np.load(out/'corrected'/'forecasts.npz') as f:
        data={k:f[k] for k in f.files if k!='target_0_sum_quantiles'}
    data['quantiles']=calibration.apply(data['quantiles'],factors,problem.target_units)
    (out/'calibrated').mkdir(exist_ok=True)
    np.savez_compressed(out/'calibrated'/'forecasts.npz',**data)
    save(folder/'calibration.json',dict(definition=calibration.__doc__,factors={str(k):v for k,v in factors.items()},
        fits={str(k):v for k,v in record.items()},origins=sorted({e['context_dates'][-1] for e in eps}),
        seasons=sorted({cv.season(e['context_dates'][-1]) for e in eps}),applied_to='corrected'))


def replay(fold, output, inputs, dataset, device='cpu'):
    """Evaluate one saved fold (`model.pt`, `nowcaster.pkl`) on other evaluation inputs, without refitting.

    `inputs='reported'`: the archived Wednesday reports at each Hub deadline (finalized
    values only where nothing was archived, as in the standard evaluation);
    `'synthetic'`: the fold's prescribed artificial histories. The dataset must be the
    panel the fold was fitted on. Writes the same view folders as `fit`, plus a manifest
    stating that no model was refitted."""
    import json
    fold, out = Path(fold), Path(output)
    meta = json.loads((fold / 'manifest.json').read_text())
    if sha256(dataset) != meta['dataset_sha256']:
        raise ValueError(f'{dataset} is not the panel {fold} was fitted on')
    scenario = Scenario.from_string(meta['scenario'])
    problem = Problem.load(meta['problem'])
    if inputs == 'reported':
        if problem.evaluation_inputs != 'reported':
            raise ValueError('This problem does not define reported evaluation inputs')
    elif inputs != 'synthetic' or problem.evaluation_inputs != 'prescribed':
        raise ValueError('Replay inputs are reported, or synthetic for a prescribed-revision fit')
    panel, held_out = load(dataset), meta['held_out_season']
    keep = np.isin(cv.week_roles(panel['dates'], problem, scenario, held_out), ['fit', 'validation'])
    model = load_model(torch.load(fold / 'model.pt', map_location='cpu', weights_only=False))
    with (fold / 'nowcaster.pkl').open('rb') as f:
        corrector = pickle.load(f)
    out.mkdir(parents=True, exist_ok=True)
    evaluate_views(model, corrector, problem, scenario, panel, held_out, meta['seed'], meta['eval_members'], device, out,
                   bank=evaluation_bank(panel, problem, scenario, held_out, keep))
    save(out / 'manifest.json', dict(replay_of=str(fold.resolve()), inputs=inputs, refit=False,
                                     scenario=meta['scenario'], problem=meta['problem'], problem_id=problem.id,
                                     problem_sha256=problem.hash, seed=meta['seed'], held_out_season=held_out,
                                     eval_members=meta['eval_members'], dataset_sha256=meta['dataset_sha256']))
