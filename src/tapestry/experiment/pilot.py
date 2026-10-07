"""Exploratory reporting/transfer fits through the common manager and Hub evaluator."""
from dataclasses import asdict, replace
from pathlib import Path
import pickle
import numpy as np
import torch
from tapestry.dataset import cv
from tapestry.dataset.build import load, covariate_names_for
from tapestry.dataset.episodes import episodes
from tapestry.dataset.reporting_error import ReportingErrors
from tapestry.model.revision_tree import TrajectoryNowcaster
from tapestry.model.network import checkpoint
from . import training
from .provenance import save, sha256, environment
from .weekend import ForecastSlice, subset_labels


class ScopedErrors(ReportingErrors):
    def __init__(self,panel,scenario,held_out,keep):
        super().__init__(panel,scenario,held_out,keep)
        self.scenario=scenario
        self.reported={e['context_dates'][-1]:e for e in episodes(cv.masked(panel,keep),scenario.lookback,'reported',self.names,horizons=tuple(range(1-scenario.lookback,5)))}
        self.metadata.update(signals=scenario.revision_signals,scope=scenario.revision_scope)
    def draw(self,e,rng):
        day=e['context_dates'][-1]
        if self.scenario.actual_share and day in self.reported and rng.random()<self.scenario.actual_share:
            # Actual archived report (finalized value where nothing was archived) instead of an artificial draw.
            x=self.reported[day]
            return dict(e,**{k:x[k] for k in ('values','available','known_final','covariates','filled') if k in x})
        if self.scenario.revision_scope=='early_actual' and cv.season(e['context_dates'][-1])==self.donor_season:
            x=self.reported[e['context_dates'][-1]]
            return dict(e,**{k:x[k] for k in ('values','available','known_final','covariates','filled') if k in x})
        out=super().draw(e,rng)
        if self.scenario.revision_signals=='admissions':
            out['values'][:,3:]=e['values'][:,3:]
            out['available'][:,3:]=e['available'][:,3:]
            if 'covariates' in e:out['covariates']=e['covariates']
        return out


def real_pairs(panel,scenario,held_out,keep):
    """Genuine reported/mature pairs only; no finalized fills supervise revisions."""
    p=cv.masked(panel,keep)
    names=covariate_names_for(scenario.covariate_set)
    reported=episodes(p,scenario.lookback,'reported',names)
    dates=panel['dates'].astype(str);issues=panel['issuance_dates'].astype(str)
    index={d:i for i,d in enumerate(dates)}
    allowed=set(dates[keep]);donor=max(cv.season(d) for d in allowed)
    # vintage_seasons=all (2026-10-06): genuine pairs from every permitted season, not only the latest.
    donors={cv.season(d) for d in allowed} if scenario.vintage_seasons=='all' else {donor}
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
            y=p['asof_targets'][w,index[d]].T
            v[a]=np.nan_to_num(y)
            ok[a]=np.isfinite(y)&e['available'][a]&~e['filled'][a]
        pairs.append((e,dict(e,values=v,available=ok)))
    if not pairs:raise ValueError('No genuine mature report pairs before cutoff')
    return pairs


def fit_corrector(train,bank,scenario,seed,pairs=None,excluded=None,residuals=False):
    rng=np.random.default_rng(seed+410000)
    synthetic=[(bank.draw(e,rng),e) for e in train]
    examples=synthetic if scenario.pilot_nowcaster=='synthetic_tree' else pairs
    if excluded:
        def purge(examples):
            return [(x,dict(y,available=y['available'] & np.array([d not in excluded for d in y['context_dates']])[:,None,None])) for x,y in examples]
        examples=purge(examples);synthetic=purge(synthetic)
    model=TrajectoryNowcaster(min(scenario.correction_weeks,scenario.lookback),scenario.correction_penalty,scenario.correction_strength,'phase')
    model.pathogen_inputs=scenario.pathogen_inputs
    model.channels=[0] if scenario.forecast_targets=='flu' else [0,1,2]
    model.fit(examples,pretraining=synthetic if scenario.pilot_nowcaster=='pretrained_mlp' else None,
              neural=scenario.pilot_nowcaster in ('real_mlp','pretrained_mlp'),seed=seed)
    model.observed_donor_pairs_only=scenario.pilot_nowcaster!='synthetic_tree'
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
            if corrected and self.scenario.nowcast_noise_train:
                e=self.corrector.perturb(e,rng,self.scenario.nowcast_noise)
            out.append(e)
        return out


def cross_correct(train,bank,scenario,seed,pairs):
    # Four disjoint origin blocks; exclude their entire context-date union from
    # correction labels, even when those labels occur in other episodes.
    order=sorted({e['context_dates'][-1] for e in train})
    groups={d:(i//8)%4 for i,d in enumerate(order)}
    result=[]
    for g in range(4):
        selected=[e for e in train if groups[e['context_dates'][-1]]==g]
        excluded={d for e in selected for d in e['context_dates']}
        model=fit_corrector(train,bank,scenario,seed,pairs,excluded)
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


def fit(scenario,seed,held_out,members,device,output,dataset):
    panel=load(dataset);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    effective=replace(scenario,input_mode='reported' if scenario.pilot_method=='reported' else 'scheduled_final')
    full=cv.fold(panel,effective,held_out)
    inner=cv.fold(panel,effective,held_out,inner=True) if scenario.patience else None
    roles=cv.week_roles(panel['dates'],effective,held_out)
    keep=np.isin(roles,['fit','validation'])
    # Reported values are not finalized observations for native loss scales.
    if scenario.pilot_method=='reported':
        for e in full.train+(inner.train if inner else []):e['known_final']=e['filled'].copy()
    bank=ScopedErrors(panel,effective,held_out,keep)
    selection=ScopedErrors(panel,effective,held_out,roles=='fit') if inner else None
    pairs=real_pairs(panel,effective,held_out,keep) if scenario.pilot_nowcaster!='synthetic_tree' else None
    corrector_train=cv.fold(panel,replace(effective,input_mode='scheduled_final'),held_out).train
    corrector=fit_corrector(corrector_train,bank,effective,seed,pairs,residuals=scenario.nowcast_noise>0)
    augmentation=selection_augmentation=None
    validation_corrector=inner_pairs=None
    if scenario.pilot_method in ('errors','joint'):
        augmentation,selection_augmentation=bank,selection
        if inner:inner.validation=selection.batch(inner.validation,np.random.default_rng(seed+800000))
    if scenario.pilot_method in ('corrected','two_stage'):
        if inner:
            inner_pairs=real_pairs(panel,effective,held_out,roles=='fit') if pairs is not None else None
            validation_corrector=fit_corrector(inner.train,selection,effective,seed,inner_pairs,residuals=scenario.nowcast_noise_train)
            inner.train=cross_correct(inner.train,selection,effective,seed,inner_pairs)
            inner.validation=validation_corrector.apply_batch(selection.batch(inner.validation,np.random.default_rng(seed+800000)))
        full.train=cross_correct(full.train,bank,effective,seed,pairs)
        if scenario.correction_realizations>1 or scenario.uncorrected_share or scenario.nowcast_noise_train:
            augmentation=CorrectedHistories(scenario,corrector)
            selection_augmentation=CorrectedHistories(scenario,validation_corrector) if inner else None
    pop=training.populations('data/metadata/locations.csv',full.train[0]['locations'])
    model,records,selector=training.fit_models(inner.train if inner else full.train,inner.validation if inner else None,
        full.train,effective,seed,device,pop,augmenter=augmentation,selection_augmenter=selection_augmentation,keep_selection=True)
    meta=dict(scenario=scenario.scenario_string,config=asdict(scenario),seed=seed,held_out_season=held_out,
        fold=full.info,inner_fold=inner.info if inner else None,records=records,eval_members=members,
        dataset_sha256=sha256(dataset),ili_sha256=sha256(scenario.ili_path) if scenario.ili_training!='none' else None,
        reporting_errors=bank.metadata,corrector=corrector.record(),corrector_kind=scenario.pilot_nowcaster,
        parameter_count=sum(p.numel() for p in model.parameters()),protocol='b3_pilot_standard_hub_inputs',**environment())
    torch.save(checkpoint(model,meta),out/'model.pt')
    with (out/'nowcaster.pkl').open('wb') as f:pickle.dump(corrector,f)
    save(out/'manifest.json',meta)
    model.to(device)
    future=np.flatnonzero(np.array(scenario.horizons)>0).tolist()
    if scenario.pilot_method=='joint':
        model=ForecastSlice(model,future)
    def prep(eps):
        return [subset_labels(e,future) for e in eps] if scenario.pilot_method=='joint' else eps
    # Correctors operate on the episodes already cut at each Hub's own deadline.
    names=['raw','half','corrected']
    if scenario.nowcast_noise:names.append('sampled')
    if scenario.stress_views:
        names.append('delayed')
        if scenario.covariate_set:names.append('nokinsa')
    direct=bool(model.config.get('direct_quantiles'))
    for name in names:
        destination=out if name=='raw' else out/name
        destination.mkdir(exist_ok=True)
        def prepare(eps):
            eps=prep(eps)
            if name=='raw':return eps
            if name=='delayed':
                # Stress view: the newest flu-admission week was not reported by the deadline.
                def late(e):
                    available=e['available'].copy();available[-1,0]=False
                    late_e=dict(e,available=available,values=np.where(available,e['values'],0).astype(np.float32))
                    if 'filled' in e:
                        filled=e['filled'].copy();filled[-1,0]=False;late_e['filled']=filled
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
                return [dict(c,history_samples=np.stack([corrector.perturb(c,rng,scenario.nowcast_noise)['values'] for _ in range(k)])) for c in changed]
            return [dict(e,mixture_episode=c) for e,c in zip(eps,changed)]
        training.evaluate_hubs(model,panel,effective,held_out,seed,members,device,destination,prepare=prepare,first=full.score)
    if selector is not None:
        if validation_corrector is None:
            if inner_pairs is None and pairs is not None:
                inner_pairs=real_pairs(panel,effective,held_out,roles=='fit')
            inner_final=cv.fold(panel,replace(effective,input_mode='scheduled_final'),held_out,inner=True).train
            validation_corrector=fit_corrector(inner_final,selection,effective,seed,inner_pairs)
        calibrate(selector,validation_corrector,inner,panel,effective,held_out,keep,seed,members,device,out,prep,future)
    # Direct history accuracy on the same genuine observed cells, independent of forecast score.
    evaluation_keep=np.array([cv.season(d)==held_out for d in panel['dates']])
    # Maturity labels may be observed after forecast issuance; fitting above never uses them.
    try:
        eval_pairs=real_pairs(panel,effective,held_out,evaluation_keep)
    except ValueError:
        # Production (2026-27): no matured report pairs exist yet, so no direct nowcast diagnostics.
        save(out/'nowcast-diagnostics.json',dict(unavailable='no mature report pairs in the forecast season'))
        return out/'model.pt'
    raw=[e for e,_ in eval_pairs];changed=corrector.apply_batch(raw)
    diagnostics={}
    for k in range(3):
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


def calibrate(selector,corrector,inner,panel,scenario,held_out,keep,seed,members,device,out,prep,future):
    """Fit spread factors on the inner validation weeks, apply them to the corrected view.

    The early-stopped inner model never trained on the validation weeks' labels, and the
    inner corrector excluded them. Calibration forecasts use the evaluation pipeline:
    FluSight-deadline Wednesday reports, cut from the fold's training-season panel only,
    with the newest admission weeks corrected. Origins in 2022-23 are skipped because no
    reports were archived then, so their inputs would all be finalized fills."""
    import json
    from tapestry.dataset.build import for_hub, HUBS
    from tapestry.dataset.episodes import restrict_labels
    from tapestry.evaluation import calibration
    dates=panel['dates'].astype(str);index={d:i for i,d in enumerate(dates)}
    hidden=set(inner.info['validation_weeks'])
    origins={dates[index[d]-h] for d in hidden for h in scenario.horizons if h>0 and index[d]-h>=0}
    origins={d for d in origins if cv.season(d) in CALIBRATION_SEASONS and keep[index[d]]}
    eps=episodes(for_hub(cv.masked(panel,keep),HUBS[0]),scenario.lookback,'reported',
                 covariate_names_for(scenario.covariate_set),scenario.asof_weeks,horizons=scenario.horizons)
    eps=[e for e in eps if e['context_dates'][-1] in origins]
    eps=[e for e in (restrict_labels(e,hidden) for e in eps) if e is not None]
    eps=corrector.apply_batch(prep(eps))
    model=ForecastSlice(selector,future) if scenario.pilot_method=='joint' else selector
    model.to(device)
    folder=out/'calibration';folder.mkdir(exist_ok=True)
    torch.manual_seed(seed+1000)
    training.evaluate(model,eps,members,device,folder)
    with np.load(folder/'forecasts.npz') as f:
        factors,record=calibration.fit(f['quantiles'],f['truth'],f['mask'],f['locations'])
    with np.load(out/'corrected'/'forecasts.npz') as f:
        data={k:f[k] for k in f.files if k!='flu_admission_sum_quantiles'}
    data['quantiles']=calibration.apply(data['quantiles'],factors)
    (out/'calibrated').mkdir(exist_ok=True)
    np.savez_compressed(out/'calibrated'/'forecasts.npz',**data)
    save(folder/'calibration.json',dict(definition=calibration.__doc__,factors={str(k):v for k,v in factors.items()},
        fits={str(k):v for k,v in record.items()},origins=sorted({e['context_dates'][-1] for e in eps}),
        seasons=sorted({cv.season(e['context_dates'][-1]) for e in eps}),applied_to='corrected'))


def evaluation_views(run):
    """Small standard-scorer views of the same checkpoints and alternative inputs."""
    import json
    run=Path(run);manifest=json.loads((run/'manifest.json').read_text())
    result={'raw':run}
    names=[n for n in ('half','corrected','calibrated','sampled','delayed','nokinsa')
           if all((run/f'eval_{season}'/n).exists() for season in manifest['folds'])]
    for name in names:
        view=run/f'evaluation-{name}';view.mkdir(exist_ok=True)
        save(view/'manifest.json',manifest)
        for season in manifest['folds']:
            folder=view/f'eval_{season}';folder.mkdir(exist_ok=True)
            for filename in ('forecasts.npz','manifest.json'):
                src=run/f'eval_{season}'/(name if filename=='forecasts.npz' else '')/filename
                dest=folder/filename
                if not dest.exists():dest.symlink_to(src.resolve())
        result[name]=view
    return result


def cache_pilot_scores(run,frozen):
    """Score each immutable completed run once, in its worker, for cheap large-sweep ranking."""
    import pandas as pd
    from tapestry.evaluation.standard import hub_relative, raw_wis
    run=Path(run)
    if (run/'pilot-view-scores.csv').exists() and (run/'pilot-raw-scores.csv').exists():return
    tables=[];raw=[]
    for history,path in evaluation_views(run).items():
        tables.append(hub_relative(path,frozen).assign(history=history))
        raw.append(raw_wis(path,frozen).assign(history=history))
    pd.concat(tables,ignore_index=True).to_csv(run/'pilot-view-scores.csv',index=False)
    pd.concat(raw,ignore_index=True).to_csv(run/'pilot-raw-scores.csv',index=False)
    from tapestry.evaluation.distribution import distribution_scores
    distribution_scores(run).to_csv(run/'distribution-scores.csv',index=False)


def rank_pilot(runs,frozen,destination):
    """Keep six-target native ranking and report flu natural/log alongside it."""
    import pandas as pd
    from tapestry.evaluation.standard import hub_relative
    records=[]
    raw_records=[]
    distribution_records=[]
    for run in runs:
        cache_pilot_scores(run['path'],frozen)
        if 'quantile_small' in run['config_id'] or 'sum_wis_weight=' in run['config_id'] or (Path(run['path'])/'distribution-scores.csv').exists():
            from tapestry.evaluation.distribution import distribution_scores
            cache=Path(run['path'])/'distribution-scores.csv'
            if not cache.exists():distribution_scores(run['path']).to_csv(cache,index=False)
            distribution_records.append(pd.read_csv(cache).assign(config_id=run['config_id'],seed=run['seed']))
        frame=pd.read_csv(Path(run['path'])/'pilot-view-scores.csv')
        frame=frame[frame.geography=='all'].copy()
        frame['config_id']=run['config_id'];frame['seed']=run['seed']
        records.append(frame)
        raw_records.append(pd.read_csv(Path(run['path'])/'pilot-raw-scores.csv').assign(config_id=run['config_id'],seed=run['seed']))
    if distribution_records:
        pd.concat(distribution_records,ignore_index=True).to_csv(destination/'distribution-scores.csv',index=False)
    pd.concat(raw_records,ignore_index=True).to_csv(destination/'pilot-raw-wis.csv',index=False)
    cells=pd.concat(records,ignore_index=True)
    cells.to_csv(destination/'pilot-target-season-scores.csv',index=False)
    rows=[]
    for (config,seed,history),g in cells.groupby(['config_id','seed','history']):
        flu_only='forecast_targets=flu' in config.split(',')
        for label,subset in [('flu_native' if flu_only else 'six_target_native',g[g.scale=='natural']),
                             ('flu_ed_native',g[(g.target=='wk inc flu prop ed visits')&(g.scale=='natural')]),
                             ('flu_admissions_native',g[(g.target=='wk inc flu hosp')&(g.scale=='natural')]),
                             ('flu_admissions_log',g[(g.target=='wk inc flu hosp')&(g.scale=='log')])]:
            if subset.empty:continue
            seasonal=[]
            for season,sg in subset.groupby('season'):
                weights=np.where(sg.target.str.contains('prop ed'),.5,1.)
                score=float(np.average(sg.wis_ratio,weights=weights));seasonal.append(score)
                rows.append(dict(config_id=config,seed=seed,history=history,metric=label,season=season,score=score))
            if len(seasonal)!=2 and label!='flu_ed_native':raise ValueError('Pilot ranking requires both seasons')
            rows.append(dict(config_id=config,seed=seed,history=history,metric=label,season='equal_season_mean' if len(seasonal)==2 else 'available_season_mean',score=float(np.mean(seasonal))))
    table=pd.DataFrame(rows);table.to_csv(destination/'pilot-seed-scores.csv',index=False)
    summary=table.groupby(['config_id','history','metric','season']).score.agg(['mean','std','count']).reset_index()
    summary=summary.sort_values(['metric','season','mean'])
    summary.to_csv(destination/'pilot-rankings.csv',index=False)
    print(summary[(summary.season=='equal_season_mean')&(summary.metric.isin(['six_target_native','flu_native']))].head(12).to_string(index=False),flush=True)
