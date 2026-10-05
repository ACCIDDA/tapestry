"""Scientific invariants for boundary dates, vintage isolation and score weighting."""
import numpy as np
import pandas as pd

from tapestry.dataset.finalization import signals, boundary_rows, split, training_statistics, validation_split
from tapestry.experiment.finalization import metrics, baseline_predictions
from tapestry.model.finalization import choose_shrinkage, triangle_predictions
from tapestry.model.scenario import Scenario


def test_boundary_dates_vintages_and_national_support(panel):
    for name, truth, asof, locations in signals(panel):
        rows = boundary_rows(panel, name, truth, asof, locations, 3)
        lag = int(name in ('ilinet_ili', 'clinical_lab_flu_pct_positive', 'flusurv_flu_rate'))
        np.testing.assert_array_equal(np.asarray(rows['issuance'], dtype='datetime64[D]') -
                                      np.asarray(rows['boundary'], dtype='datetime64[D]'),
                                      np.full(len(rows['truth']), np.timedelta64(4 + 7*lag, 'D')))
        for i in (0, len(rows['truth']) // 2, len(rows['truth']) - 1):
            w = np.flatnonzero(panel['issuance_dates'].astype(str) == rows['issuance'][i])[0]
            t = np.flatnonzero(panel['dates'].astype(str) == rows['boundary'][i])[0]
            l = rows['location'][i]
            np.testing.assert_allclose(rows['values'][i, -1], asof[w, t, l], equal_nan=True)
            assert rows['truth'][i] == truth[t, l]
        if name == 'kinsa_ili':
            assert list(locations) == ['US']
            assert len(np.unique(rows['issuance'])) == len(rows['truth'])


def test_future_truth_never_changes_inputs_training_scales_or_bias(panel):
    name, truth, asof, locations = next(signals(panel))
    rows = boundary_rows(panel, name, truth, asof, locations, 3)
    scenario = Scenario(task='finalize', input_mode='vintaged', lookback=3)
    train, score, info = split(panel, rows, scenario, 'rolling_2')
    assert train.any() and score.any() and not (train & score).any()
    assert (np.asarray(rows['boundary'][train], dtype='datetime64[D]') <
            np.datetime64(info['fit_cutoff']) - np.timedelta64(28, 'D')).all()
    a = training_statistics(rows, train)
    changed = dict(rows, truth=rows['truth'].copy())
    changed['truth'][~train] = 1e9
    b = training_statistics(changed, train)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    assert baseline_predictions(rows, train, *a[:2])[2] == baseline_predictions(changed, train, *b[:2])[2]
    altered = boundary_rows(panel, name, truth * 100, asof, locations, 3)
    np.testing.assert_array_equal(rows['values'], altered['values'])


def test_score_weights_locations_equally_despite_missing_weeks():
    frame = pd.DataFrame(dict(signal=['s']*3, kind=['reported']*3, location=['NC', 'NC', 'US'],
                              issuance=['a','b','a'], truth=[0.,0.,0.], prediction=[1.,1.,3.],
                              persistence=[1.,1.,3.], median_revision=[1.,1.,3.], scale=[1.,1.,1.]))
    scored = metrics(frame)
    np.testing.assert_allclose(scored.mae, 2.)
    np.testing.assert_allclose(scored.normalized_mae, 2.)
    np.testing.assert_allclose(scored.rmse, np.sqrt(5.))


def test_multiweek_age_alignment_and_repeated_label_purge(panel):
    name, truth, asof, locations = next(signals(panel))
    rows = boundary_rows(panel, name, truth, asof, locations, 3, weeks=8)
    expected = (4 + 7 * rows['age']).astype('timedelta64[D]')
    np.testing.assert_array_equal(np.asarray(rows['issuance'], dtype='datetime64[D]') -
                                  np.asarray(rows['boundary'], dtype='datetime64[D]'), expected)
    np.testing.assert_allclose(rows['elapsed_weeks'], rows['age'] + 4/7, rtol=1e-6)
    s = Scenario(task='finalize', input_mode='vintaged', lookback=3, finalization_weeks=8)
    train, score, _ = split(panel, rows, s, 'rolling_1')
    assert set(rows['age'][score]) == set(range(8))
    assert not set(rows['boundary'][train]) & set(rows['boundary'][score])


def test_scale_does_not_double_count_same_reference_week(panel):
    name, truth, asof, locations = next(signals(panel))
    rows = boundary_rows(panel, name, truth, asof, locations, 3)
    train = np.isfinite(rows['truth'])
    original = training_statistics(rows, train)
    duplicated = {k: (np.concatenate((v, v[:10])) if isinstance(v, np.ndarray) and k != 'locations' else v)
                  for k, v in rows.items()}
    repeated = training_statistics(duplicated, np.concatenate((train, train[:10])))
    for a, b in zip(original, repeated):
        np.testing.assert_array_equal(a, b)


def test_complete_source_outage_after_first_report_remains_eligible(panel):
    name, truth, asof, locations = next(signals(panel))
    asof = asof.copy()
    asof[30] = np.nan
    rows = boundary_rows(panel, name, truth, asof, locations, 3, weeks=4)
    outage = rows['issuance'] == str(panel['issuance_dates'][30])
    assert outage.any() and rows['active'][outage].all()
    assert np.isnan(rows['values'][outage]).all()


def test_validation_purge_and_feature_vintage_isolation(panel):
    name, truth, asof, locations = next(signals(panel))
    rows = boundary_rows(panel, name, truth, asof, locations, 3, weeks=8)
    scenario = Scenario(task='finalize', input_mode='vintaged', finalization_weeks=8)
    training, scoring, _ = split(panel, rows, scenario, 'rolling_1')
    inner, validation = validation_split(rows, training)
    assert inner.any() and validation.any()
    assert not set(rows['boundary'][inner]) & set(rows['boundary'][validation])
    assert not (validation & scoring).any()
    anchor = np.nan_to_num(rows['baseline_history'][:,-1])
    x, _, _ = triangle_predictions(panel,asof,rows,anchor)
    cutoff = max(rows['issuance'][inner])
    altered = asof.copy()
    altered[panel['issuance_dates'].astype(str)>cutoff] = 1e9
    z, _, _ = triangle_predictions(panel,altered,rows,anchor)
    np.testing.assert_array_equal(x[inner],z[inner])



def test_validation_caps_the_blended_prediction_as_in_inference():
    # Raw candidate 15 with cap 10: alpha=.25 predicts 10, not 9.25.
    # Validation and deployment must apply the cap in the same order.
    alpha, losses = choose_shrinkage(np.array([10.]),np.array([9.]),np.array([6.]),
                                    np.array([1.]),np.array([0]),upper_bound=10.)
    np.testing.assert_array_equal(losses,[1.,0.,0.,0.])
    assert alpha==.25


def test_reported_count_resolution_is_part_of_validation():
    alpha, losses = choose_shrinkage(np.array([1.]),np.array([0.]),np.array([.8]),
                                    np.ones(1),np.zeros(1),integer=True)
    np.testing.assert_array_equal(losses,[1.,1.,1.,0.])
    assert alpha==1


def test_triangle_recovers_known_development_and_preserves_downward_changes():
    dates = np.arange(np.datetime64('2023-01-07'),np.datetime64('2024-06-01'),np.timedelta64(7,'D'))
    issues = dates+np.timedelta64(4,'D')
    asof = np.full((len(dates),len(dates),1),np.nan)
    for w in range(len(dates)):
        for t in range(w+1):
            asof[w,t,0] = 10*(2 if w-t>=1 else 1)
    panel = dict(dates=dates,issuance_dates=issues)
    rows = dict(locations=np.array(['US']),lag=0,issuance=np.array([str(issues[-1])]),boundary=np.array([str(dates[-1])]),
                age=np.array([0]),location=np.array([0]),baseline_history=np.array([[10.]]))
    p, _, _ = triangle_predictions(panel,asof,rows,np.array([10.]),prior_weeks=0)
    np.testing.assert_allclose(p,[20.])
    asof[asof==20]=5
    p, _, _ = triangle_predictions(panel,asof,rows,np.array([10.]),prior_weeks=0)
    np.testing.assert_allclose(p,[5.])


def test_triangle_l1_resists_rare_revisions_and_respects_rate_units():
    dates = np.arange(np.datetime64('2023-01-07'),np.datetime64('2024-06-01'),np.timedelta64(7,'D'))
    issues = dates+np.timedelta64(4,'D')
    asof = np.full((len(dates),len(dates),1),np.nan)
    for w in range(len(dates)):
        for t in range(w+1):
            asof[w,t,0] = 10 if w==t or t%10 else 1000
    panel = dict(dates=dates,issuance_dates=issues)
    rows = dict(locations=np.array(['US']),lag=0,issuance=np.array([str(issues[-1])]),boundary=np.array([str(dates[-1])]),
                age=np.array([0]),location=np.array([0]),baseline_history=np.array([[10.]]))
    p, _, _ = triangle_predictions(panel,asof,rows,np.array([10.]))
    np.testing.assert_allclose(p,[10.])
    scaled=dict(rows,baseline_history=rows['baseline_history']/100)
    q, _, _ = triangle_predictions(panel,asof/100,scaled,np.array([.1]))
    np.testing.assert_allclose(q,p/100)


def test_gap_inputs_exclude_current_target_and_future_vintages(panel):
    from tapestry.dataset.finalization import gap_features
    name,truth,asof,locations=next(signals(panel))
    rows=boundary_rows(panel,name,truth,asof,locations,12,8)
    train,_,_=split(panel,rows,Scenario(task='finalize',input_mode='vintaged',finalization_weeks=8),'rolling_1')
    x,_,_=gap_features(panel,rows,name,train)
    changed=dict(panel,targets=panel['targets']*1000)
    y,_,_=gap_features(changed,rows,name,train)
    np.testing.assert_array_equal(x,y)
    cutoff=max(rows['issuance'][train]);changed=dict(panel,asof_targets=panel['asof_targets'].copy())
    changed['asof_targets'][panel['issuance_dates'].astype(str)>cutoff]=1e8
    y,_,_=gap_features(changed,rows,name,train)
    np.testing.assert_array_equal(x[train],y[train])
    single={k:(v[-1:] if isinstance(v,np.ndarray) and k!='locations' else v) for k,v in rows.items()}
    a,_,_=gap_features(panel,single,name,np.ones(1,bool))
    w=np.searchsorted(panel['issuance_dates'].astype(str),single['issuance'][0])
    t=np.searchsorted(panel['dates'].astype(str),single['boundary'][0])
    changed=dict(panel,asof_targets=panel['asof_targets'].copy());changed['asof_targets'][w,t,:,0]=1e8
    b,_,_=gap_features(changed,single,name,np.ones(1,bool))
    np.testing.assert_array_equal(a,b)


def test_shared_triangle_borrows_without_national_or_population_domination():
    dates=np.arange(np.datetime64('2023-01-07'),np.datetime64('2024-06-01'),np.timedelta64(7,'D'))
    issues=dates+np.timedelta64(4,'D')
    locations=np.array(['A','B','C','D','US'])
    base=np.array([10.,10.,1e6,10.,1e8])
    factor=np.array([2.,2.,3.,1.,10.])
    asof=np.full((len(dates),len(dates),5),np.nan)
    for w in range(len(dates)):
        for t in range(w+1):asof[w,t]=base*(factor if w>t else 1.)
    asof[:-1,:,3]=np.nan  # D has no local pairs but has a report today.
    panel=dict(dates=dates,issuance_dates=issues)
    rows=dict(locations=locations,lag=0,issuance=np.array([str(issues[-1])]),
        boundary=np.array([str(dates[-1])]),age=np.array([0]),location=np.array([3]),
        baseline_history=np.array([[10.]]))
    pred,_,_=triangle_predictions(panel,asof,rows,np.array([10.]))
    np.testing.assert_allclose(pred,[20.])  # Equal-state center is 2, not 3 or 10.


def test_seasonal_curves_causal_and_unit_preserving():
    from tapestry.model.finalization import seasonal_predictions
    dates = np.arange(np.datetime64('2023-01-07'), np.datetime64('2025-06-01'), np.timedelta64(7, 'D'))
    issues = dates + np.timedelta64(4, 'D')
    asof = np.full((len(dates), len(dates), 1), np.nan)
    for w in range(len(dates)):
        for t in range(w+1):
            asof[w, t, 0] = 10 * (1.5 if w-t >= 1 else 1.)
    panel = dict(dates=dates, issuance_dates=issues)
    i = len(dates)-10
    rows = dict(locations=np.array(['US']), lag=0, issuance=np.array([str(issues[i])]),
                boundary=np.array([str(dates[i])]), age=np.array([0]), location=np.array([0]),
                baseline_history=np.array([[10.]]))
    changed = asof.copy()
    changed[i+1:] = 1e9
    for mode in ('seasonal', 'adaptive', 'adaptive_chain'):
        p, _, _ = seasonal_predictions(panel, asof, rows, np.array([10.]), mode=mode)
        q, _, _ = seasonal_predictions(panel, changed, rows, np.array([10.]), mode=mode)
        np.testing.assert_allclose(p, [15.])
        np.testing.assert_array_equal(p, q)
        scaled = dict(rows, baseline_history=rows['baseline_history']/100)
        r, _, _ = seasonal_predictions(panel, asof/100, scaled, np.array([.1]), mode=mode)
        np.testing.assert_allclose(r, p/100)


def test_seasonal_gap_uses_issuance_vintage_only():
    from tapestry.model.finalization import seasonal_gap_predictions
    dates=np.arange(np.datetime64('2023-01-07'),np.datetime64('2025-06-01'),np.timedelta64(7,'D'))
    issues=dates+np.timedelta64(4,'D');n=len(dates)
    asof=np.full((n,n,1),np.nan)
    for w in range(n):
        asof[w,:w+1,0]=np.exp(np.arange(w+1)*.05)
    i=n-10;asof[i,i-3:i+1]=np.nan
    panel=dict(dates=dates,issuance_dates=issues)
    rows=dict(issuance=np.array([str(issues[i])]),boundary=np.array([str(dates[i])]),location=np.array([0]),baseline_history=np.array([[np.nan]]))
    altered=asof.copy();altered[i+1:]=1e9
    for trend in (False,True):
        a=seasonal_gap_predictions(panel,asof,rows,np.ones(1),trend=trend)
        b=seasonal_gap_predictions(panel,altered,rows,np.ones(1),trend=trend)
        np.testing.assert_allclose(a,[np.exp(i*.05)])
        np.testing.assert_array_equal(a,b)


def test_target_percent_metrics_use_observed_denominator_and_exclude_zero_ape():
    from tapestry.experiment.finalization import target_percent_metrics
    f=pd.DataFrame(dict(signal=['nhsn_flu_admissions']*3,age=[0]*3,location=['NC','NC','NY'],
        truth=[100.,0.,10.],prediction=[105.,1.,12.],scale=[100.,100.,10.]))
    r=target_percent_metrics(f).iloc[0]
    np.testing.assert_allclose(r.wape,8/110)
    np.testing.assert_allclose(r.location_wape,(6/100+2/10)/2)
    np.testing.assert_allclose(r.within5,.5)


def test_proxy_gap_never_reads_future_vintages_or_reference_labels(panel):
    from tapestry.model.finalization import proxy_gap_predictions
    name,truth,asof,locations=next(signals(panel))
    rows=boundary_rows(panel,name,truth,asof,locations,12,1)
    issue=np.unique(rows['issuance'])[80]
    keep=rows['issuance']==issue
    rows={k:(v[keep] if isinstance(v,np.ndarray) and k!='locations' else v) for k,v in rows.items()}
    wi=np.searchsorted(panel['issuance_dates'],np.datetime64(issue));ti=np.searchsorted(panel['dates'],np.datetime64(rows['boundary'][0]))
    asof=asof.copy();asof[wi,ti]=np.nan;rows['baseline_history'][:,-1]=np.nan
    changed=dict(panel,targets=panel['targets']*1e8,asof_covariates=panel['asof_covariates'].copy(),asof_targets=panel['asof_targets'].copy())
    changed['asof_covariates'][wi+1:]=1e9;changed['asof_targets'][wi+1:]=1e9
    altered=asof.copy();altered[wi+1:]=1e9
    a=proxy_gap_predictions(panel,asof,rows,np.ones(len(keep[keep])),name)
    b=proxy_gap_predictions(changed,altered,rows,np.ones(len(keep[keep])),name)
    np.testing.assert_array_equal(a,b)


def test_adaptive_median_resists_rare_revisions():
    from tapestry.model.finalization import seasonal_predictions
    dates=np.arange(np.datetime64('2023-01-07'),np.datetime64('2025-06-01'),np.timedelta64(7,'D'))
    issues=dates+np.timedelta64(4,'D');n=len(dates)
    a=np.full((n,n,1),np.nan)
    for w in range(n):
        for t in range(w+1):a[w,t,0]=1000 if w>t and t%10==0 else 10
    p=dict(dates=dates,issuance_dates=issues)
    r=dict(locations=np.array(['US']),lag=0,issuance=np.array([str(issues[-1])]),
           boundary=np.array([str(dates[-1])]),age=np.array([0]),location=np.array([0]),baseline_history=np.array([[10.]]))
    y,_,_=seasonal_predictions(p,a,r,np.array([10.]),mode='adaptive_chain',statistic='median')
    np.testing.assert_allclose(y,[10.])


def test_proxy_long_gap_cannot_inherit_reference_fallback(panel):
    from tapestry.model.finalization import proxy_gap_predictions
    name,truth,asof,locations=next(signals(panel))
    rows=boundary_rows(panel,name,truth,asof,locations,12,1)
    issue=np.unique(rows['issuance'])[80];keep=rows['issuance']==issue
    rows={k:(v[keep] if isinstance(v,np.ndarray) and k!='locations' else v) for k,v in rows.items()}
    wi=np.searchsorted(panel['issuance_dates'],np.datetime64(issue));ti=np.searchsorted(panel['dates'],np.datetime64(rows['boundary'][0]))
    asof=asof.copy();asof[wi,ti-20:]=np.nan;rows['baseline_history'][:]=np.nan
    a=proxy_gap_predictions(panel,asof,rows,np.zeros(len(rows['location'])),name)
    b=proxy_gap_predictions(panel,asof,rows,np.full(len(rows['location']),1e9),name)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_allclose(a,np.nanmedian(asof[wi,:ti-20],axis=0))
    asof[wi]=np.nan
    z=proxy_gap_predictions(panel,asof,rows,np.full(len(rows['location']),1e9),name)
    np.testing.assert_array_equal(z,0.)


def test_reporting_support_distinguishes_backfill_from_uninterrupted_reports(panel):
    from tapestry.experiment.finalization import reporting_support
    p=dict(panel,asof_targets=panel['asof_targets'].copy());w=100
    t=np.searchsorted(p['dates'],p['issuance_dates'][w]-np.timedelta64(4,'D'))
    for j in range(8):p['asof_targets'][w-j,t-j,0,0]=1.
    f=pd.DataFrame(dict(signal=[str(p['target_names'][0])],issuance=[str(p['issuance_dates'][w])],
        boundary=[str(p['dates'][t])],location=[str(p['locations'][0])]))
    a=reporting_support(p,f).iloc[0]
    assert a.complete_history_12 and a.uninterrupted_8
    p['asof_targets'][w-3,t-3,0,0]=np.nan
    b=reporting_support(p,f).iloc[0]
    assert b.complete_history_12 and not b.uninterrupted_8
    p['asof_targets'][w,t-5,0,0]=np.nan
    c=reporting_support(p,f).iloc[0]
    assert not c.complete_history_12 and not c.uninterrupted_8
