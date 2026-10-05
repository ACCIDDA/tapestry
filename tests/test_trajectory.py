import numpy as np
import pandas as pd
import pytest
from tapestry.evaluation.trajectory import trajectory_cells
from tapestry.model.finalization import seasonal_predictions


def test_trajectory_distinguishes_level_from_growth():
    f = pd.DataFrame(dict(signal=['s']*4, issuance=['2025-01-01']*4, location=['US']*4,
        age=[0,1,2,3], truth=[4.,3.,2.,1.], prediction=[5.,4.,3.,2.], scale=[10.]*4))
    g = trajectory_cells(f).iloc[0]
    np.testing.assert_allclose([g.point,g.level,g.growth,g.trajectory],[.1,.1,0.,.2/3])
    f.prediction=[6.,5.,2.,1.]
    g=trajectory_cells(f).iloc[0]
    np.testing.assert_allclose([g.point,g.level,g.growth],[.2,.1,.2])
    assert trajectory_cells(f.iloc[:3]).empty


def test_conditional_curve_future_vintage_isolation_and_rate_units():
    dates=np.arange(np.datetime64('2023-01-07'),np.datetime64('2024-06-01'),np.timedelta64(7,'D'))
    issues=dates+np.timedelta64(4,'D'); n=len(dates)
    a=np.full((n,n,1),np.nan)
    for w in range(n):
        for t in range(w+1):
            a[w,t,0]=(10+np.sin(t/4))*(1 if w==t else 1.2)
    t=n-8
    panel=dict(dates=dates,issuance_dates=issues)
    rows=dict(locations=np.array(['US']),lag=0,issuance=np.array([str(issues[t])]),
        boundary=np.array([str(dates[t])]),age=np.array([0]),location=np.array([0]),
        baseline_history=np.array([[a[t,t,0]]]))
    def pred(v, r):
        return seasonal_predictions(panel,v,r,np.zeros(1),mode='conditional_chain',statistic='median')[0]
    p=pred(a,rows); b=a.copy(); b[t+1:]=1e8
    np.testing.assert_array_equal(p,pred(b,rows))
    np.testing.assert_allclose(p/100,pred(a/100,dict(rows,baseline_history=rows['baseline_history']/100)))
    np.testing.assert_allclose(p,[1.2*a[t,t,0]])


@pytest.mark.parametrize('features',['basic','momentum','age','age_momentum'])
def test_residual_training_uses_only_due_mature_reports(features):
    from tapestry.dataset.finalization import boundary_rows
    from tapestry.model.context_nowcast import residual_predictions
    dates=np.arange(np.datetime64('2023-01-07'),np.datetime64('2024-10-01'),np.timedelta64(7,'D'))
    issues=dates+np.timedelta64(4,'D');n=len(dates)
    a=np.full((n,n,1),np.nan)
    for w in range(n):a[w,:w+1]=10.
    panel=dict(dates=dates,issuance_dates=issues)
    rows=boundary_rows(panel,'nhsn_flu_admissions',np.zeros((n,1)),a,np.array(['US']),12,4)
    base=np.full(len(rows['age']),8.)
    p=residual_predictions(panel,a,rows,base,integer=True,penalty=10.,features_mode=features)
    last=rows['issuance']==str(issues[-1])
    assert abs(p[last]-10).mean()<.2
    b=a.copy();cutoff=60;b[cutoff+1:]=10000.
    q=residual_predictions(panel,b,rows,base,integer=True,penalty=10.,features_mode=features)
    early=rows['issuance']<=str(issues[cutoff])
    np.testing.assert_array_equal(p[early],q[early])


def test_revision_momentum_aligns_event_weeks_and_preserves_rate_units():
    from tapestry.model.context_nowcast import revision_momentum
    a=np.ones((8,8,2))*10
    a[5,4,1]=12; a[5,3,1]=15; a[5,2,1]=20
    f=revision_momentum(a,np.array([5]),np.array([5]),np.array([1]),np.array([0]))
    np.testing.assert_allclose(f[0,:3],np.log([1.2,1.5,2]))
    scaled=revision_momentum(a/100,np.array([5]),np.array([5]),np.array([1]),np.array([0]))
    np.testing.assert_allclose(f,scaled)
    a[6:]=999
    np.testing.assert_array_equal(f,revision_momentum(a,np.array([5]),np.array([5]),np.array([1]),np.array([0])))


def test_causal_gate_uses_mature_prequential_paths_and_rejects_harm():
    from tapestry.dataset.finalization import boundary_rows
    from tapestry.model.context_nowcast import causal_gate
    dates=np.arange(np.datetime64('2023-01-07'),np.datetime64('2024-05-01'),np.timedelta64(7,'D'))
    issues=dates+np.timedelta64(4,'D');n=len(dates)
    a=np.full((n,n,1),np.nan)
    for w in range(n):a[w,:w+1]=10.
    panel=dict(dates=dates,issuance_dates=issues)
    rows=boundary_rows(panel,'nhsn_flu_admissions',np.zeros((n,1)),a,np.array(['US']),12,4)
    base=np.full(len(rows['age']),8.);candidate=np.full(len(base),10.)
    result,alpha=causal_gate(panel,a,rows,base,candidate,True)
    assert np.all(alpha[rows['issuance']<str(issues[12])]==0)
    assert np.all(result[rows['issuance']==str(issues[-1])]==10)
    bad,_=causal_gate(panel,a,rows,base,np.full(len(base),6.),True)
    np.testing.assert_array_equal(bad,base)
    b=a.copy();b[46:]=10000.
    altered,_=causal_gate(panel,b,rows,base,candidate,True)
    early=rows['issuance']<=str(issues[45])
    np.testing.assert_array_equal(result[early],altered[early])
