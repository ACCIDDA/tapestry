"""Scientific checks for causal, joint nowcast uncertainty and its forecast wiring."""
import numpy as np
import pytest
from tapestry.model.revision_uncertainty import RevisionUncertainty,empirical_crps,trajectory_distribution_scores


def test_empirical_crps_and_trajectory_functionals():
    x=np.array([1.,2.,5.]);y=3.
    expected=abs(x-y).mean()-.5*abs(x[:,None]-x).mean()
    assert empirical_crps(x,y)==pytest.approx(expected)
    truth=np.array([1.,2.,3.,4.])[:,None,None]
    center=truth+1
    result=trajectory_distribution_scores(center[None],truth,center,np.ones((1,1)))
    assert result['point_crps']==1 and result['level_crps']==1
    assert result['growth_crps']==0 and result['trajectory_crps']==pytest.approx(2/3)


def example():
    dates=np.arange(np.datetime64('2023-01-07'),np.datetime64('2025-01-01'),7)
    issues=dates+np.timedelta64(4,'D');n=len(dates)
    base=np.array([[100.,.01],[100.,.01]]) # location, target
    offset=np.array([1.,.0001])
    asof=np.full((n,n,2,2),np.nan)
    for w in range(n):
        for t in range(w+1):
            asof[w,t]=(base+offset)*np.exp(.2*np.sin(t/3) if w-t>=12 else 0)-offset
    predictions=np.broadcast_to(base,(n,8,2,2)).copy()
    panel=dict(dates=dates,issuance_dates=issues,asof_targets=asof,
        locations=np.array(['NC','US']),target_names=np.array(['nhsn_flu_admissions','nssp_flu_ed_visits']))
    return panel,predictions,np.broadcast_to(base.T,(12,2,2)).astype(np.float32)


def test_joint_errors_preserve_locations_and_exclude_future_reports_and_proxies():
    p,pred,center=example();cutoff=70
    u=RevisionUncertainty(p,pred)
    x,scale,meta=u.draw(str(p['issuance_dates'][cutoff]),center)
    np.testing.assert_array_equal(x[:,:-8],np.broadcast_to(center[:-8],x[:,:-8].shape))
    np.testing.assert_array_equal(x[:,:,:,0],x[:,:,:,1])
    assert x[:,-1,1].std()>.0001
    assert np.datetime64(meta['latest_mature_issuance'])<=p['issuance_dates'][cutoff]
    future=dict(p,asof_targets=p['asof_targets'].copy())
    future['asof_targets'][cutoff+1:]=9999
    pred2=pred.copy();pred2[cutoff+1:]=8888
    changed=RevisionUncertainty(future,pred2).draw(str(p['issuance_dates'][cutoff]),center)
    np.testing.assert_array_equal(x,changed[0]);np.testing.assert_array_equal(scale,changed[1])
    # An absent current archived report is an availability proxy, not uncertain evidence.
    future['asof_targets'][cutoff,cutoff,0,1]=np.nan
    proxy=RevisionUncertainty(future,pred2).draw(str(p['issuance_dates'][cutoff]),center)[0]
    np.testing.assert_array_equal(proxy[:,-1,1,0],np.full(len(proxy),center[-1,1,0],np.float32))
    warmup=u.draw(str(p['issuance_dates'][15]),center)[0]
    np.testing.assert_array_equal(warmup,np.broadcast_to(center,warmup.shape))
    # A candidate cannot improve its normalized score by changing the scale.
    for w in [15,cutoff]:
        scale1=u.draw(str(p['issuance_dates'][w]),center,members=2)[1]
        scale2=u.draw(str(p['issuance_dates'][w]),center*20,members=2)[1]
        np.testing.assert_array_equal(scale1,scale2)


@pytest.mark.parametrize('noise',['global','local'])
def test_degenerate_histories_preserve_forecast_latent_pairing(tmp_path,noise):
    import torch
    from tapestry.model.network import Model,IndependentBundle
    from tapestry.experiment.training import evaluate
    models=[Model(lookback=12,width=8,latent=4,geography=False,noise=noise,
                  location_ids=['NC','US'],supplied_final=True) for _ in range(3)]
    model=IndependentBundle(models,[[0,3],[1,4],[2,5]])
    values=np.ones((12,6,2),np.float32);values[:,3:]=.01
    dates=np.arange(np.datetime64('2025-01-04'),np.datetime64('2025-03-23'),7).astype(str)
    e=dict(values=values,available=np.ones_like(values,bool),known_final=np.ones_like(values,bool),
           context_dates=tuple(dates),locations=['NC','US'],target_values=np.ones((4,6,2)),
           target_available=np.ones((4,6,2),bool),target_dates=['2025-03-29','2025-04-05','2025-04-12','2025-04-19'])
    old=tmp_path/'old';new=tmp_path/'new';old.mkdir();new.mkdir()
    torch.manual_seed(1032);evaluate(model,[e],16,'cpu',old)
    e['history_samples']=np.broadcast_to(values,(16,*values.shape)).copy()
    torch.manual_seed(1032);evaluate(model,[e],16,'cpu',new)
    with np.load(old/'forecasts.npz') as a,np.load(new/'forecasts.npz') as b:
        np.testing.assert_allclose(a['quantiles'],b['quantiles'],rtol=1e-5,atol=1e-6)
