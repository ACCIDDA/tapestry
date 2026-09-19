"""Scientific boundaries: task weights, leakage, revision units and forecast bypass."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from tapestry.model_data.wednesday import output_dates
from tapestry.models.b1 import B1
from tapestry.models.b1_run import objective_weights
from tapestry.models.b1_scenarios import B1Scenario
from tapestry.models.b1_revision import RevisionAugmenter


@pytest.mark.parametrize('weight', [.1, .2])
@pytest.mark.parametrize('pipeline', ['joint_aux', 'two_stage', 'gated_revision'])
def test_forecast_selection_and_recent_weight(weight, pipeline):
    context, dates = output_dates('2025-01-08')
    x = np.ones((12, 6, 3, 2), np.float32); x[-2:, :, 2] = 0
    e = dict(X=x, Y=np.ones((6, 6, 2, 2), np.float32), context_dates=context,
             target_dates=dates, locations=('NC', 'US'))
    s = B1Scenario(pipeline=pipeline, nowcast_weight=weight, validation_mode='natural_forecast')
    train = objective_weights([e], [0, 3], s)
    val = objective_weights([e], [0, 3], s, validation=True)
    assert train[:, :2].sum() == pytest.approx(weight)
    assert train[:, 2:].sum() == pytest.approx(1.)
    assert not val[:, :2].any()
    np.testing.assert_array_equal(train[:, 2:], val[:, 2:])
    np.testing.assert_allclose(train.sum((0, 1, 2)), [(.8*(1+weight)), (.2*(1+weight))], rtol=1e-6)
    x[-2:, :, 2] = 1
    absent = objective_weights([e], [0, 3], s)
    assert not absent[:, :2].any()
    assert absent[:, 2:].sum() == pytest.approx(1.)


def inputs():
    x = torch.full((2, 12, 6, 2), .02); x[:, :, :3] = 100
    x[:, -2, :3] = 80
    a = torch.ones_like(x, dtype=torch.bool)
    k = a.clone(); k[:, -2:] = False
    return x, a, k


def model(**kwargs):
    return B1(list(range(6)), {'NC':1e6, 'US':1e8}, ['NC','US'], width=8, latent=4,
              supplied_final=True, **kwargs)


def test_revision_bridge_starts_at_B_and_hides_values_and_final_flags():
    torch.manual_seed(42); base = model(direct=True)
    torch.manual_seed(42); revised = model(parallel_recent=True, revision_bridge=True)
    x,a,k = inputs(); cal=torch.zeros(2,3)
    z=torch.randn(5,2,4); zr=torch.randn_like(z)
    expected=base(x,a,cal,known_final=k,z_future=z)
    got=revised(x,a,cal,known_final=k,z_future=z,z_recent=zr)
    torch.testing.assert_close(got[:,:,2:],expected,rtol=1e-6,atol=1e-6)
    # Known finals remain exactly native, including zero counts and ED endpoints.
    k[:, -1]=True; x[:, -1, 0]=0; x[:, -1, 3]=1
    result=revised(x,a,cal,known_final=k,z_future=z,z_recent=zr)
    torch.testing.assert_close(result[:,:,1],x[:,-1][None].expand(5,-1,-1,-1),rtol=0,atol=0)
    dropout=torch.zeros_like(a);dropout[:,-2]=True
    hidden=revised(x,a,cal,known_final=k,dropout=dropout,z_future=z,z_recent=zr)
    changed=x.clone();changed[:,-2]=float('nan'); k[:,-2]=True
    torch.testing.assert_close(hidden,revised(changed,a,cal,known_final=k,dropout=dropout,z_future=z,z_recent=zr))
    # The zero output initialization must still admit a forecast-loss gradient.
    hidden[:,:,2:].sum().backward()
    assert revised.direct_model.revision_adjustment[-1].weight.grad.abs().sum()>0


def test_zero_recent_correction_is_anchored_to_each_observed_week():
    m=model(parallel_recent=True,revision_bridge=True)
    with torch.no_grad():
        for p in m.direct_model.recent_heads.parameters(): p.zero_()
    x,a,k=inputs()
    result=m(x,a,torch.zeros(2,3),members=3,known_final=k)
    torch.testing.assert_close(result[:,:,:2],x[:,-2:][None].expand(3,-1,-1,-1,-1),rtol=1e-5,atol=1e-5)


def test_revision_bank_excludes_hidden_truth_and_preserves_units_and_status():
    m=model(direct=True);x,a,k=inputs()
    truth=x[:,-2:].clone();truth[:,:,:3]*=1.2;truth[:,:,3:]*=1.1
    valid=torch.ones_like(truth,dtype=torch.bool)
    # No permitted truth for channel 1; supplied finals in channel 2.
    valid[:,:,1]=False;k[:,-2:,2]=True
    bank=RevisionAugmenter(m,x,a,k,truth,valid)
    changed=truth.clone();changed[:,:,1]=float('nan');changed[:,:,2]=1e20
    other=RevisionAugmenter(m,x,a,k,changed,valid)
    torch.testing.assert_close(bank.errors,other.errors)
    assert not bank.eligible[:,:,1:3].any()
    # Identical donors reproduce reports, not report + a second reporting bias.
    d=torch.zeros_like(a)
    augmented,donors,count=bank.apply(x,d,np.random.default_rng(4),1.)
    torch.testing.assert_close(augmented,x,rtol=1e-5,atol=1e-5)
    assert count>0 and (donors>=0).all()
    d[:,-2:]=True
    augmented,_,count=bank.apply(x,d,np.random.default_rng(4),1.)
    torch.testing.assert_close(augmented,x,rtol=0,atol=0)
    assert count==0


def test_nowcast_baseline_uses_own_week_and_excludes_supplied_finals(tmp_path):
    from tapestry.evaluation.nowcast import nowcast_cells
    from tapestry.models.season_cv import SEASONS
    from tapestry.models.quantiles import LEVELS
    for held in SEASONS:
        folder=tmp_path/f'eval_{held}';folder.mkdir()
        truth=np.full((1,2,6,1),10.)
        valid=np.zeros_like(truth,dtype=bool);valid[:,:,:1]=True
        base=np.zeros_like(truth);base[:,0]=4.;base[:,1]=9.
        final=np.zeros((1,12,6,1),bool);final[:,-1,0]=True
        np.savez(folder/'forecasts-demo-s42-natural.npz',horizons=[-2,-1],
            quantiles=np.repeat(truth[None],len(LEVELS),axis=0),quantile_levels=LEVELS,
            truth=truth,mask=valid,locations=['US'],X_final=final,
            target_dates=[[f'{held[5:]}-01-06',f'{held[5:]}-01-13']],
            baseline=base,baseline_mask=valid,baseline_kind='same-week genuine preliminary report')
    totals,audit=nowcast_cells(tmp_path,dict(run_id='demo',seed=42))
    assert audit['scored_cells']==3
    assert audit['excluded_supplied_final']==3
    assert totals.model_wis.sum()==0
    # The older week's report is 4: error 6 each season, not newest report 9.
    assert totals.ensemble_wis.sum()==18
    assert audit['score_version']=='nowcast-same-week-report-season-first-us20-v3'
