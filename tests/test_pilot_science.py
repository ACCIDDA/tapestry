"""Small guards for scientific errors introduced by the exploratory pilot."""
import numpy as np
import torch
from chromantis.model.series import SeriesModel, quantile_loss
from chromantis.experiment.training import mixture_quantiles


def test_ordered_quantile_loss_matches_interval_wis():
    from chromantis.evaluation.standard import quantile_scores
    q=np.linspace(0,20,23)
    y=np.array([7.])
    calculated=quantile_loss(torch.tensor(q[:,None]),torch.tensor(y),torch.ones(1)).item()
    expected=quantile_scores(q[None],y).wis.iloc[0]
    assert np.isclose(calculated,expected)


def test_series_head_units_order_and_masked_inputs():
    torch.manual_seed(42)
    names=('fa','ca','ra','fe','ce','re');units=('count',)*3+('proportion',)*3
    groups=('flu','covid','rsv')*2
    m=SeriesModel(lookback=8,width=16,input_names=names,input_units=units,input_groups=groups,
                  target_names=names,target_units=units,target_groups=groups,target_input_indices=range(6))
    x=torch.rand(2,8,6,3);mask=torch.rand(x.shape)>.2;cal=torch.zeros(2,3)
    a=m(x,mask,cal)
    b=m(torch.where(mask,x,x+10000),mask,cal)
    assert torch.equal(a,b)
    assert bool((a[1:]>=a[:-1]).all())
    assert bool((a>=0).all()) and bool((a[:,:,:,3:]<=1).all())


def test_mixture_is_distribution_mixture_not_quantile_average():
    a=np.zeros((23,1));b=np.full((23,1),10.)
    q=mixture_quantiles(a,b)
    assert q[2,0]==0 and q[-3,0]==10
    assert not np.all(q==5)
