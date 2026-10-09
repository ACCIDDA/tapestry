import numpy as np
import torch
from chromantis.model.network import ForecastHead
from chromantis.experiment.training import four_week_sum_wis, sum_wis_weights
from chromantis.evaluation.standard import quantile_scores


def test_small_quantiles_order_and_two_spreads():
    head=ForecastHead(8,4,False,'quantile_small')
    q=head(torch.randn(2,4,3,6,8),None,None)
    assert q.shape==(23,2,4,3,6,1)
    assert (q[1:]>q[:-1]).all()
    q.sum().backward()
    assert torch.isfinite(head.shape_gaps.grad).all()


def test_sum_wis_uses_member_alignment_and_excludes_reconstruction():
    draws=torch.tensor([0.,1.,2.,3.])
    x=torch.zeros(4,1,6,6,1,requires_grad=True)
    # Perfect anticorrelation makes every four-week sum identical, despite wide marginals.
    z=x+0
    z[:,:,0:2]=10000
    z[:,0,2,0,0]=draws
    z[:,0,3,0,0]=3-draws
    z[:,0,4,0,0]=draws
    z[:,0,5,0,0]=3-draws
    y=torch.zeros(1,6,6,1);y[:,2:,0]=1.5
    m=torch.ones_like(y,dtype=torch.bool)
    loss=four_week_sum_wis(z,y,m,[-1,0,1,2,3,4])
    assert loss.item()==0
    y[:,5,0]=2.5
    loss=four_week_sum_wis(z,y,m,[-1,0,1,2,3,4])
    expected=quantile_scores(np.full((1,23),6.),np.array([7.])).wis.iloc[0]
    assert np.isclose(loss.item(),expected)
    loss.sum().backward()
    assert torch.isfinite(x.grad).all()
    m[:,2,0]=False
    assert four_week_sum_wis(z,y,m,[-1,0,1,2,3,4]).item()==0


def test_sum_weights_equal_seasons_and_geographies():
    episodes=[]
    for dates,n in [(['2024-12-07','2024-12-14','2024-12-21','2024-12-28'],2),(['2025-12-06','2025-12-13','2025-12-20','2025-12-27'],4)]:
        for _ in range(n):episodes.append(dict(Y=np.ones((4,6,2,3)),target_dates=dates,locations=['A','B','US']))
    w=sum_wis_weights(episodes,[1,2,3,4])
    assert np.isclose(w[:2].sum(),.5) and np.isclose(w[2:].sum(),.5)
    assert np.isclose(w[:,:,:,2].sum(),.2)
    assert np.isclose(w[:,:,:,:2].sum(),.8)
