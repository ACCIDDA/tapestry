"""Scientific guards: flu ED receives loss, excluded pathogen histories cannot leak."""
import numpy as np
import torch
from tapestry.model.network import Model, IndependentBundle
from tapestry.model.series import SeriesModel, HistoricalILI
from tapestry.model.objective import LOSS_WEIGHTS
from tapestry.model.revision_tree import TrajectoryNowcaster


def test_flu_scope_excludes_both_values_and_presence():
    torch.manual_seed(42)
    x=torch.rand(2,8,6,3);mask=torch.ones_like(x,dtype=torch.bool);cal=torch.zeros(2,3)
    other=x.clone();other[:,:, [1,2,4,5]]=10000
    absent=mask.clone();absent[:,:,[1,2,4,5]]=False
    for model in [Model(lookback=8,width=16,decoder='quantile',pathogen_inputs='flu'),SeriesModel(lookback=8,width=16,pathogen_inputs='flu')]:
        a=model(values=x,available=mask,calendar=cal)
        b=model(values=other,available=absent,calendar=cal)
        torch.testing.assert_close(a[:,:,:, [0,3]],b[:,:,:, [0,3]],rtol=0,atol=0)


def test_flu_loss_and_bundle_preserve_ed():
    assert LOSS_WEIGHTS['flu_hosp_ed']==[1,0,0,.5,0,0]
    model=IndependentBundle([SeriesModel(lookback=8,width=16),SeriesModel(lookback=8,width=16)],[[0],[3]])
    y=model(values=torch.rand(1,8,6,2),available=torch.ones(1,8,6,2,dtype=torch.bool),calendar=torch.zeros(1,3))
    assert y[:,:,:,3].sum()>0
    assert y[:,:,:, [1,2,4,5]].sum()==0


def test_nowcaster_scope_excludes_other_pathogen_histories():
    m=TrajectoryNowcaster(4,10.,1.,'phase');m.pathogen_inputs='flu'
    e=dict(values=np.ones((8,6,3)),available=np.ones((8,6,3),bool),context_dates=np.array(['2024-10-05']*8))
    a=m.design(e,0)[0]
    e['values'][:,[1,2,4,5]]=10000;e['available'][:,[1,2,4,5]]=False
    np.testing.assert_array_equal(a,m.design(e,0)[0])
