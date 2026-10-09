"""Nonlinear residual nowcaster trained only on permitted synthetic trajectories.

The predictor sees reported histories, their masks, annual phase, and optional
reported covariates. Finalized histories are labels, never predictors. Tree
capacity is deliberately small; `experiment/fit.py` cross-fits origin blocks before
this nowcaster supplies forecaster inputs.

Saved correctors (`nowcaster.pkl`) are pickles of `TrajectoryNowcaster`: keep this
module path and class name. Its former base class, the linear `RevisionRegression`
of the deleted B2 route, was folded in on 2026-10-08 (only `__init__` was inherited).
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from tapestry.dataset.cv import season
from tapestry.dataset.episodes import calendar

class TrajectoryNowcaster:
    def __init__(self, weeks=4, penalty=10., strength=1., features='phase'):
        self.weeks, self.penalty, self.strength, self.features = weeks, penalty, strength, features

    def design(self,e,channel):
        values=e['values'].astype(float); valid=e['available'].copy()
        channels={'all':range(6),'flu':[0,3],'flu_hosp':[0],'flu_ed':[3],'flu_covid':[0,1,3,4],'flu_rsv':[0,2,3,5]}[getattr(self,'pathogen_inputs','all')]
        valid[:,[c for c in range(6) if c not in channels]]=False
        values=np.where(valid,values,0)
        units=np.array([1.,1.,1.,.0001,.0001,.0001])[:,None]
        scale=np.maximum(np.max(np.where(valid,values,0),axis=0),units)
        floor=np.maximum(.05*scale,units)
        log=np.log((values+floor)/(scale+floor))
        log=np.where(valid,log,np.nan)
        l=values.shape[-1]
        # A full recent path preserves direction, acceleration and isolated dips.
        own=log[-8:,channel].T
        ownmask=valid[-8:,channel].T.astype(float)
        cross=[]
        for a,b in ((-2,None),(-4,-2),(-8,-4)):
            z=log[a:b];ok=valid[a:b]
            cross.append(np.divide(np.nansum(z,axis=0),ok.sum(0),out=np.zeros_like(scale),where=ok.sum(0)>0).T)
        cal=np.broadcast_to(calendar([e['context_dates'][-1]])[0],(l,3))
        pieces=[own,ownmask,*cross,cal,np.eye(l)]
        if 'covariates' in e:
            cov=e['covariates']; cv=cov[...,0,:]; ca=cov[...,1,:].astype(bool)
            for a,b in ((-2,None),(-6,-2)):
                x=np.divide(np.where(ca[a:b],cv[a:b],0).sum(0),ca[a:b].sum(0),out=np.zeros_like(cv[0]),where=ca[a:b].sum(0)>0)
                pieces.append(np.sign(x.T)*np.log1p(np.abs(x.T)))
        shared=np.concatenate(pieces,axis=1)
        # Keep age explicit, with one common estimator to share phase relationships.
        rows=[np.column_stack((np.full(l,age),shared)) for age in range(self.weeks)]
        return np.stack(rows),floor[channel]

    def fit(self,examples, pretraining=None, neural=False, seed=42):
        self.models=[];self.training_cells=[];self.training_cells_by_age=[];self.neural=neural;self.pretrained=bool(pretraining)
        self.training_seasons=sorted({season(t['context_dates'][-1]) for _,t in examples})
        def arrays(examples,k):
            xs=[];ys=[]
            indices=np.arange(-1,-self.weeks-1,-1)
            for reported,truth in examples:
                x,floor=self.design(reported,k)
                ok=reported['available'][indices,k]&truth['available'][indices,k]
                y=np.log((truth['values'][indices,k]+floor)/(reported['values'][indices,k]+floor))
                xs.append(x[ok]);ys.append(y[ok])
            return np.concatenate(xs),np.concatenate(ys)
        for k in getattr(self,'channels',[0,1,2]):
            x,y=arrays(examples,k);self.training_cells.append(len(y))
            self.training_cells_by_age.append([int((x[:,0]==age).sum()) for age in range(self.weeks)])
            if len(y)<30:
                raise ValueError(f'Only {len(y)} real/synthetic admission pairs for channel {k}')
            if neural:
                model=NeuralResidual(x.shape[1],seed+k)
                if pretraining:
                    px,py=arrays(pretraining,k);model.fit(px,py)
                model.fit(x,y)
            else:
                model=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,min_samples_leaf=100,
                    l2_regularization=self.penalty,learning_rate=.05,early_stopping=False,random_state=seed)
                model.fit(x,y)
            self.models.append(model)
        return self

    def fit_residuals(self,examples,seed=42,folds=4):
        """Out-of-fold log-correction errors, one whole age vector per (example, location).

        Origins are split into blocks of 8 consecutive dates assigned to `folds` groups. For
        each group a tree with the same settings is fitted on the other groups, with labels
        on any date in the held group's context windows removed (as in `cross_correct`),
        and its errors y - prediction on the held group are pooled. Keeping the age vector
        together preserves dependence between the newest weeks. Trees only."""
        if self.neural:
            raise ValueError('Correction residual pools are implemented for tree nowcasters only')
        order=sorted({t['context_dates'][-1] for _,t in examples})
        group={d:(i//8)%folds for i,d in enumerate(order)}
        indices=np.arange(-1,-self.weeks-1,-1)
        def target(reported,truth,k):
            x,floor=self.design(reported,k)
            ok=reported['available'][indices,k]&truth['available'][indices,k]
            y=np.log((truth['values'][indices,k]+floor)/(reported['values'][indices,k]+floor))
            return x,ok,y
        self.residual_pools=[]
        for k in getattr(self,'channels',[0,1,2]):
            rows=[]
            for g in range(folds):
                test=[ex for ex in examples if group[ex[1]['context_dates'][-1]]==g]
                if not test:continue
                excluded={d for _,t in test for d in t['context_dates']}
                xs=[];ys=[]
                for reported,truth in examples:
                    if group[truth['context_dates'][-1]]==g:continue
                    keep=np.array([d not in excluded for d in truth['context_dates']])[indices]
                    x,ok,y=target(reported,truth,k);ok=ok&keep[:,None]
                    xs.append(x[ok]);ys.append(y[ok])
                x=np.concatenate(xs);y=np.concatenate(ys)
                if len(y)<30:continue
                model=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,min_samples_leaf=100,
                    l2_regularization=self.penalty,learning_rate=.05,early_stopping=False,random_state=seed).fit(x,y)
                for reported,truth in test:
                    x,ok,y=target(reported,truth,k)
                    pred=np.clip(model.predict(x.reshape(-1,x.shape[-1])).reshape(x.shape[:2]),-np.log(4),np.log(4))
                    r=np.where(ok,y-pred,np.nan).T
                    rows.append(r[ok.any(0)])
            if not rows:
                raise ValueError(f'No out-of-fold correction residuals for channel {k}')
            self.residual_pools.append(np.concatenate(rows).astype(np.float32))
        return self

    def perturb(self,e,rng,scale):
        """One plausible history: add scale x a sampled out-of-fold error vector per location to the corrected weeks."""
        v=e['values'].copy()
        indices=np.arange(-1,-self.weeks-1,-1)
        for k,pool in zip(getattr(self,'channels',[0,1,2]),self.residual_pools):
            valid=e['available'][:,k]
            unit=1. if k<3 else .0001
            floor=np.maximum(.05*np.maximum(np.max(np.where(valid,v[:,k],0),axis=0),unit),unit)
            r=np.nan_to_num(pool[rng.integers(len(pool),size=v.shape[-1])]).T  # [weeks, locations]
            v[indices,k]=np.maximum(0,(v[indices,k]+floor)*np.exp(scale*r)-floor)
        v[:,3:]=np.minimum(1,v[:,3:])
        return dict(e,values=np.where(e['available'],v,0).astype(np.float32))

    def apply_batch(self,episodes):
        values=[e['values'].copy() for e in episodes]
        indices=np.arange(-1,-self.weeks-1,-1)
        for k,model in zip(getattr(self,'channels',[0,1,2]),self.models):
            designs=[self.design(e,k) for e in episodes]
            matrices=[x.reshape(-1,x.shape[-1]) for x,_ in designs]
            predictions=model.predict(np.concatenate(matrices))
            pos=0
            for v,(x,floor),matrix in zip(values,designs,matrices):
                correction=np.clip(predictions[pos:pos+len(matrix)].reshape(x.shape[:2]),-np.log(4),np.log(4))*self.strength
                pos+=len(matrix)
                v[indices,k]=np.maximum(0,(v[indices,k]+floor)*np.exp(correction)-floor)
        out=[]
        for e,v in zip(episodes,values):
            v[:,3:]=np.minimum(1,v[:,3:])
            out.append(dict(e,values=np.where(e['available'],v,0).astype(np.float32),known_final=np.zeros_like(e['available'])))
            if 'filled' in e:
                out[-1]['filled']=e['filled'].copy()
                out[-1]['filled'][indices[:,None],getattr(self,'channels',[0,1,2])]=False
        return out

    def apply(self,e):
        return self.apply_batch([e])[0]

    def record(self):
        return dict(model=('32-wide residual MLP;200 fine-tuning updates' if self.neural else 'Histogram gradient boosting;100trees;7leaves;minleaf100'),pretrained=self.pretrained,weeks=self.weeks,
            penalty=self.penalty,strength=self.strength,features=self.features,channels=getattr(self,'channels',[0,1,2]),pathogen_inputs=getattr(self,'pathogen_inputs','all'),training_seasons=self.training_seasons,
            observed_donor_pairs_only=getattr(self,'observed_donor_pairs_only',False),training_cells=getattr(self,'training_cells',None),
            training_cells_by_age=getattr(self,'training_cells_by_age',None),
            features_description='Reported eight-week own trajectory, six-channel multiscale levels, availability, annual phase, location indicators, optional reported covariates; no final predictors')


class NeuralResidual:
    """Identical two-layer residual MLP for scratch and synthetic-history pretraining."""
    def __init__(self,features,seed):
        import torch
        self.seed=seed;torch.manual_seed(seed)
        self.net=torch.nn.Sequential(torch.nn.Linear(features,32),torch.nn.SiLU(),torch.nn.Linear(32,1))
    def fit(self,x,y):
        import torch
        x=np.nan_to_num(x).astype(np.float32)
        if not hasattr(self,'mean'):
            self.mean=x.mean(0);self.sd=np.maximum(x.std(0),.1)
        x=torch.tensor((x-self.mean)/self.sd);y=torch.tensor(y,dtype=torch.float32)
        rng=np.random.default_rng(self.seed)
        opt=torch.optim.Adam(self.net.parameters(),lr=.001,weight_decay=.001)
        for step in range(200):
            ids=rng.integers(len(y),size=256)
            opt.zero_grad();loss=(self.net(x[ids])[:,0]-y[ids]).square().mean()
            loss.backward();opt.step()
        return self
    def predict(self,x):
        import torch
        with torch.no_grad(): return self.net(torch.tensor((np.nan_to_num(x).astype(np.float32)-self.mean)/self.sd))[:,0].numpy()
