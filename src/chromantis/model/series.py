"""Small shared series quantile models and forecasting transfer from historical ILI."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from chromantis.evaluation.quantiles import LEVELS


def quantile_loss(q, y, mask):
    tau = q.new_tensor(LEVELS).reshape(-1, *([1] * y.ndim))
    error = y[None] - q
    return (2 * torch.maximum(tau * error, (tau - 1) * error).mean(0)) * mask


class SeriesModel(nn.Module):
    def __init__(self, lookback=12, width=64, horizons=(1,2,3,4), scale=None,
                 encoder='series_mlp', growth_anchor=False, **options):
        super().__init__()
        self.config = dict(options, lookback=lookback, width=width, horizons=list(horizons),
                           scale=scale, encoder=encoder, growth_anchor=growth_anchor,
                           direct_quantiles=True)
        self.register_buffer('scale', torch.tensor(scale if scale is not None else np.ones((6,52)), dtype=torch.float32))
        self.encoder = nn.Sequential(nn.Linear(lookback*2+3, width), nn.SiLU(), nn.Linear(width,width), nn.SiLU())
        self.time_mix = nn.Sequential(nn.Linear(lookback, lookback), nn.SiLU(), nn.Linear(lookback,lookback)) if encoder=='series_mixer' else None
        self.identity = nn.Embedding(7,width)
        self.adapter_down = nn.Embedding(7,width*4)
        self.adapter_up = nn.Embedding(7,4*width)
        nn.init.zeros_(self.adapter_up.weight)
        self.context = nn.Linear(width,width,bias=False)
        self.gate = nn.Parameter(torch.tensor(-4.))
        self.horizon = nn.Linear(1,width)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width,width), nn.SiLU(),nn.Linear(width,len(LEVELS)))
        self.head[-1].bias.data.fill_(-3.)
        self.head[-1].bias.data[11]=0.

    def series(self, values, available, calendar, sources, horizons=None):
        # [batch, history, channel, location]; independent scale from visible history.
        n,p,c,l=values.shape
        scale=torch.where(available,values,0).amax(1).clamp(min=1e-4)
        z=torch.log1p(torch.where(available,values,0)/scale[:,None])
        x=z.permute(0,2,3,1)
        if self.time_mix is not None: x=x+self.time_mix(x)
        mask=available.permute(0,2,3,1).to(x.dtype)
        cal=calendar[:,None,None].expand(n,c,l,3)
        h=self.encoder(torch.cat((x,mask,cal),-1))
        source=torch.as_tensor(sources,device=values.device)
        down=self.adapter_down(source).reshape(c,self.config['width'],4)
        up=self.adapter_up(source).reshape(c,4,self.config['width'])
        h=h+torch.einsum('nclr,crw->nclw',F.silu(torch.einsum('nclw,cwr->nclr',h,down)),up)
        # Shared context, no cross-location exchange. Historical single-series ILI has no donors.
        observed=available.any(1)[...,None]
        context=(h*observed).sum(1,keepdim=True)/observed.sum(1,keepdim=True).clamp(min=1)
        if c>1: h=h+torch.sigmoid(self.gate)*self.context(context)
        h=h+self.identity(source)[None,:,None]
        hs=self.config['horizons'] if horizons is None else horizons
        h=h[:,None]+self.horizon(values.new_tensor(hs)[:,None]/4)[None,:,None,None]
        raw=self.head(h)
        indices=(available*torch.arange(1,p+1,device=values.device)[None,:,None,None]).argmax(1)
        anchor=z.gather(1,indices[:,None]).squeeze(1)
        anchor=anchor[:,None].expand(n,len(hs),c,l)
        if self.config['growth_anchor'] and p>=3:
            valid=available[:,-1]&available[:,-3]
            slope=(z[:,-1]-z[:,-3])/2*valid
            damping=values.new_tensor([sum(.5**i for i in range(max(0,t))) for t in hs])
            anchor=anchor+slope[:,None]*damping[None,:,None,None]
        median=anchor+raw[...,11]*.2
        lower=median[...,None]-torch.flip(torch.cumsum(F.softplus(torch.flip(raw[...,:11],[-1]))*.1,-1),[-1])
        upper=median[...,None]+torch.cumsum(F.softplus(raw[...,12:])*.1,-1)
        latent=torch.cat((lower,median[...,None],upper),-1).clamp(0,12)
        q=torch.expm1(latent)*scale[:,None,:,:,None]
        return q.permute(4,0,1,2,3)

    def forward(self, values, available, calendar, **kwargs):
        channels={'all': range(6),'flu':[0,3],'flu_hosp':[0],'flu_ed':[3],'flu_covid':[0,1,3,4],'flu_rsv':[0,2,3,5]}[self.config.get('pathogen_inputs','all')]
        keep=values.new_tensor([c in channels for c in range(6)],dtype=torch.bool)[None,None,:,None]
        available=available & keep
        values=torch.where(available,values,0)
        q=self.series(values,available,calendar,list(range(6)))
        return torch.cat((q[:,:,:,:3],q[:,:,:,3:].clamp(max=1)),3)


class HistoricalILI:
    def __init__(self,path,lookback,device,units='own',steps=200):
        self.units=units;self.steps=steps
        from chromantis.dataset.episodes import calendar
        with np.load(path) as d:
            data=d['values'];dates=d['dates'].astype(str)
            assert max(dates) < '2022-08-01'
        self.historical_q95=float(np.nanquantile(data,.95))
        windows=[];targets=[];cal=[]
        for i in range(lookback-1,len(data)-4):
            x=data[i-lookback+1:i+1];y=data[i+1:i+5]
            ok=np.isfinite(x).sum(0)>=lookback//2
            ok &= np.isfinite(y).all(0)
            for j in np.flatnonzero(ok):
                windows.append(x[:,j]);targets.append(y[:,j]);cal.append(dates[i])
        if not windows: raise ValueError('No historical ILI forecasting windows')
        self.x=torch.tensor(np.nan_to_num(windows),device=device,dtype=torch.float32)[:,:,None,None]
        self.mask=torch.tensor(np.isfinite(windows),device=device)[:,:,None,None]
        self.y=torch.tensor(np.array(targets),device=device,dtype=torch.float32)[:,:,None,None]
        self.cal=torch.tensor(calendar(cal),device=device)
        self.count=len(windows)

    def loss(self,model):
        ids=torch.randint(self.count,(64,),device=self.x.device)
        x,y=self.x[ids],self.y[ids]
        if self.units == 'flu_scaled':
            # Auxiliary pseudo-task: transfer ILI dynamics into modern flu source heads.
            # Modern scales come only from the current fit partition's permitted truth.
            source=0 if torch.rand((),device=x.device)<.5 else 3
            factor=model.scale[source].mean()/max(self.historical_q95,1e-6)
            x,y=x*factor,y*factor
            if source==3:x,y=x.clamp(max=1),y.clamp(max=1)
            normalizer=model.scale[source].mean().clamp(min=1e-6)
        else:
            source=6;normalizer=x.amax(1).clamp(min=.001)[:,None]
        q=model.series(x,self.mask[ids],self.cal[ids],[source],horizons=[1,2,3,4])
        return (quantile_loss(q,y,torch.ones_like(y))/normalizer).mean()

    def pretrain(self,model,seed):
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        for step in range(self.steps):
            optimizer.zero_grad();loss=self.loss(model);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        print(f'Historical ILI forecasting pretraining: {self.steps} updates, {self.count} windows, units={self.units}',flush=True)
