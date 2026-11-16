"""Compact radar-model experiments, not the authors' pretrained checkpoints.

DGMR components are adapted from DeepMind's CC-BY-4.0 pseudocode, archived in
data/alta_nowcast/model_sources/dgmr_pseudocode. Attribution: Ravuri et al.
(2021), https://doi.org/10.1038/s41586-021-03854-z. Changes include reduced
width, input validity channels, 10-minute cadence, and a three-hour horizon.
NowcastNet is an independent compact implementation of the mechanisms in
Zhang et al. (2023), https://doi.org/10.1038/s41586-023-06184-4, not a port of
the inaccessible Code Ocean checkpoint. Every result must retain these labels.

Public API consumes physical radar precipitation rates (mm/hour), shape
[sample,time,height,width], or [sample,time,2,height,width] containing rate and
causally available radar quality/validity in [0,1]. NaNs indicate missing radar.
The networks operate in log1p(rate); transport in the evolution network uses
linear rate. Returned ensembles are in physical mm/hour, shape [N,E,L,H,W].
"""
from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


MODEL_LABELS = {
    "convlstm": "ConvLSTM (compact)",
    "dgmr": "DGMR adaptation (compact; 3-hour extension)",
    "nowcastnet": "NowcastNet adaptation (compact)",
}


@dataclass
class RadarConfig:
    name: str
    input_frames: int = 9
    output_frames: int = 18
    base_channels: int = 8
    cadence_minutes: int = 10
    rate_cap_mm_h: float = 128.0
    seed: int = 42
    learning_rate: float = 3e-4
    batch_size: int = 4
    steps: int = 400
    ensemble_training_members: int = 2
    threads: int = 2
    recipe: str = "uniform_l1"
    linear_rate_head: bool = False


def conv(cin, cout, kernel=3, stride=1, spectral=False):
    layer = nn.Conv2d(cin, cout, kernel, stride, kernel // 2)
    return spectral_norm(layer) if spectral else layer


class ResidualBlock(nn.Module):
    def __init__(self, cin, cout, *, down=False, up=False, spectral=False):
        super().__init__()
        self.a = conv(cin, cout, spectral=spectral)
        self.b = conv(cout, cout, spectral=spectral)
        self.skip = conv(cin, cout, 1, spectral=spectral) if cin != cout else nn.Identity()
        self.down, self.up = down, up

    def forward(self, x):
        if self.up:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        y = self.b(F.leaky_relu(self.a(F.leaky_relu(x, .2)), .2)) + self.skip(x)
        return F.avg_pool2d(y, 2) if self.down else y


class ConvLSTMCell(nn.Module):
    def __init__(self, cin, hidden):
        super().__init__()
        self.hidden = hidden
        self.gates = conv(cin + hidden, 4 * hidden)

    def forward(self, x, state=None):
        if state is None:
            h = x.new_zeros(x.shape[0], self.hidden, *x.shape[-2:])
            c = torch.zeros_like(h)
        else:
            h, c = state
        i, f, o, g = self.gates(torch.cat([x, h], 1)).chunk(4, 1)
        c = torch.sigmoid(f + 1) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c


class ConvLSTMNowcaster(nn.Module):
    """Two-scale encoder/forecaster with locked dropout per trajectory."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        c = cfg.base_channels
        self.encoder = conv(2, c, stride=2)
        self.cell1 = ConvLSTMCell(c, 2*c)
        self.cell2 = ConvLSTMCell(2*c, 4*c)
        self.decode = nn.Sequential(conv(6*c, 2*c), nn.SiLU(), conv(2*c, 4, 1))
        nn.init.constant_(self.decode[-1].bias, -3.)

    def forward(self, x, stochastic=True):
        s1 = s2 = None
        # One channel mask is shared through the trajectory, preserving temporal
        # dependence. MC dropout is not a calibrated predictive distribution.
        c = self.cfg.base_channels
        dropout = (self.training or stochastic)
        mask = x.new_ones(x.shape[0], c, 1, 1)
        if dropout:
            mask = torch.bernoulli(mask * .9) / .9
        for frame in x.unbind(1):
            z = F.silu(self.encoder(frame)) * mask
            s1 = self.cell1(z, s1)
            s2 = self.cell2(F.avg_pool2d(s1[0], 2), s2)
        previous = x[:, -1, :1]
        validity = x[:, -1, 1:2]
        out = []
        for _ in range(self.cfg.output_frames):
            z = F.silu(self.encoder(torch.cat([previous, validity], 1))) * mask
            s1 = self.cell1(z, s1)
            s2 = self.cell2(F.avg_pool2d(s1[0], 2), s2)
            h = torch.cat([s1[0], F.interpolate(s2[0], size=s1[0].shape[-2:])], 1)
            previous = F.softplus(F.pixel_shuffle(self.decode(h), 2))
            out.append(previous[:, 0])
        return torch.stack(out, 1)


class ConvGRUCell(nn.Module):
    def __init__(self, cin, hidden):
        super().__init__()
        self.read = conv(cin+hidden, hidden, spectral=True)
        self.update = conv(cin+hidden, hidden, spectral=True)
        self.candidate = conv(cin+hidden, hidden, spectral=True)

    def forward(self, x, h):
        joined = torch.cat([x, h], 1)
        r, u = torch.sigmoid(self.read(joined)), torch.sigmoid(self.update(joined))
        candidate = F.relu(self.candidate(torch.cat([x, r*h], 1)))
        return u*h + (1-u)*candidate


class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        q = max(channels//8, 1)
        self.q, self.k, self.v = conv(channels,q,1), conv(channels,q,1), conv(channels,channels,1)
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        b,c,h,w = x.shape
        q = self.q(x).flatten(2).transpose(1,2)
        k = self.k(x).flatten(2)
        a = torch.softmax(q @ k / math.sqrt(k.shape[1]), -1)
        v = self.v(x).flatten(2) @ a.transpose(1,2)
        return x + self.gamma*v.reshape(b,c,h,w)


class CompactDGMR(nn.Module):
    """Four-scale context, latent stack and recurrent stochastic sampler."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        c = cfg.base_channels
        widths = [c, 2*c, 4*c, 8*c]
        self.blocks = nn.ModuleList([
            ResidualBlock(a,b,down=True,spectral=True)
            for a,b in zip([8]+widths[:-1], widths)])
        self.mix = nn.ModuleList([conv(w*cfg.input_frames,w,spectral=True) for w in widths])
        self.latent = nn.Sequential(
            ResidualBlock(c,4*c,spectral=True), SpatialAttention(4*c),
            ResidualBlock(4*c,16*c,spectral=True))
        self.grus = nn.ModuleList([ConvGRUCell(2*w,w) for w in reversed(widths)])
        self.refine = nn.ModuleList([ResidualBlock(w,2*w,spectral=True) for w in reversed(widths)])
        self.ups = nn.ModuleList([ResidualBlock(2*w,w,up=True,spectral=True) for w in reversed(widths)])
        self.output = conv(c,4,1)
        nn.init.constant_(self.output.bias, -3.)

    def forward(self, x, stochastic=True):
        b,t,ch,h,w = x.shape
        z = F.pixel_unshuffle(x.reshape(b*t,ch,h,w), 2)
        context = []
        for block,mix in zip(self.blocks,self.mix):
            z = block(z)
            zz = z.reshape(b,t*z.shape[1],*z.shape[-2:])
            context.append(F.relu(mix(zz)))
        shape = (b,self.cfg.base_channels,*context[-1].shape[-2:])
        noise = torch.randn(shape,device=x.device) if stochastic else x.new_zeros(shape)
        latent = self.latent(noise)
        states = list(reversed(context))
        out = []
        for _ in range(self.cfg.output_frames):
            z = latent
            for i,(gru,block,up) in enumerate(zip(self.grus,self.refine,self.ups)):
                states[i] = gru(z, states[i])
                z = up(block(states[i]))
            positive=F.softplus(F.pixel_shuffle(self.output(F.relu(z)),2))[:,0]
            # Corrected DGMR variant parameterizes physical rain directly.
            # Older checkpoints keep their original log-rate head unchanged.
            out.append(torch.log1p(positive) if self.cfg.linear_rate_head else positive)
        return torch.stack(out,1)


class UNet(nn.Module):
    def __init__(self, cin, cout, c):
        super().__init__()
        self.stem = ResidualBlock(cin,c)
        self.d1 = ResidualBlock(c,2*c,down=True)
        self.d2 = ResidualBlock(2*c,4*c,down=True)
        self.d3 = ResidualBlock(4*c,8*c,down=True)
        self.u2 = ResidualBlock(12*c,4*c)
        self.u1 = ResidualBlock(6*c,2*c)
        self.u0 = ResidualBlock(3*c,c)
        self.head = conv(c,cout,1)
        nn.init.normal_(self.head.weight,0,.001)
        nn.init.zeros_(self.head.bias)

    def forward(self,x):
        a=self.stem(x); b=self.d1(a); c=self.d2(b); d=self.d3(c)
        c=self.u2(torch.cat([F.interpolate(d,size=c.shape[-2:]),c],1))
        b=self.u1(torch.cat([F.interpolate(c,size=b.shape[-2:]),b],1))
        a=self.u0(torch.cat([F.interpolate(b,size=a.shape[-2:]),a],1))
        return self.head(a)


def warp_rate(rate, flow):
    """Backwards semi-Lagrangian bilinear sampling; flow is pixels/step.

    Border padding is an explicit pilot boundary condition. Metrics should
    use the interior, and sufficient upstream context is still necessary.
    """
    b,c,h,w = rate.shape
    yy,xx = torch.meshgrid(torch.linspace(-1,1,h,device=rate.device),
                          torch.linspace(-1,1,w,device=rate.device),indexing="ij")
    grid = torch.stack([xx,yy],-1)[None].expand(b,-1,-1,-1)
    scaled = flow.permute(0,2,3,1)*flow.new_tensor([2/(w-1),2/(h-1)])
    return F.grid_sample(rate,grid-scaled,mode="bilinear",padding_mode="border",align_corners=True)


class EvolutionNetwork(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        self.unet=UNet(2*cfg.input_frames,3*cfg.output_frames,cfg.base_channels)

    def forward(self,x):
        b,t,c,h,w=x.shape
        fields=self.unet(x.reshape(b,t*c,h,w)).reshape(b,self.cfg.output_frames,3,h,w)
        flow=4*torch.tanh(fields[:,:,:2])
        growth=2*torch.tanh(fields[:,:,2:3])
        rate=torch.expm1(x[:,-1,:1]).clamp_min(0)
        out=[]
        for j in range(self.cfg.output_frames):
            rate=F.relu(warp_rate(rate,flow[:,j])+growth[:,j])
            out.append(torch.log1p(rate[:,0]))
        return torch.stack(out,1),flow


class SPADE(nn.Module):
    def __init__(self,channels,condition_channels):
        super().__init__()
        self.norm=nn.GroupNorm(1,channels,affine=False)
        self.shared=conv(condition_channels,channels)
        self.gamma=conv(channels,channels)
        self.beta=conv(channels,channels)

    def forward(self,x,condition):
        z=F.relu(self.shared(F.interpolate(condition,size=x.shape[-2:])))
        return self.norm(x)*(1+self.gamma(z))+self.beta(z)


class SPADEBlock(nn.Module):
    def __init__(self,cin,cout,cond):
        super().__init__()
        self.n1=SPADE(cin,cond);self.n2=SPADE(cout,cond)
        self.a=conv(cin,cout);self.b=conv(cout,cout)
        self.skip=conv(cin,cout,1)

    def forward(self,x,cond):
        y=self.a(F.leaky_relu(self.n1(x,cond),.2))
        return self.b(F.leaky_relu(self.n2(y,cond),.2))+self.skip(x)


class CompactNowcastNet(nn.Module):
    """Decoupled evolution network + noise-conditioned SPADE generator."""
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg;c=cfg.base_channels
        self.evolution=EvolutionNetwork(cfg)
        cond=2*cfg.input_frames+cfg.output_frames
        self.encoder=nn.Sequential(ResidualBlock(cond,2*c,down=True),
                                   ResidualBlock(2*c,4*c,down=True),
                                   ResidualBlock(4*c,8*c,down=True))
        self.noise=nn.Sequential(conv(c,4*c),nn.SiLU(),conv(4*c,8*c))
        self.spade=nn.ModuleList([SPADEBlock(16*c,8*c,cond),
                                  SPADEBlock(8*c,4*c,cond),SPADEBlock(4*c,2*c,cond)])
        self.head=conv(2*c,cfg.output_frames,1)
        nn.init.normal_(self.head.weight,0,.001);nn.init.zeros_(self.head.bias)

    def forward(self,x,stochastic=True,detach_evolution=True):
        evo,_=self.evolution(x)
        if detach_evolution: evo=evo.detach()
        b,t,c,h,w=x.shape
        cond=torch.cat([x.reshape(b,t*c,h,w),evo],1)
        z=self.encoder(cond)
        noise=torch.randn((b,self.cfg.base_channels,*z.shape[-2:]),device=x.device) if stochastic else x.new_zeros((b,self.cfg.base_channels,*z.shape[-2:]))
        z=torch.cat([z,self.noise(noise)],1)
        for block in self.spade:
            z=block(z,cond)
            z=F.interpolate(z,scale_factor=2,mode="nearest")
        residual=3*torch.tanh(self.head(F.leaky_relu(z,.2)))
        # Multiplicative stochastic correction in linear rate, represented
        # stably as softplus(log(rate)+residual). One latent field is shared
        # across all lead times; future pixels are not independent draws.
        return F.softplus(torch.log(torch.expm1(evo).clamp_min(.001))+residual)


class RadarDiscriminator(nn.Module):
    """Separate spatial and temporal branches for DGMR; temporal for NCN."""
    def __init__(self,c=8):
        super().__init__()
        self.spatial=nn.Sequential(
            ResidualBlock(1,c,down=True,spectral=True),
            ResidualBlock(c,2*c,down=True,spectral=True),
            ResidualBlock(2*c,4*c,down=True,spectral=True),nn.LeakyReLU(.2),
            nn.AdaptiveAvgPool2d(1),nn.Flatten(),spectral_norm(nn.Linear(4*c,1)))
        self.temporal=nn.Sequential(
            spectral_norm(nn.Conv3d(1,c,3,stride=(1,2,2),padding=1)),nn.LeakyReLU(.2),
            spectral_norm(nn.Conv3d(c,2*c,3,stride=2,padding=1)),nn.LeakyReLU(.2),
            spectral_norm(nn.Conv3d(2*c,4*c,3,stride=2,padding=1)),nn.LeakyReLU(.2),
            nn.AdaptiveAvgPool3d(1),nn.Flatten(),spectral_norm(nn.Linear(4*c,1)))

    def forward(self,x,future,spatial=True):
        complete=torch.cat([x[:,:,0],future],1)
        temporal=self.temporal(complete[:,None])[:,0]
        if not spatial: return temporal
        # Four evenly spaced future frames make scoring deterministic while
        # preserving the original separate spatial/temporal adversarial idea.
        ix=torch.linspace(0,future.shape[1]-1,4,device=x.device).long()
        frames=future[:,ix].reshape(-1,1,*future.shape[-2:])
        score=self.spatial(frames).reshape(x.shape[0],4).mean(1)
        return temporal+score


def make_model(cfg):
    if cfg.name=="convlstm": return ConvLSTMNowcaster(cfg)
    if cfg.name=="dgmr": return CompactDGMR(cfg)
    if cfg.name=="nowcastnet": return CompactNowcastNet(cfg)
    raise ValueError(f"Unknown model {cfg.name}")


def encode_inputs(x,cap=128.):
    a=np.asarray(x,dtype=np.float32)
    if a.ndim==5 and a.shape[2]==2:
        rate,quality=a[:,:,0],a[:,:,1]
    elif a.ndim==4:
        rate=a;quality=np.ones_like(a)
    else:
        raise ValueError("Input must be [N,T,H,W] or [N,T,2,H,W] (rate,quality)")
    valid=np.isfinite(rate)&(rate>=0)
    quality=np.where(np.isfinite(quality),np.clip(quality,0,1),0)*valid
    rate=np.where(valid,np.clip(rate,0,cap),0)
    return torch.from_numpy(np.stack([np.log1p(rate),quality.astype(np.float32)],2))


def encode_targets(y,cap=128.):
    a=np.asarray(y,dtype=np.float32)
    valid=np.isfinite(a)&(a>=0)
    return (torch.from_numpy(np.log1p(np.where(valid,np.clip(a,0,cap),0))),
            torch.from_numpy(valid.astype(np.float32)))


def weighted_loss(pred,target,mask):
    w=(1+target.clamp(max=4))*mask
    return ((pred-target).abs()*w).sum()/w.sum().clamp_min(1)


def physical_mean_mse(samples,target,mask):
    """Physical ensemble-mean MSE plus a small log-space stabilizer."""
    rates=torch.stack([torch.expm1(s.clamp(max=math.log1p(128))) for s in samples]).mean(0)
    truth=torch.expm1(target)
    physical=(((rates-truth)**2)*mask).sum()/mask.sum().clamp_min(1)
    log_error=(((torch.log1p(rates)-target)**2)*mask).sum()/mask.sum().clamp_min(1)
    return physical+.1*log_error


def ensemble_pool_loss(samples,target,mask):
    # Max-pool regularization is a defining NowcastNet component. Pool only
    # fully observed neighborhoods to prevent missing radar acting as dry.
    shape=target.shape
    yt=F.max_pool2d(target.flatten(0,1)[:,None],5,2,2)
    vm=1-F.max_pool2d((1-mask).flatten(0,1)[:,None],5,2,2)
    ps=[F.max_pool2d(p.flatten(0,1)[:,None],5,2,2) for p in samples]
    return weighted_loss(torch.stack(ps).mean(0),yt,vm)


def architecture_manifest(cfg,model):
    specific={
        "convlstm":["two spatial scales and trajectory-locked MC dropout"],
        "dgmr":["nine input frames instead of four; ten-minute rather than five-minute cadence",
                "forecast horizon extended beyond the original ninety minutes",
                "reduced adversarial loss weight; two training ensemble members by default"],
        "nowcastnet":["compact U-Net/SPADE and simplified temporal discriminator",
                      "group normalization in place of original normalization details",
                      "reduced adversarial loss weight; two training ensemble members by default"],
    }
    return {
        "label":MODEL_LABELS[cfg.name],"author_checkpoint":False,
        "parameters":sum(p.numel() for p in model.parameters()),
        "config":asdict(cfg),"physical_units":"mm/hour liquid-equivalent radar precipitation",
        "network_transform":"log1p(rate); cap128mm/hour; explicit causal input quality/validity channel",
        "changes_from_paper":[
            "reduced channel width and local training domain",
            f"{cfg.cadence_minutes}-minute cadence, {cfg.input_frames} input frames and {cfg.output_frames} output frames",
            "log1p losses; fixed pilot optimization schedule",
            "no pretrained author weights or full published skill reproduction",
        ]+specific[cfg.name],
        "sources":[
            "https://proceedings.neurips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf",
            "https://github.com/google-deepmind/deepmind-research/tree/master/nowcasting",
            "https://storage.googleapis.com/dm-nowcasting-example-data/pseudocode.zip",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC10356617/",
        ],
    }


@torch.no_grad()
def _validation_score(model,x,y,mask,batch_size):
    model.eval();sq=den=0.
    # Fixed validation noise avoids choosing a checkpoint from random luck.
    with torch.random.fork_rng():
        torch.manual_seed(98123)
        for start in range(0,len(x),batch_size):
            bx=x[start:start+batch_size];by=y[start:start+batch_size];bm=mask[start:start+batch_size]
            p=torch.stack([torch.expm1(model(bx)).clamp(0,128) for _ in range(2)]).mean(0)
            sq+=float((((p-torch.expm1(by))**2)*bm).sum())
            den+=float(bm.sum())
    return sq/max(den,1)


def train_model(name,x,y,*,validation=None,steps=400,batch_size=4,base_channels=8,
                seed=42,learning_rate=3e-4,checkpoint_path=None,threads=2,
                ensemble_training_members=2,log_every=25,recipe="uniform_l1",linear_rate_head=False):
    """Train only the supplied training split; select checkpoint on validation.

    Callers must split by complete storms/winters before making windows. This
    routine never sees test targets and does not select dates or hyperparameters.
    For NowcastNet, the first max(steps//3,1) steps train evolution; remaining
    steps train the stochastic generator with evolution frozen.
    """
    torch.set_num_threads(threads);torch.manual_seed(seed);np.random.seed(seed)
    cfg=RadarConfig(name=name,input_frames=x.shape[1],output_frames=y.shape[1],
        base_channels=base_channels,steps=steps,batch_size=batch_size,seed=seed,
        learning_rate=learning_rate,threads=threads,
        ensemble_training_members=ensemble_training_members,recipe=recipe,linear_rate_head=linear_rate_head)
    xx=encode_inputs(x);yy,mm=encode_targets(y)
    if len(xx)!=len(yy) or len(xx)==0: raise ValueError("Nonempty matched input/target samples required")
    if x.shape[-1]%32 or x.shape[-2]%32: raise ValueError("Grid sides must be multiples of32")
    if mm.sum()==0: raise ValueError("No observed training target pixels")
    model=make_model(cfg)
    discriminator=None if name=="convlstm" else RadarDiscriminator(base_channels)
    opt=torch.optim.Adam(model.parameters(),lr=learning_rate,betas=(.5,.999))
    dopt=None if discriminator is None else torch.optim.Adam(discriminator.parameters(),lr=learning_rate,betas=(.5,.999))
    vv=None
    if validation is not None:
        vx,vy=validation;vyt,vmt=encode_targets(vy);vv=(encode_inputs(vx),vyt,vmt)
    if recipe not in {"uniform_l1","wet_mse"}:raise ValueError("Unknown training recipe")
    probabilities=None
    sampling_audit={"recipe":recipe,"training_cases":len(xx),"importance_sampling":False}
    if recipe=="wet_mse":
        valid=mm.numpy().astype(bool)
        wet=((np.expm1(yy.numpy())>.1)&valid).sum(axis=(1,2,3))/np.maximum(valid.sum(axis=(1,2,3)),1)
        selected=wet>=.1
        fallback=False
        if not selected.any():selected=wet>0;fallback=True
        if not selected.any():raise ValueError("No wet training radar pixels for wet_mse recipe")
        probabilities=np.full(len(xx),.3/len(xx),dtype=np.float64)
        probabilities[selected]+=.7/selected.sum()
        probabilities/=probabilities.sum()
        sampling_audit.update({"importance_sampling":True,"wet_case_count":int(selected.sum()),
            "fallback_any_wet_case":fallback,"wet_case_definition":"at least10percent validtargetpixels above0.1mm/hour",
            "sampling_probability":probabilities.tolist(),"wet_pixel_fraction":wet.tolist(),
            "formula":"0.3uniform +0.7uniform among wettrainingcases",
            "main_loss":"maskedphysicalensemblemeanMSE +0.1maskedlog1pMSE",
            "loss_weights":{"convlstm":"main","dgmr":"20main+0.1hingeGAN",
                "nowcastnet_evolution":"main+0.01flowsmoothness",
                "nowcastnet_generator":"20main+5maxpoolregularizer+0.1logisticGAN"}})
    rng=np.random.default_rng(seed);logs=[];best=math.inf;best_state=None;best_step=None
    begin=time.monotonic();warm=max(steps//3,1) if name=="nowcastnet" else 0
    for step in range(1,steps+1):
        ix=rng.choice(len(xx),min(batch_size,len(xx)),replace=False,p=probabilities)
        bx,by,bm=xx[ix],yy[ix],mm[ix]
        model.train();dloss_value=0.;phase="generator"
        if name=="nowcastnet" and step<=warm:
            phase="evolution"
            evo,flow=model.evolution(bx)
            smooth=(flow[...,1:,:]-flow[...,:-1,:]).abs().mean()+(flow[...,:,1:]-flow[...,:,:-1]).abs().mean()
            loss=(physical_mean_mse([evo],by,bm) if recipe=="wet_mse" else weighted_loss(evo,by,bm))+.01*smooth
        elif name=="convlstm":
            pred=model(bx)
            loss=physical_mean_mse([pred],by,bm) if recipe=="wet_mse" else weighted_loss(pred,by,bm)
        else:
            if name=="nowcastnet":
                for p in model.evolution.parameters(): p.requires_grad_(False)
            discriminator.train()
            for p in discriminator.parameters(): p.requires_grad_(True)
            with torch.no_grad(): fake=model(bx)*bm
            real_score=discriminator(bx,by*bm,spatial=name=="dgmr")
            fake_score=discriminator(bx,fake,spatial=name=="dgmr")
            if name=="dgmr":
                dl=F.relu(1-real_score).mean()+F.relu(1+fake_score).mean()
            else:
                dl=F.softplus(-real_score).mean()+F.softplus(fake_score).mean()
            dopt.zero_grad();dl.backward();torch.nn.utils.clip_grad_norm_(discriminator.parameters(),5);dopt.step()
            dloss_value=float(dl.detach())
            for p in discriminator.parameters(): p.requires_grad_(False)
            samples=[model(bx) for _ in range(ensemble_training_members)]
            score=discriminator(bx,samples[0]*bm,spatial=name=="dgmr")
            if name=="dgmr":
                reconstruction=physical_mean_mse(samples,by,bm) if recipe=="wet_mse" else weighted_loss(torch.stack(samples).mean(0),by,bm)
                loss=20*reconstruction-.1*score.mean()
            else:
                loss=20*ensemble_pool_loss(samples,by,bm)+.1*F.softplus(-score).mean()
                if recipe=="wet_mse":
                    loss=20*physical_mean_mse(samples,by,bm)+5*ensemble_pool_loss(samples,by,bm)+.1*F.softplus(-score).mean()
        if not torch.isfinite(loss): raise FloatingPointError(f"Nonfinite {name} loss at step{step}")
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        if step%log_every==0 or step==steps or step==1 or step==warm:
            row={"step":step,"phase":phase,"loss":float(loss.detach()),
                 "discriminator_loss":dloss_value,"elapsed_seconds":time.monotonic()-begin}
            if vv is not None and (step%log_every==0 or step==steps) and (name!="nowcastnet" or step>warm):
                score=_validation_score(model,*vv,batch_size)
                row["validation_mse_mm_h_squared"]=score
                if score<best:
                    best=score;best_state=copy.deepcopy(model.state_dict());best_step=step
                    if checkpoint_path:
                        provisional={"model":model,"config":cfg,
                            "manifest":architecture_manifest(cfg,model),
                            "training_log":logs+[row],"best_step":best_step,
                            "training_samples":len(xx),"validation_samples":len(vv[0]),
                            "elapsed_seconds":time.monotonic()-begin,
                            "training_in_progress":True,"sampling_audit":sampling_audit}
                        Path(checkpoint_path).parent.mkdir(parents=True,exist_ok=True)
                        save_bundle(provisional,checkpoint_path)
            logs.append(row)
            print(json.dumps({"model":name,**row}),flush=True)
            if checkpoint_path:
                Path(checkpoint_path).parent.mkdir(parents=True,exist_ok=True)
                Path(str(checkpoint_path)+".training.json").write_text(json.dumps(logs,indent=2))
    if best_state is not None: model.load_state_dict(best_state)
    model.eval()
    bundle={"model":model,"config":cfg,"manifest":architecture_manifest(cfg,model),
            "training_log":logs,"best_step":best_step or steps,
            "training_samples":len(xx),"validation_samples":0 if vv is None else len(vv[0]),
            "elapsed_seconds":time.monotonic()-begin}
    bundle["sampling_audit"]=sampling_audit
    if checkpoint_path: save_bundle(bundle,checkpoint_path)
    return bundle


def save_bundle(bundle,path):
    state={k:v for k,v in bundle.items() if k not in {"model","config"}}
    state["config"]=asdict(bundle["config"]);state["state_dict"]=bundle["model"].state_dict()
    torch.save(state,path)
    Path(str(path)+".json").write_text(json.dumps({k:v for k,v in state.items() if k!="state_dict"},indent=2))


def load_bundle(path):
    state=torch.load(path,map_location="cpu",weights_only=True)
    cfg=RadarConfig(**state.pop("config"));model=make_model(cfg)
    model.load_state_dict(state.pop("state_dict"));model.eval()
    return {**state,"model":model,"config":cfg}


@torch.no_grad()
def predict_ensemble(bundle,x,members=8,batch_size=4,seed=12345):
    model=bundle["model"];model.eval();cfg=bundle["config"]
    torch.set_num_threads(cfg.threads)
    xx=encode_inputs(x,cfg.rate_cap_mm_h);chunks=[]
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for start in range(0,len(xx),batch_size):
            b=xx[start:start+batch_size]
            samples=[torch.expm1(model(b)).clamp(0,cfg.rate_cap_mm_h).cpu().numpy() for _ in range(members)]
            chunks.append(np.stack(samples,1))
    return np.concatenate(chunks,0)
