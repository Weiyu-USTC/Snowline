"""Compact weather-conditioned neural distributions for two snowfall days.

The temporal encoder consumes 48 hourly *issued weather forecasts*. It predicts
two full 24-hour amounts, not independently supervised hourly snowfall. The
optional radar branch consumes observations available before forecast issuance.

Each day has a zero-inflated Gamma distribution: occurrence probability p and
Gamma positive amount with concentration a and rate b. Expected snowfall is
p*a/b. Joint paths use independent days unless the caller supplies an explicitly
estimated dependence model; marginal uncertainty alone does not establish the
dependence of storm snowfall across days.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.special import gammaincinv, ndtr
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Neural48Config:
    nwp_features: int
    static_features: int = 0
    local_features: int = 16
    width: int = 32
    dropout: float = .15
    variant: str = 'weather_radar'
    number_of_sites: int = 4
    site_embedding: int = 8
    radar_channels: int = 5
    radar_width: int = 32
    local_width: int = 16
    head_width: int = 48
    positive_mean_floor: float = .02
    positive_mean_ceiling: float = 100.
    concentration_floor: float = .2
    concentration_ceiling: float = 50.

    def __post_init__(self):
        if self.variant not in {'weather_only', 'weather_radar'}:
            raise ValueError('variant must be weather_only or weather_radar')
        if self.nwp_features < 1 or self.static_features < 0 or self.local_features < 0:
            raise ValueError('Invalid feature dimensions')
        if min(self.width, self.number_of_sites, self.site_embedding,
               self.radar_width, self.head_width) < 1:
            raise ValueError('Network widths and site count must be positive')
        if self.radar_channels < 5:
            raise ValueError('Radar encoding requires rate, quality and three mask/coverage channels')
        if not 0 <= self.dropout < 1:
            raise ValueError('dropout must lie in [0,1)')
        if not 0 < self.positive_mean_floor < self.positive_mean_ceiling:
            raise ValueError('Invalid positive-amount mean bounds')
        if not 0 < self.concentration_floor < self.concentration_ceiling:
            raise ValueError('Invalid Gamma concentration bounds')


class ChannelLayerNorm(nn.Module):
    """Normalize channels within one forecast hour, without mixing lead times."""
    def __init__(self, width):
        super().__init__(); self.norm = nn.LayerNorm(width)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class CausalConv(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(width, width, 3, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.padding, 0)))


class TemporalBlock(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.norm1 = ChannelLayerNorm(width); self.norm2 = ChannelLayerNorm(width)
        self.conv1 = CausalConv(width, dilation); self.conv2 = CausalConv(width, dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.dropout(self.conv1(F.gelu(self.norm1(x))))
        h = self.dropout(self.conv2(F.gelu(self.norm2(h))))
        return x + h


class RadarEncoder(nn.Module):
    """Per-frame spatial features followed by causal context recurrence."""
    def __init__(self, config):
        super().__init__(); self.config = config
        self.spatial = nn.Sequential(
            nn.Conv2d(config.radar_channels, 16, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
            nn.Linear(128, config.radar_width), nn.GELU())
        self.temporal = nn.GRU(config.radar_width, config.radar_width, batch_first=True)
        self.local = (nn.GRU(config.local_features, config.local_width, batch_first=True)
                      if config.local_features else None)

    @property
    def output_features(self):
        return self.config.radar_width + (self.config.local_width if self.local else 0) + 4

    def forward(self, radar, local_stats):
        batch, time, channels, height, width = radar.shape
        frames = self.spatial(radar.reshape(batch*time, channels, height, width))
        _, state = self.temporal(frames.reshape(batch, time, -1))
        # Channels2–4 explicitly distinguish missing inputs from dry weather.
        # Preserve raw quality even where the rate is not accepted. A fully
        # missing context cannot create a learned synthetic radar signal.
        quality = radar[:, :, 1].mean((1, 2, 3))
        qualified = radar[:, :, 2].mean((1, 2, 3))
        source_valid = radar[:, :, 3].mean((1, 2, 3))
        quality_available = radar[:, :, 4].mean((1, 2, 3))
        available = (qualified > 0).to(radar.dtype).unsqueeze(-1)
        parts = [state[-1] * available]
        if self.local:
            _, local_state = self.local(local_stats)
            parts.append(local_state[-1] * available)
        parts.append(torch.stack([quality, qualified, source_valid, quality_available], dim=-1))
        return torch.cat(parts, dim=-1)


class Neural48Model(nn.Module):
    """Four-block residual TCN, optional radar CNN–GRU, two daily Gamma heads."""
    def __init__(self, config: Neural48Config | Mapping):
        super().__init__()
        self.config = config if isinstance(config, Neural48Config) else Neural48Config(**dict(config))
        c = self.config
        self.nwp_projection = nn.Linear(c.nwp_features + 1, c.width)
        self.temporal = nn.Sequential(*[TemporalBlock(c.width, d, c.dropout) for d in [1, 2, 4, 8]])
        self.site = nn.Embedding(c.number_of_sites, c.site_embedding)
        self.radar_encoder = RadarEncoder(c) if c.variant == 'weather_radar' else None
        combined = 2*c.width + c.site_embedding + c.static_features
        if self.radar_encoder: combined += self.radar_encoder.output_features
        self.head = nn.Sequential(nn.Linear(combined, c.head_width), nn.GELU(),
                                  nn.Dropout(c.dropout), nn.Linear(c.head_width, 3))
        with torch.no_grad():
            self.head[-1].bias.copy_(torch.tensor([-1.5, 1., 1.]))
        self.register_buffer('lead_coordinate', torch.linspace(0., 1., 48).reshape(1, 48, 1))

    def forward(self, nwp, radar=None, local_stats=None, site_index=None, static=None):
        """Return per-day parameters/means of shape [batch,2].

        Inputs must already have training-only scaling and explicit missingness
        encodings. The daily outputs correspond to NWP hours1–24 and25–48.
        weather_only ignores radar and local_stats entirely.
        """
        c = self.config
        if nwp.ndim != 3 or nwp.shape[1:] != (48, c.nwp_features):
            raise ValueError('NWP must be [batch,48,nwp_features]')
        batch = len(nwp)
        if not torch.isfinite(nwp).all():
            raise ValueError('Encode missing NWP values with training-only preprocessing and masks')
        if site_index is None or site_index.shape != (batch,):
            raise ValueError('site_index must be [batch]')
        if site_index.dtype not in (torch.int32, torch.int64):
            raise ValueError('site_index must contain integer site indices')
        if ((site_index < 0) | (site_index >= c.number_of_sites)).any():
            raise ValueError('Unknown site index')
        if static is None and c.static_features == 0:
            static = nwp.new_zeros((batch, 0))
        if static is None or static.shape != (batch, c.static_features) or not torch.isfinite(static).all():
            raise ValueError('static must be finite [batch,static_features]')
        lead = self.lead_coordinate.to(dtype=nwp.dtype).expand(batch, -1, -1)
        h = self.temporal(self.nwp_projection(torch.cat([nwp, lead], dim=-1)).transpose(1, 2)).transpose(1, 2)
        pooled = torch.stack([h[:, :24].mean(1), h[:, 24:].mean(1)], dim=1)
        endpoints = torch.stack([h[:, 23], h[:, 47]], dim=1)
        parts = [pooled, endpoints, self.site(site_index).unsqueeze(1).expand(-1, 2, -1),
                 static.unsqueeze(1).expand(-1, 2, -1)]
        if self.radar_encoder:
            if radar is None or radar.shape != (batch, 9, c.radar_channels, 16, 16):
                raise ValueError('radar must be [batch,9,radar_channels,16,16]')
            if not torch.isfinite(radar).all():
                raise ValueError('Radar values must be finite with explicit masks')
            if c.local_features and (local_stats is None or local_stats.shape != (batch, 9, c.local_features)
                                     or not torch.isfinite(local_stats).all()):
                raise ValueError('local_stats must be finite [batch,9,local_features]')
            encoded = self.radar_encoder(radar, local_stats)
            parts.append(encoded.unsqueeze(1).expand(-1, 2, -1))
        raw = self.head(torch.cat(parts, dim=-1))
        logit = raw[..., 0]
        probability = torch.sigmoid(logit)
        positive_mean = (F.softplus(raw[..., 1]) + c.positive_mean_floor).clamp(max=c.positive_mean_ceiling)
        concentration = (F.softplus(raw[..., 2]) + c.concentration_floor).clamp(max=c.concentration_ceiling)
        rate = concentration / positive_mean
        return {'occurrence_logit': logit, 'probability': probability, 'positive_mean': positive_mean,
                'concentration': concentration, 'rate': rate, 'mean': probability * positive_mean}


def _target_mask(target, mask=None):
    if target.ndim != 2 or target.shape[1] != 2:
        raise ValueError('Daily target must have shape [batch,2]')
    finite = torch.isfinite(target)
    if ((target < 0) & finite & (torch.ones_like(finite) if mask is None else mask.bool())).any():
        raise ValueError('Observed snowfall targets cannot be negative')
    valid = finite if mask is None else finite & mask.bool()
    return valid


def zero_inflated_gamma_nll(params, target, mask=None, reduction='none'):
    """Masked daily NLL; unreduced missing cells are zero, never fitted as dry.

    For reduction='case', returns one mean NLL per case; cases with no valid
    target return NaN. 'mean' divides by the number of valid daily targets.
    """
    valid = _target_mask(target, mask)
    if params['mean'].shape != target.shape:
        raise ValueError('Forecast and target shapes differ')
    value = torch.where(valid & (target > 0), target, torch.ones_like(target))
    a, b, logit = params['concentration'], params['rate'], params['occurrence_logit']
    log_density = a*torch.log(b) - torch.lgamma(a) + (a-1)*torch.log(value) - b*value
    positive_nll = F.softplus(-logit) - log_density
    zero_nll = F.softplus(logit)
    nll = torch.where(valid, torch.where(target > 0, positive_nll, zero_nll), torch.zeros_like(target))
    if reduction == 'none': return nll
    if reduction == 'sum': return nll.sum()
    if reduction == 'mean': return nll.sum() / valid.sum().clamp_min(1)
    if reduction == 'case':
        result = nll.sum(1) / valid.sum(1).clamp_min(1)
        return torch.where(valid.any(1), result, torch.full_like(result, float('nan')))
    raise ValueError('reduction must be none, sum, mean, or case')


def forecast_loss(params, target, mask=None, nll_weight=.1):
    """Configurable physical daily MSE plus NLL; selection remains caller-owned."""
    if nll_weight < 0:
        raise ValueError('nll_weight must be nonnegative')
    valid = _target_mask(target, mask)
    safe = torch.where(valid, target, params['mean'].detach())
    mse = ((params['mean'] - safe).square() * valid).sum() / valid.sum().clamp_min(1)
    nll = zero_inflated_gamma_nll(params, target, valid, reduction='mean')
    return {'loss': mse + nll_weight*nll, 'mse': mse, 'nll': nll, 'valid_targets': valid.sum()}


def _numpy_parameters(params):
    def arr(value):
        return value.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(value) else np.asarray(value, np.float64)
    p, a, b = [arr(params[k]) for k in ['probability', 'concentration', 'rate']]
    if p.ndim != 2 or p.shape[1] != 2 or a.shape != p.shape or b.shape != p.shape:
        raise ValueError('Distribution parameters must have shape [batch,2]')
    if not all(np.isfinite(x).all() for x in [p, a, b]) or (p < 0).any() or (p > 1).any() or (a <= 0).any() or (b <= 0).any():
        raise ValueError('Invalid zero-inflated Gamma parameters')
    return p, a, b


def _mixture_quantile(p, a, b, q):
    positive_q = np.clip((q - (1-p)) / np.maximum(p, np.finfo(float).tiny), 0., 1.)
    values = gammaincinv(a, positive_q) / b
    return np.where((q <= (1-p)) | (p == 0), 0., values)


def distribution_quantiles(params, quantiles=(.1, .9)):
    """Return [quantile,batch,2] marginal quantiles in the target's units."""
    p, a, b = _numpy_parameters(params)
    q = np.asarray(quantiles, np.float64)
    if q.ndim != 1 or not np.isfinite(q).all() or ((q < 0) | (q > 1)).any():
        raise ValueError('Quantiles must be a finite vector in [0,1]')
    return _mixture_quantile(p[None], a[None], b[None], q[:, None, None])


def sample_joint_paths(params, n_paths=4096, seed=20261110, rho=0., uniforms=None):
    """Return [batch,path,2] daily amounts with explicitly chosen dependence.

    Default rho=0 means conditionally independent days. Nonzero rho is a Gaussian
    copula parameter that the caller must estimate using training data only.
    Alternatively, pass paired uniforms [batch,path,2] from a training-only
    empirical dependence construction. No fitting or dependence estimation is
    done here. Sum paths before computing any 48-hour uncertainty interval.
    """
    p, a, b = _numpy_parameters(params)
    if int(n_paths) != n_paths or n_paths < 1:
        raise ValueError('n_paths must be a positive integer')
    if uniforms is None:
        if not np.isfinite(rho) or not -.999 <= rho <= .999:
            raise ValueError('rho must lie in [-.999,.999]')
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((len(p), int(n_paths), 2))
        z[..., 1] = rho*z[..., 0] + np.sqrt(1-rho*rho)*z[..., 1]
        u = np.clip(ndtr(z), 1e-12, 1-1e-12)
    else:
        u = np.asarray(uniforms, np.float64)
        if u.ndim != 3 or u.shape[0] != len(p) or u.shape[2] != 2:
            raise ValueError('Paired uniforms must have shape [batch,path,2]')
        if not np.isfinite(u).all() or ((u <= 0) | (u >= 1)).any():
            raise ValueError('Paired uniforms must lie strictly between zero and one')
    return _mixture_quantile(p[:, None], a[:, None], b[:, None], u).astype(np.float32)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def checkpoint_payload(model, metadata=None):
    """Only configuration and tensors; caller owns scalers, split and provenance."""
    return {'model_family': 'weather_conditioned_neural48_zero_inflated_gamma',
            'format_version': 1, 'config': asdict(model.config),
            'state_dict': model.state_dict(), 'metadata': dict(metadata or {}),
            'target_definition': 'Two full24-hour snowfall amounts aligned to the supplied48-hour NWP sequence',
            'dependence_default': 'Conditionally independent daily Gamma mixtures unless caller supplies trained dependence'}


def restore_model(checkpoint, device='cpu'):
    """Restore only repository-owned checkpoints; never load untrusted files."""
    saved = torch.load(Path(checkpoint), map_location=device, weights_only=True) if isinstance(checkpoint, (str, Path)) else checkpoint
    if saved.get('model_family') != 'weather_conditioned_neural48_zero_inflated_gamma' or saved.get('format_version') != 1:
        raise ValueError('Unsupported neural48 checkpoint')
    model = Neural48Model(saved['config']).to(device)
    model.load_state_dict(saved['state_dict'], strict=True)
    model.eval()
    return model

