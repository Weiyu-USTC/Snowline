"""Multivariate temporal architectures for the Dillon snowfall benchmark.

All models consume already observed context [batch, 60, 14] and known future
calendar/lead/lagged-ENSO features [batch, 61, 10]. They return snow-occurrence
logits and positive-lognormal mu/sigma, each [batch, 61]. No architecture receives
future observed weather. More depth is a candidate design, not a skill claim.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

CONTEXT_DAYS = 60
HISTORICAL_FEATURES = 14
FUTURE_FEATURES = 10


def snowfall_parameters(raw: torch.Tensor):
    """Use exactly the original GRU's distribution support and bounds."""
    return (raw[..., 0], 3 * torch.tanh(raw[..., 1] / 3),
            .15 + 1.35 * torch.sigmoid(raw[..., 2]))


def initialize_distribution_head(layer: nn.Linear):
    # Match the original model's initialization of occurrence/amount biases.
    with torch.no_grad():
        layer.bias.copy_(torch.tensor([-1.2, -.1, .3], dtype=layer.bias.dtype))


def validate_inputs(x: torch.Tensor, future: torch.Tensor):
    if x.ndim != 3 or tuple(x.shape[1:]) != (CONTEXT_DAYS, HISTORICAL_FEATURES):
        raise ValueError(f'Expected context [batch,{CONTEXT_DAYS},{HISTORICAL_FEATURES}], got {tuple(x.shape)}')
    if future.ndim != 3 or future.shape[0] != x.shape[0] or future.shape[2] != FUTURE_FEATURES:
        raise ValueError(f'Expected future features [batch,horizon,{FUTURE_FEATURES}], got {tuple(future.shape)}')


class ChannelLayerNorm(nn.Module):
    """Normalize channels within each historical date, never across dates."""
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class CausalConv1d(nn.Conv1d):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.left_padding = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor):
        return super().forward(F.pad(x, (self.left_padding, 0)))


class TemporalResidualBlock(nn.Module):
    """Two pre-normalized causal convolutions with an identity residual path."""
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.branch = nn.Sequential(
            ChannelLayerNorm(channels), nn.SiLU(),
            CausalConv1d(channels, kernel_size, dilation), nn.Dropout(dropout),
            ChannelLayerNorm(channels), nn.SiLU(),
            CausalConv1d(channels, kernel_size, dilation), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor):
        return x + self.branch(x)


class ResidualSnowTCN(nn.Module):
    """Learn weather patterns at several time scales using dilated filters.

    The historical encoder is causal at every date. Its final state and pooled
    past states summarize the complete observation window. A shared decoder
    predicts each lead using that summary and the lead's known future features.
    """
    def __init__(self, width=32, blocks=5, dropout=.2, kernel_size=3):
        super().__init__()
        if width < 1 or blocks < 1 or kernel_size < 2:
            raise ValueError('TCN requires positive width/blocks and kernel_size >= 2')
        self.width, self.blocks, self.kernel_size = width, blocks, kernel_size
        self.dilations = [2 ** block for block in range(blocks)]
        self.receptive_field_days = 1 + 2 * (kernel_size - 1) * sum(self.dilations)
        self.input_projection = nn.Conv1d(HISTORICAL_FEATURES, width, kernel_size=1)
        self.temporal_blocks = nn.Sequential(*[
            TemporalResidualBlock(width, kernel_size, dilation, dropout)
            for dilation in self.dilations])
        self.output_norm = ChannelLayerNorm(width)
        self.decoder = nn.Sequential(
            nn.Linear(2 * width + FUTURE_FEATURES, 128), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.SiLU(), nn.Dropout(dropout), nn.Linear(64, 3))
        initialize_distribution_head(self.decoder[-1])

    def encode(self, x: torch.Tensor):
        return self.output_norm(self.temporal_blocks(self.input_projection(x.transpose(1, 2))))

    def forward(self, x: torch.Tensor, future: torch.Tensor):
        validate_inputs(x, future)
        encoded = self.encode(x)
        summary = torch.cat([encoded[:, :, -1], encoded.mean(dim=-1)], dim=-1)
        summary = summary[:, None, :].expand(-1, future.shape[1], -1)
        return snowfall_parameters(self.decoder(torch.cat([summary, future], dim=-1)))


class PatchSnowTransformer(nn.Module):
    """Multivariate patch attention with forecast queries over observed history.

    Consecutive historical days form multivariate patches. All patches are
    observed by issue time, so historical self-attention need not be masked.
    Each future query attends only to this history and has no access to labels,
    actual future weather, or other future query values. This is a compact
    custom architecture inspired by patch forecasting, not a PatchTST replica.
    """
    def __init__(self, d_model=64, layers=3, nhead=4, patch_days=5,
                 ff_multiplier=3, dropout=.2):
        super().__init__()
        if CONTEXT_DAYS % patch_days:
            raise ValueError('patch_days must divide the 60-day context')
        if d_model % nhead:
            raise ValueError('d_model must be divisible by nhead')
        if layers < 1 or ff_multiplier < 1:
            raise ValueError('Positive layers and feed-forward multiplier required')
        self.patch_days, self.patch_count = patch_days, CONTEXT_DAYS // patch_days
        self.d_model, self.layers, self.nhead = d_model, layers, nhead
        self.patch_projection = nn.Linear(patch_days * HISTORICAL_FEATURES, d_model)
        self.position = nn.Parameter(torch.empty(1, self.patch_count, d_model))
        nn.init.normal_(self.position, std=.02)
        self.embedding_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * ff_multiplier,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.history_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=layers, norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False)
        # PyTorch clones its prototype encoder layer. Independently initialize
        # each layer so the stack does not begin with identical learned weights.
        for layer in self.history_encoder.layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            for linear in [layer.self_attn.out_proj, layer.linear1, layer.linear2]:
                nn.init.xavier_uniform_(linear.weight)
                if linear.bias is not None:
                    nn.init.zeros_(linear.bias)
            if layer.self_attn.in_proj_bias is not None:
                nn.init.zeros_(layer.self_attn.in_proj_bias)
        self.query_projection = nn.Linear(FUTURE_FEATURES, d_model)
        self.query_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_dropout = nn.Dropout(dropout)
        self.forecast_ff = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model * ff_multiplier),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * ff_multiplier, d_model),
            nn.Dropout(dropout))
        self.output_norm = nn.LayerNorm(d_model)
        self.distribution_head = nn.Linear(d_model, 3)
        initialize_distribution_head(self.distribution_head)

    def encode(self, x: torch.Tensor):
        patches = x.reshape(x.shape[0], self.patch_count, self.patch_days * HISTORICAL_FEATURES)
        tokens = self.embedding_dropout(self.patch_projection(patches) + self.position)
        return self.history_encoder(tokens)

    def forward(self, x: torch.Tensor, future: torch.Tensor):
        validate_inputs(x, future)
        memory = self.encode(x)
        query = self.query_projection(future)
        attended, _ = self.cross_attention(self.query_norm(query), memory, memory, need_weights=False)
        forecast = query + self.cross_dropout(attended)
        forecast = forecast + self.forecast_ff(forecast)
        return snowfall_parameters(self.distribution_head(self.output_norm(forecast)))


def make_model(config: dict[str, Any]):
    family = config['family']
    if family == 'gru':
        from deep_dillon import SnowGRU
        return SnowGRU(hidden=config.get('hidden', 32), dropout=config.get('dropout', .25))
    if family == 'tcn':
        return ResidualSnowTCN(width=config.get('width', 32), blocks=config.get('blocks', 5),
                               dropout=config.get('dropout', .2), kernel_size=config.get('kernel_size', 3))
    if family == 'patch_transformer':
        return PatchSnowTransformer(d_model=config.get('d_model', 64), layers=config.get('layers', 3),
                                    nhead=config.get('nhead', 4), patch_days=config.get('patch_days', 5),
                                    ff_multiplier=config.get('ff_multiplier', 3), dropout=config.get('dropout', .2))
    raise ValueError(f'Unknown snowfall model family: {family}')


def count_parameters(model: nn.Module):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def architecture_metadata(config: dict[str, Any], model: nn.Module | None = None):
    model = make_model(config) if model is None else model
    result = {'family': config['family'], 'trainable_parameters': count_parameters(model),
              'context_days': CONTEXT_DAYS, 'historical_features': HISTORICAL_FEATURES,
              'future_known_features': FUTURE_FEATURES,
              'output': 'zero-inflated lognormal occurrence logits, positive-amount log-space mean and log-space standard deviation',
              'future_weather_observations_used': False}
    if isinstance(model, ResidualSnowTCN):
        result.update(description='Residual dilated causal temporal convolutions, last-state plus mean pooling, shared forecast decoder',
                      residual_blocks=model.blocks, temporal_convolution_layers=2 * model.blocks,
                      dilations=model.dilations, width=model.width,
                      encoder_receptive_field_days=model.receptive_field_days,
                      available_context_days=CONTEXT_DAYS)
    elif isinstance(model, PatchSnowTransformer):
        result.update(description='Multivariate historical patch encoder with forecast-query cross-attention',
                      encoder_blocks=model.layers, attention_heads=model.nhead, embedding_width=model.d_model,
                      historical_patch_days=model.patch_days, historical_tokens=model.patch_count,
                      forecast_cross_attention_blocks=1, future_query_self_attention=False)
    else:
        result.update(description='Original two-layer GRU historical encoder and shared SiLU forecast decoder',
                      recurrent_layers=2, recurrent_width=config.get('hidden', 32))
    return result
