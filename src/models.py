from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_type: str
    input_length: int
    label_length: int
    prediction_length: int
    encoder_input_size: int
    time_feature_size: int
    d_model: int = 64
    n_heads: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 1
    d_ff: int = 128
    dropout: float = 0.1
    attention_type: str = "prob"
    factor: int = 5
    distil: bool = True
    generative_decoder: bool = True
    mix: bool = True


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        position = torch.arange(length, device=x.device, dtype=x.dtype).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, self.d_model, 2, device=x.device, dtype=x.dtype)
            * (-math.log(10000.0) / self.d_model)
        )
        embedding = torch.zeros(length, self.d_model, device=x.device, dtype=x.dtype)
        embedding[:, 0::2] = torch.sin(position * frequencies)
        embedding[:, 1::2] = torch.cos(position * frequencies[: embedding[:, 1::2].shape[1]])
        return embedding.unsqueeze(0)


class DataEmbedding(nn.Module):
    """Value + sinusoidal position + projected calendar features."""

    def __init__(self, value_size: int, mark_size: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.value_embedding = nn.Conv1d(
            value_size,
            d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        self.position_embedding = PositionalEmbedding(d_model)
        self.time_embedding = nn.Linear(mark_size, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, marks: torch.Tensor) -> torch.Tensor:
        value_embedding = self.value_embedding(values.transpose(1, 2)).transpose(1, 2)
        return self.dropout(
            value_embedding + self.position_embedding(value_embedding) + self.time_embedding(marks)
        )


class FullAttention(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        causal: bool = False,
    ) -> torch.Tensor:
        # Inputs use [batch, length, heads, head_dim].
        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(queries.size(-1))
        if causal:
            mask = torch.ones(
                scores.size(-2), scores.size(-1), device=scores.device, dtype=torch.bool
            ).triu(diagonal=1)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        output = torch.matmul(weights, values)
        return output.permute(0, 2, 1, 3).contiguous()


class ProbSparseAttention(nn.Module):
    """ProbSparse attention following the query-sparsity strategy of Informer2020."""

    def __init__(self, factor: int, dropout: float) -> None:
        super().__init__()
        self.factor = factor
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        causal: bool = False,
    ) -> torch.Tensor:
        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)
        batch, heads, query_length, head_dim = queries.shape
        key_length = keys.size(2)

        # Match Informer2020's c * ceil(log(L)); keep length-one inputs valid.
        sample_keys = min(key_length, self.factor * max(1, math.ceil(math.log(key_length))))
        top_queries = min(query_length, self.factor * max(1, math.ceil(math.log(query_length))))

        sampled_indices = torch.randint(
            key_length,
            (query_length, sample_keys),
            device=queries.device,
        )
        sampled_keys = keys[:, :, sampled_indices, :]
        sampled_scores = torch.matmul(
            queries.unsqueeze(-2), sampled_keys.transpose(-2, -1)
        ).squeeze(-2)
        # The official approximation divides the sampled sum by all keys,
        # rather than by the number of sampled keys.
        sparsity = sampled_scores.max(dim=-1).values - sampled_scores.sum(dim=-1) / key_length
        top_indices = sparsity.topk(top_queries, dim=-1, sorted=False).indices

        selected_queries = torch.gather(
            queries,
            2,
            top_indices.unsqueeze(-1).expand(batch, heads, top_queries, head_dim),
        )
        scores = torch.matmul(selected_queries, keys.transpose(-2, -1)) / math.sqrt(head_dim)
        if causal:
            key_positions = torch.arange(key_length, device=queries.device)
            mask = key_positions.view(1, 1, 1, key_length) > top_indices.unsqueeze(-1)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        updates = torch.matmul(weights, values)

        if causal:
            if query_length != key_length:
                raise ValueError("Causal ProbSparse attention requires equal query/key lengths.")
            context = values.cumsum(dim=2)
        else:
            context = values.mean(dim=2, keepdim=True).expand(
                batch, heads, query_length, head_dim
            ).clone()
        context.scatter_(
            2,
            top_indices.unsqueeze(-1).expand(batch, heads, top_queries, head_dim),
            # CUDA autocast promotes cumsum to float32 but matmul to float16.
            # In-place scatter does not autocast; preserve the context precision.
            updates.to(dtype=context.dtype),
        )
        return context.permute(0, 2, 1, 3).contiguous()


class AttentionLayer(nn.Module):
    def __init__(
        self, attention: nn.Module, d_model: int, n_heads: int, mix: bool = False
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.mix = mix
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.attention = attention

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, query_length, _ = queries.shape
        key_length = keys.size(1)
        queries = self.query_projection(queries).view(
            batch, query_length, self.n_heads, self.head_dim
        )
        keys = self.key_projection(keys).view(batch, key_length, self.n_heads, self.head_dim)
        values = self.value_projection(values).view(
            batch, key_length, self.n_heads, self.head_dim
        )
        output = self.attention(queries, keys, values, causal=causal)
        if self.mix:
            # Preserve the official decoder's transpose-before-reshape behavior.
            output = output.transpose(2, 1).contiguous()
        return self.output_projection(output.reshape(batch, query_length, -1))


def make_attention(kind: str, factor: int, dropout: float) -> nn.Module:
    if kind == "prob":
        return ProbSparseAttention(factor=factor, dropout=dropout)
    if kind == "full":
        return FullAttention(dropout=dropout)
    raise ValueError(f"Unknown attention type: {kind}")


class InformerEncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.self_attention = AttentionLayer(
            make_attention(config.attention_type, config.factor, config.dropout),
            config.d_model,
            config.n_heads,
        )
        self.conv1 = nn.Conv1d(config.d_model, config.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(config.d_ff, config.d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attention(x, x, x, causal=False)))
        feed_forward = self.conv2(
            self.dropout(F.gelu(self.conv1(x.transpose(1, 2))))
        ).transpose(1, 2)
        return self.norm2(x + self.dropout(feed_forward))


class DistilConvLayer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.convolution = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.normalization = nn.BatchNorm1d(channels)
        self.pooling = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolution(x.transpose(1, 2))
        x = self.pooling(F.elu(self.normalization(x)))
        return x.transpose(1, 2)


class InformerEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [InformerEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.distillers = nn.ModuleList(
            [DistilConvLayer(config.d_model) for _ in range(config.encoder_layers - 1)]
            if config.distil
            else []
        )
        self.normalization = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.distillers):
                x = self.distillers[index](x)
        return self.normalization(x)


class InformerDecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.self_attention = AttentionLayer(
            make_attention(config.attention_type, config.factor, config.dropout),
            config.d_model,
            config.n_heads,
            mix=config.mix,
        )
        self.cross_attention = AttentionLayer(
            FullAttention(config.dropout), config.d_model, config.n_heads
        )
        self.conv1 = nn.Conv1d(config.d_model, config.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(config.d_ff, config.d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attention(x, x, x, causal=True)))
        x = self.norm2(
            x + self.dropout(self.cross_attention(x, memory, memory, causal=False))
        )
        feed_forward = self.conv2(
            self.dropout(F.gelu(self.conv1(x.transpose(1, 2))))
        ).transpose(1, 2)
        return self.norm3(x + self.dropout(feed_forward))


class Informer(nn.Module):
    """Modern PyTorch adaptation of the official Informer2020 architecture."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder_embedding = DataEmbedding(
            config.encoder_input_size,
            config.time_feature_size,
            config.d_model,
            config.dropout,
        )
        self.encoder = InformerEncoder(config)
        if config.generative_decoder:
            self.decoder_embedding = DataEmbedding(
                1, config.time_feature_size, config.d_model, config.dropout
            )
            self.decoder_layers = nn.ModuleList(
                [InformerDecoderLayer(config) for _ in range(config.decoder_layers)]
            )
            self.decoder_normalization = nn.LayerNorm(config.d_model)
            self.projection = nn.Linear(config.d_model, 1)
        else:
            self.direct_head = nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.prediction_length),
            )

    def forward(
        self,
        encoder_values: torch.Tensor,
        encoder_marks: torch.Tensor,
        decoder_values: torch.Tensor,
        decoder_marks: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encoder(self.encoder_embedding(encoder_values, encoder_marks))
        if not self.config.generative_decoder:
            return self.direct_head(memory[:, -1]).unsqueeze(-1)

        output = self.decoder_embedding(decoder_values, decoder_marks)
        for layer in self.decoder_layers:
            output = layer(output, memory)
        output = self.projection(self.decoder_normalization(output))
        return output[:, -self.config.prediction_length :]


class VanillaTransformer(nn.Module):
    """Self-contained time-series Transformer baseline using PyTorch layers."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder_embedding = DataEmbedding(
            config.encoder_input_size,
            config.time_feature_size,
            config.d_model,
            config.dropout,
        )
        self.decoder_embedding = DataEmbedding(
            1, config.time_feature_size, config.d_model, config.dropout
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.encoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.decoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.prediction_head = nn.Linear(config.d_model, 1)

    def forward(
        self,
        encoder_values: torch.Tensor,
        encoder_marks: torch.Tensor,
        decoder_values: torch.Tensor,
        decoder_marks: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encoder(self.encoder_embedding(encoder_values, encoder_marks))
        target = self.decoder_embedding(decoder_values, decoder_marks)
        length = target.size(1)
        causal_mask = torch.ones(length, length, device=target.device, dtype=torch.bool).triu(1)
        output = self.decoder(target, memory, tgt_mask=causal_mask)
        output = self.prediction_head(output)
        return output[:, -self.config.prediction_length :]


def build_model(config: ModelConfig) -> nn.Module:
    if config.model_type == "transformer":
        return VanillaTransformer(config)
    if config.model_type == "informer":
        return Informer(config)
    raise ValueError(f"Unknown model type: {config.model_type}")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
