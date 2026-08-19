# -*- coding: utf-8 -*-
"""PyTorch 版 WT-Transformer。

结构对应原 `transformer_full.py` 的 Keras 实现：位置编码、LayerNorm、
多头自注意力、残差连接、逐时间步 feed-forward、全局池化和 MLP 回归头。
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
from torch import nn


def sinusoidal_positional_encoding(seq_len: int, dim: int, device=None, dtype=None) -> torch.Tensor:
    position = torch.arange(seq_len, device=device, dtype=dtype or torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype or torch.float32)
        * (-math.log(10000.0) / max(dim, 1))
    )
    pe = torch.zeros(seq_len, dim, device=device, dtype=dtype or torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    if dim > 1:
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe.unsqueeze(0)


class WTTransformerBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        head_size: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(input_dim, eps=1e-6)
        self.attn_dim = int(head_size) * int(num_heads)
        self.attn_in = nn.Linear(input_dim, self.attn_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.attn_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.attn_out = nn.Linear(self.attn_dim, input_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.norm2 = nn.LayerNorm(input_dim, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, int(ff_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(ff_dim), input_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        seq_len = inputs.shape[1]
        x = self.norm1(inputs)
        x = x + sinusoidal_positional_encoding(seq_len, inputs.shape[-1], x.device, x.dtype)
        qkv = self.attn_in(x)
        attn_out, _ = self.attn(qkv, qkv, qkv, need_weights=False)
        res = inputs + self.dropout(self.attn_out(attn_out))
        x = self.norm2(res)
        return res + self.ffn(x)


class WTTransformerTorch(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        seq_len: int = 144,
        head_size: int = 256,
        num_heads: int = 4,
        ff_dim: int = 4,
        num_transformer_blocks: int = 4,
        mlp_units: Sequence[int] | Iterable[int] = (100,),
        dropout: float = 0.2,
        mlp_dropout: float = 0.3,
        pooling_mode: str = "keras_channels_first",
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.seq_len = int(seq_len)
        self.pooling_mode = str(pooling_mode)
        # P1-fix#2 (2026-06-10): out_dim=D 时预测【全部通道】下一步值 (多通道新息打分);
        #   out_dim=1 保持旧单目标行为 (向后兼容)。
        self.out_dim = int(out_dim)
        self.blocks = nn.ModuleList(
            [
                WTTransformerBlock(
                    self.input_dim,
                    head_size=int(head_size),
                    num_heads=int(num_heads),
                    ff_dim=int(ff_dim),
                    dropout=float(dropout),
                )
                for _ in range(int(num_transformer_blocks))
            ]
        )
        if self.pooling_mode == "keras_channels_first":
            pooled_dim = self.seq_len
        elif self.pooling_mode == "time_mean":
            pooled_dim = self.input_dim
        else:
            raise ValueError("pooling_mode 必须是 'keras_channels_first' 或 'time_mean'")
        mlp_layers = []
        in_dim = pooled_dim
        for units in mlp_units:
            mlp_layers.extend([nn.Linear(in_dim, int(units)), nn.ReLU(), nn.Dropout(float(mlp_dropout))])
            in_dim = int(units)
        mlp_layers.append(nn.Linear(in_dim, self.out_dim))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(f"inputs 必须是 (batch, seq, channels)，实际 shape={tuple(inputs.shape)}")
        x = inputs
        for block in self.blocks:
            x = block(x)
        if self.pooling_mode == "keras_channels_first":
            # 原 Keras 代码对 (batch, steps, channels) 使用 data_format="channels_first"，
            # 因而按最后一维求均值并保留 steps 维度。
            pooled = x.mean(dim=2)
        else:
            pooled = x.mean(dim=1)
        out = self.mlp(pooled)
        # out_dim=1 → squeeze 成 (B,) 保持旧行为; out_dim=D → (B, D) 多通道预测
        return out.squeeze(-1) if self.out_dim == 1 else out
