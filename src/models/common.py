"""오디오/비주얼 프론트엔드가 공유하는 모듈: 설계 v3 4절 '②전처리' 단계.

Temporal Conv (국소 시간 구조 보존) + Positional Encoding (순서 정보 주입)
"""
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """표준 사인/코사인 위치 인코딩 (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        return x + self.pe[:, : x.size(1), :]


class TemporalConvFrontend(nn.Module):
    """설계 v3 §4 ②: Temporal Conv + Positional Encoding + TransformerEncoder.

    입력 [B, T, in_dim] (예: 멜스펙트로그램 프레임, 비주얼 프레임 임베딩)을
    받아 [B, T, d_model] 시퀀스(X_v 또는 X_a)로 변환한다.
    """

    def __init__(self, in_dim: int, d_model: int, n_heads: int, ffn_dim: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_dim, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
        )
        # conv 출력 자체에는 규제가 전혀 없었음(TransformerEncoderLayer 내부 dropout만 존재) ->
        # 과적합 대응으로 conv 뒤에도 dropout 추가
        self.conv_dropout = nn.Dropout(dropout)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        # enable_nested_tensor=False: PyTorch가 패딩 마스크 있을 때 자동으로 쓰는
        # nested-tensor 최적화 경로가 MPS(애플 GPU)에서 구현 안 된 연산을 호출해 죽는
        # 문제가 있어 꺼둔다 (CPU/CUDA에서는 속도 차이만 있고 정확도엔 영향 없음).
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, T, in_dim] -> Conv1d는 [B, in_dim, T] 형태를 요구
        h = self.temporal_conv(x.transpose(1, 2)).transpose(1, 2)  # [B, T, d_model]
        h = self.conv_dropout(h)
        h = self.pos_enc(h)
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        return h
