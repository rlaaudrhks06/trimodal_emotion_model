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


class DropPath(nn.Module):
    """Stochastic Depth — 학습 중 residual 분기(branch)를 샘플 단위로 통째로 건너뛴다.

    dropout이 "뉴런 일부"를 끄는 것과 달리, DropPath는 잔차 연결의 곁가지 전체를
    확률적으로 0으로 만들어 그 샘플에 대해 해당 블록을 사실상 항등함수로 만든다
    (Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016).
    비전 트랜스포머 계열(DeiT 등)에서 소규모 데이터 과적합 대응 표준 기법.

    8.11~8.12절 배경: BERT를 완전 동결(v7)해도 train/val 격차가 23.4pt 남았는데,
    남은 학습 가능 모듈 중 가장 큰 게 크로스어텐션 4블록(약 316만 파라미터)이다.
    여기엔 지금까지 dropout 외의 규제가 전혀 없었으므로 이 기법을 적용한다.

    학습 중에만 동작하고 eval()에서는 완전히 통과(항등)한다. 살아남은 경로는
    1/(1-p)로 나눠 스케일을 보정해서, 학습/추론 시 기댓값이 같게 유지한다.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # 배치의 샘플마다 독립적으로 살릴지 결정 — 첫 차원만 무작위이고 나머지는 브로드캐스트
        # (한 샘플의 경로를 끄면 그 샘플의 시퀀스·채널 전체가 함께 꺼져야 "경로를 건너뛴" 게 됨)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        keep_mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * keep_mask / keep_prob

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"


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
