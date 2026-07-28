"""설계 v3 §5.2: 운율 벡터 p_a의 게이트 결합.

p_a(F0/jitter/shimmer/HNR 등)는 시퀀스가 아닌 발화 단위 고정 벡터이므로
교차 어텐션에 넣지 않고, 오디오의 하이브리드 표현과 게이트로 결합한다.

g = Sigmoid( W_g · [z_audio_hybrid ; p_a] )
z_audio_final = g ⊙ z_audio_hybrid + (1-g) ⊙ Linear(p_a)

"음성 억양 신호(z_audio_hybrid)가 잡음이 많아 신뢰도가 낮을 때 다른 정보
(p_a의 안정적 통계치)쪽 비중을 자동으로 높인다"는 설계 취지를 그대로 구현한다.
"""
import torch
import torch.nn as nn


class ProsodyGatedFusion(nn.Module):
    def __init__(self, hybrid_dim: int, prosody_dim: int):
        super().__init__()
        self.prosody_proj = nn.Linear(prosody_dim, hybrid_dim)
        self.gate = nn.Sequential(
            nn.Linear(hybrid_dim + prosody_dim, hybrid_dim),
            nn.Sigmoid(),
        )

    def forward(self, z_audio_hybrid: torch.Tensor, p_a: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([z_audio_hybrid, p_a], dim=-1))  # [B, hybrid_dim]
        p_a_proj = self.prosody_proj(p_a)  # [B, hybrid_dim]
        return g * z_audio_hybrid + (1 - g) * p_a_proj
