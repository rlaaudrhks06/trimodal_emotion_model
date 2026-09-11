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


class ProsodyConcatFusion(nn.Module):
    """게이트 없이 운율을 그냥 이어붙이는 대조군(11.3.2 항목 3).

        z_audio_final = Linear([z_audio_hybrid ; p_a])

    **왜 이 조건이 필요한가.** v11 체크포인트에서 게이트 값 g를 직접 재보니
    (`scripts/inspect_prosody_gate.py`, 96발화 x 512차원) 이렇게 나왔다:

      - 게이트는 죽어 있지 않다 — 512차원 중 298개가 운율 쪽으로 기울어 있고,
        발화마다 차원별로 크게 재배분한다(발화별 변동 0.2278 vs 차원 간 차이 0.0998)
      - 그런데 소음이 와도 **총량을 안 옮긴다.** SNR 10dB에서 방향은 유의하지만
        (73/96 발화, p=1.56e-07) 옮긴 양이 평소 변동의 5.5%다. 같은 조건에서
        정확도는 46.19% -> 36.85%로 무너지는데 g는 0.4854 -> 0.4730만 움직인다
        (서버 CUDA 기준 — Mac은 깨끗 g가 0.4895로 0.004 다르다, 백엔드 차이)

    즉 게이트가 하는 일이 "입력에 따라 비중을 조절"이 아니라 사실상 **고정 배합**에
    가까울 수 있다. 그렇다면 sigmoid 게이트 기구는 값을 못 하는 것이고, 단순 선형
    결합으로 같은 성능이 나와야 한다. 이 조건이 그것을 잰다.

    **파라미터가 게이트와 거의 같다** — 비교가 깨끗하다:

        gate    273,408  (prosody_proj 5,632 + gate Linear 267,776)
        concat  267,776  (Linear(hybrid+prosody -> hybrid) 하나)
        none          0

    concat과 gate의 차이가 5,632개(2%)뿐이라, 성능 차이가 나면 그건 파라미터 수가
    아니라 **입력 의존적 게이팅이라는 기구 자체**의 값어치다.
    """

    def __init__(self, hybrid_dim: int, prosody_dim: int):
        super().__init__()
        self.proj = nn.Linear(hybrid_dim + prosody_dim, hybrid_dim)

    def forward(self, z_audio_hybrid: torch.Tensor, p_a: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([z_audio_hybrid, p_a], dim=-1))
