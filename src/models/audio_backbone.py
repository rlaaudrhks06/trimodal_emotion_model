"""설계 v3 §4 ③: Audio Backbone.

입력은 미리 계산된 멜-스펙트로그램 [B, T_a, n_mels] (features/audio_frontend.py 참고).
피치·jitter·shimmer 등 운율 특징(p_a)은 이 백본이 아니라 features/prosody.py에서
발화 단위 고정 벡터로 별도 추출되어, fusion/gated_prosody.py에서 결합된다
(설계 v3 §5.2 — 시퀀스가 아니므로 여기 포함하지 않음).
"""
import torch
import torch.nn as nn

from .common import TemporalConvFrontend


class AudioBackbone(nn.Module):
    def __init__(self, n_mels: int, d_model: int, n_heads: int, ffn_dim: int, n_layers: int):
        super().__init__()
        self.frontend = TemporalConvFrontend(
            in_dim=n_mels, d_model=d_model, n_heads=n_heads, ffn_dim=ffn_dim, n_layers=n_layers
        )

    def forward(self, mel_spec: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # mel_spec: [B, T_a, n_mels] -> X_a: [B, T_a, d_model]
        return self.frontend(mel_spec, key_padding_mask=key_padding_mask)
