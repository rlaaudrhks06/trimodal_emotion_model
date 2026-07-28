"""설계 v3 §5.1: 2단계 계층적 교차 어텐션 (총 4개 블록, 6개가 아님).

1단계 (언어적 축 결합): 오디오 ↔ 텍스트를 먼저 양방향으로 결합.
    CA1: Context_a  = CrossAttn(Q=X_a, KV=X_t)   # 오디오가 텍스트를 참조
    CA2: Context_t  = CrossAttn(Q=X_t, KV=X_a)   # 텍스트가 오디오를 참조

2단계 (표정 축 결합): 1단계 결과를 시간축으로 이어붙인 AT_seq와 시각을 결합.
    AT_seq = concat([Context_a, Context_t], dim=시간축)
    CA3: Context_v  = CrossAttn(Q=X_v,   KV=AT_seq)   # 시각이 (오디오+텍스트)를 참조
    CA4: Context_AT = CrossAttn(Q=AT_seq, KV=X_v)     # (오디오+텍스트)가 시각을 참조
         -> 다시 오디오 구간/텍스트 구간으로 분리해 각각 풀링

이렇게 하면 최종 분류기(§5.3)에 필요한 z_cross_v / z_cross_a / z_cross_t
세 벡터를 모두 얻으면서, 블록 수는 3쌍×양방향(6개)이 아닌 4개로 줄어든다.
"""
import torch
import torch.nn as nn

from .cross_attention import StackedCrossAttention


def mean_pool(x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
    """[B, T, d_model] -> [B, d_model]. 패딩 위치는 평균에서 제외."""
    if key_padding_mask is None:
        return x.mean(dim=1)
    valid = (~key_padding_mask).unsqueeze(-1).float()  # [B, T, 1], True=패딩이므로 반전
    summed = (x * valid).sum(dim=1)
    count = valid.sum(dim=1).clamp(min=1.0)
    return summed / count


class HierarchicalCrossAttentionFusion(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, n_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.ca_audio_attends_text = StackedCrossAttention(d_model, n_heads, ffn_dim, n_layers, dropout)
        self.ca_text_attends_audio = StackedCrossAttention(d_model, n_heads, ffn_dim, n_layers, dropout)
        self.ca_visual_attends_at = StackedCrossAttention(d_model, n_heads, ffn_dim, n_layers, dropout)
        self.ca_at_attends_visual = StackedCrossAttention(d_model, n_heads, ffn_dim, n_layers, dropout)

    def forward(
        self,
        x_v: torch.Tensor,
        x_a: torch.Tensor,
        x_t: torch.Tensor,
        v_mask: torch.Tensor | None = None,
        a_mask: torch.Tensor | None = None,
        t_mask: torch.Tensor | None = None,
    ):
        t_a = x_a.size(1)

        # --- 1단계: 오디오 <-> 텍스트 ---
        context_a = self.ca_audio_attends_text(x_a, x_t, kv_key_padding_mask=t_mask)  # [B, T_a, d]
        context_t = self.ca_text_attends_audio(x_t, x_a, kv_key_padding_mask=a_mask)  # [B, T_t, d]

        at_seq = torch.cat([context_a, context_t], dim=1)  # [B, T_a+T_t, d]
        if a_mask is not None or t_mask is not None:
            a_mask_ = a_mask if a_mask is not None else torch.zeros_like(context_a[..., 0], dtype=torch.bool)
            t_mask_ = t_mask if t_mask is not None else torch.zeros_like(context_t[..., 0], dtype=torch.bool)
            at_mask = torch.cat([a_mask_, t_mask_], dim=1)
        else:
            at_mask = None

        # --- 2단계: 시각 <-> (오디오+텍스트) ---
        context_v = self.ca_visual_attends_at(x_v, at_seq, kv_key_padding_mask=at_mask)  # [B, T_v, d]
        context_at = self.ca_at_attends_visual(at_seq, x_v, kv_key_padding_mask=v_mask)  # [B, T_a+T_t, d]

        context_at_a = context_at[:, :t_a, :]
        context_at_t = context_at[:, t_a:, :]

        z_cross_v = mean_pool(context_v, v_mask)
        z_cross_a = mean_pool(context_at_a, a_mask)
        z_cross_t = mean_pool(context_at_t, t_mask)

        return z_cross_v, z_cross_a, z_cross_t
