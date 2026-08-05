"""양방향 교차 어텐션 블록. 설계 v3 §4.5(계승) / multimodal_pipeline_v2 §4.5.

Z = LayerNorm( Q_seq + MHCA(Q=Q_seq, K/V=KV_seq) )
Context = LayerNorm( Z + FFN(Z) )

패딩 마스킹은 KV_seq 쪽 key_padding_mask로 처리한다(설계 문서 4.3 "패딩된
오디오 위치는 Softmax 이전에 -inf로 마스킹").
"""
import torch
import torch.nn as nn

from ..models.common import DropPath


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1, drop_path: float = 0.0):
        super().__init__()
        self.mhca = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # 과적합 대응(§11.2): 어텐션 분기와 FFN 분기 각각에 stochastic depth를 건다.
        # drop_path=0.0(기본)이면 항등이라 기존 버전(v1~v8)과 완전히 동일하게 동작한다.
        self.drop_path = DropPath(drop_path)

    def forward(
        self,
        q_seq: torch.Tensor,
        kv_seq: torch.Tensor,
        kv_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # q_seq: [B, T_q, d_model], kv_seq: [B, T_kv, d_model]
        attn_out, _ = self.mhca(
            query=q_seq, key=kv_seq, value=kv_seq, key_padding_mask=kv_key_padding_mask
        )
        # drop_path가 걸린 샘플은 곁가지가 0이 되어 z = norm1(q_seq) — 즉 이 블록을 건너뛴 셈
        z = self.norm1(q_seq + self.drop_path(self.dropout(attn_out)))
        context = self.norm2(z + self.drop_path(self.ffn(z)))
        return context


class StackedCrossAttention(nn.Module):
    """N회 반복(설계 문서의 "N 회 반복 가능")을 지원하는 래퍼.

    매 반복마다 동일한 KV_seq를 참조하되, Q_seq는 이전 블록의 출력으로 갱신된다.
    """

    def __init__(
        self, d_model: int, n_heads: int, ffn_dim: int, n_layers: int = 1,
        dropout: float = 0.1, drop_path: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(d_model, n_heads, ffn_dim, dropout, drop_path) for _ in range(n_layers)]
        )

    def forward(
        self,
        q_seq: torch.Tensor,
        kv_seq: torch.Tensor,
        kv_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = q_seq
        for block in self.blocks:
            h = block(h, kv_seq, kv_key_padding_mask=kv_key_padding_mask)
        return h
