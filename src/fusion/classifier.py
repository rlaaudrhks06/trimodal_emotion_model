"""설계 v3 §5.3 / multimodal_pipeline_v2 §3.3(하이브리드 융합 취지 계승).

각 모달리티의 "미세 타이밍 단서"(교차 어텐션 풀링)와 "거시 분위기"(단일
모달리티 평균)를 함께 결합(concat)한 뒤 최종 분류기에 넣는다:

    z_v_final = concat[z_cross_v, mean(X_v)]
    z_t_final = concat[z_cross_t, mean(X_t)]
    z_audio_hybrid = concat[z_cross_a, mean(X_a)]  -> ProsodyGatedFusion -> z_audio_final

    z = concat[z_v_final, z_audio_final, z_t_final]
    p = Softmax( MLP(z) )  : [C]  (C는 config의 num_classes, src/datasets/labels.py 참고 — §11 기준 7)
"""
import torch
import torch.nn as nn


class HybridClassifier(nn.Module):
    def __init__(self, hybrid_dim: int, num_classes: int, hidden_dim: int | None = None, dropout: float = 0.2,
                 num_coarse: int = 0):
        super().__init__()
        hidden_dim = hidden_dim or hybrid_dim
        in_dim = hybrid_dim * 3  # z_v_final + z_audio_final + z_t_final
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        # v11g(13.14절): 3클래스(긍정/부정/중립) 머리. 은닉층(mlp[:3])을 7클래스 머리와 **공유**하고
        # 마지막 Linear만 따로 둔다 — 몸통 하나에 머리 둘. num_coarse=0(기본)이면 만들지 않아
        # v1~v11f와 파라미터·체크포인트가 완전히 같다.
        self.coarse = nn.Linear(hidden_dim, num_coarse) if num_coarse > 0 else None

    def forward(self, z_v_final: torch.Tensor, z_audio_final: torch.Tensor, z_t_final: torch.Tensor) -> torch.Tensor:
        z = torch.cat([z_v_final, z_audio_final, z_t_final], dim=-1)
        return self.mlp(z)  # logits [B, num_classes] — CrossEntropyLoss가 내부에서 softmax 처리

    def forward_with_coarse(self, z_v_final, z_audio_final, z_t_final) -> tuple[torch.Tensor, torch.Tensor]:
        """(7클래스 로짓, 3클래스 로짓). 은닉 표현을 한 번만 계산해 두 머리에 준다."""
        assert self.coarse is not None, "coarse 머리가 없다 — model.coarse_head를 켜야 한다"
        h = self.mlp[:3](torch.cat([z_v_final, z_audio_final, z_t_final], dim=-1))
        return self.mlp[3](h), self.coarse(h)
