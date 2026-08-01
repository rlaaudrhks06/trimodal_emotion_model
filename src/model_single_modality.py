"""설계 v3 §7.3 베이스라인 1: 단일 모달리티(텍스트 / 오디오 / 영상 only) 모델.

트리모달 본 모델(TrimodalEmotionModel)과 비교하기 위한 베이스라인이다.
교차 어텐션·운율 게이트 없이 "해당 모달리티 백본 -> 평균 풀링 -> MLP 분류기"로
끝나는 가장 단순한 구조 — 어느 모달리티가 실제로 감정 정보를 주는지 확인하는 용도.

오디오만 예외적으로 prosody_vec(F0/jitter/shimmer 등)을 같이 쓴다 — 설계상
"오디오 모달리티" 자체가 멜스펙트로그램+운율의 합이라(§3.5), 운율을 빼면
오디오 단일 모달리티를 제대로 대표하지 못하기 때문이다.
"""
import torch
import torch.nn as nn

from .config import Config
from .models.audio_backbone import AudioBackbone
from .models.visual_backbone import VisualBackbone
from .models.text_backbone import TextBackbone
from .fusion.hierarchical_fusion import mean_pool


class SingleModalityModel(nn.Module):
    def __init__(self, cfg: Config, modality: str):
        super().__init__()
        if modality not in ("audio", "visual", "text"):
            raise ValueError(f"modality must be audio/visual/text, got {modality!r}")
        self.modality = modality
        m = cfg.model

        if modality == "audio":
            self.backbone = AudioBackbone(
                n_mels=cfg.audio_n_mels, d_model=m.d_model, n_heads=m.n_heads,
                ffn_dim=m.ffn_dim, n_layers=m.backbone_layers, dropout=m.backbone_dropout,
            )
            self.prosody_proj = nn.Linear(m.prosody_dim, m.d_model)
            feat_dim = m.d_model * 2  # mean(X_a) + prosody
        elif modality == "visual":
            self.backbone = VisualBackbone(
                d_model=m.d_model, n_heads=m.n_heads, ffn_dim=m.ffn_dim, n_layers=m.backbone_layers,
                dropout=m.backbone_dropout, cnn_dropout=m.visual_cnn_dropout,
            )
            feat_dim = m.d_model
        else:  # text
            self.backbone = TextBackbone(
                pretrained_model=cfg.text_pretrained, d_model=m.d_model,
                dropout=m.text_dropout, freeze_layers=m.text_freeze_layers,
            )
            feat_dim = m.d_model

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(m.classifier_dropout),
            nn.Linear(feat_dim, m.num_classes),
        )

    def forward(
        self,
        mel_spec: torch.Tensor | None = None,
        prosody_vec: torch.Tensor | None = None,
        frames: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        audio_padding_mask: torch.Tensor | None = None,
        visual_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # train.py의 run_epoch()가 트리모달 모델과 동일한 키워드 인자로 호출하므로
        # (필요없는 나머지 모달리티 텐서는 그냥 무시) run_epoch/평가 로직을 그대로 재사용할 수 있다.
        if self.modality == "audio":
            x_a = self.backbone(mel_spec, key_padding_mask=audio_padding_mask)
            pooled = mean_pool(x_a, audio_padding_mask)
            feat = torch.cat([pooled, self.prosody_proj(prosody_vec)], dim=-1)
        elif self.modality == "visual":
            x_v = self.backbone(frames, key_padding_mask=visual_padding_mask)
            feat = mean_pool(x_v, visual_padding_mask)
        else:
            x_t = self.backbone(input_ids, attention_mask)
            t_key_padding_mask = attention_mask == 0
            feat = mean_pool(x_t, t_key_padding_mask)

        return self.classifier(feat)
