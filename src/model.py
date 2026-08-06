"""설계 v3 전체 아키텍처를 하나로 조립한 TrimodalEmotionModel.

End-to-End 흐름 (설계 문서 §4):
① 입력 -> ② 전처리 -> ③ 백본(X_v,X_a,X_t) -> ④ 계층적 교차 어텐션(§5.1)
-> 운율 게이트 결합(§5.2) -> ⑤ 하이브리드 결합 -> ⑥ 분류기(§5.3)
"""
import random

import torch
import torch.nn as nn

from .config import Config
from .models.audio_backbone import AudioBackbone
from .models.audio_backbone_w2v import Wav2Vec2AudioBackbone
from .models.visual_backbone import VisualBackbone
from .models.text_backbone import TextBackbone
from .fusion.hierarchical_fusion import HierarchicalCrossAttentionFusion, mean_pool
from .fusion.gated_prosody import ProsodyGatedFusion
from .fusion.classifier import HybridClassifier


class TrimodalEmotionModel(nn.Module):
    def __init__(self, cfg: Config, modality_dropout_prob: float = 0.0):
        super().__init__()
        m = cfg.model
        self.d_model = m.d_model
        self.modality_dropout_prob = modality_dropout_prob

        # 오디오 백본은 config로 고른다(11.0.1절). "mel"이 기본이라 v1~v10은 그대로 재현된다.
        self.use_w2v = cfg.audio_backbone == "wav2vec2"
        if self.use_w2v:
            self.audio_backbone = Wav2Vec2AudioBackbone(
                pretrained_model=cfg.audio_pretrained, d_model=m.d_model, n_heads=m.n_heads,
                ffn_dim=m.ffn_dim, n_layers=m.backbone_layers, layer=cfg.audio_w2v_layer,
                dropout=m.backbone_dropout, freeze=cfg.audio_w2v_freeze,
            )
        else:
            self.audio_backbone = AudioBackbone(
                n_mels=cfg.audio_n_mels, d_model=m.d_model, n_heads=m.n_heads,
                ffn_dim=m.ffn_dim, n_layers=m.backbone_layers, dropout=m.backbone_dropout,
            )
        self.visual_backbone = VisualBackbone(
            d_model=m.d_model, n_heads=m.n_heads, ffn_dim=m.ffn_dim, n_layers=m.backbone_layers,
            dropout=m.backbone_dropout, cnn_dropout=m.visual_cnn_dropout, cnn_freeze_layers=m.visual_freeze_layers,
        )
        self.text_backbone = TextBackbone(
            pretrained_model=cfg.text_pretrained, d_model=m.d_model,
            dropout=m.text_dropout, freeze_layers=m.text_freeze_layers,
        )

        self.fusion = HierarchicalCrossAttentionFusion(
            d_model=m.d_model, n_heads=m.n_heads, ffn_dim=m.ffn_dim,
            n_layers=1, dropout=m.cross_attn_dropout, drop_path=m.cross_attn_drop_path,
        )

        hybrid_dim = m.d_model * 2  # [z_cross_*, mean(X_*)] concat
        self.prosody_gate = ProsodyGatedFusion(hybrid_dim=hybrid_dim, prosody_dim=m.prosody_dim)
        self.classifier = HybridClassifier(hybrid_dim=hybrid_dim, num_classes=m.num_classes, dropout=m.classifier_dropout)

    def _maybe_drop_modalities(self, mel_spec, prosody_vec, frames, input_ids, attention_mask, waveform=None):
        """설계 v3 §9 강건성: 학습 시 모달리티 드롭아웃.

        배치의 각 샘플에 대해 확률적으로 한 모달리티를 통째로 마스킹한다
        (원본 입력을 0으로 치환 — 오디오/시각은 전부 0, 텍스트는 attention_mask를
        전부 0으로 만들어 사실상 빈 입력으로 취급).

        "오디오" 모달리티는 설계상 멜스펙트로그램 + 운율(prosody) 벡터를 합친
        개념(§3.5)이므로, 오디오를 드롭할 때는 **셋을 전부** 지워야 한다. 하나라도
        남기면 "마이크가 완전히 죽은 상황"을 시뮬레이션하려던 목적이 절반만 작동한다 —
        운율만 남겨두던 버그를 8.6절에서 고쳤고, wav2vec2 백본을 붙이면서 **파형**이
        같은 함정이 됐다(그 경로에선 모델이 멜이 아니라 파형을 읽으므로, 멜만 지우면
        드롭아웃이 아예 무효가 된다).
        """
        if not self.training or self.modality_dropout_prob <= 0.0:
            return mel_spec, prosody_vec, frames, input_ids, attention_mask, waveform

        b = mel_spec.size(0)
        mel_spec, frames = mel_spec.clone(), frames.clone()
        prosody_vec = prosody_vec.clone()
        attention_mask = attention_mask.clone()
        if waveform is not None:
            waveform = waveform.clone()
        for i in range(b):
            if random.random() < self.modality_dropout_prob:
                choice = random.choice(["audio", "visual", "text"])
                if choice == "audio":
                    mel_spec[i].zero_()
                    prosody_vec[i].zero_()
                    if waveform is not None:
                        waveform[i].zero_()
                elif choice == "visual":
                    frames[i].zero_()
                else:
                    attention_mask[i].zero_()
                    attention_mask[i, 0] = 1  # BERT류는 최소 1개 유효 토큰 필요
        return mel_spec, prosody_vec, frames, input_ids, attention_mask, waveform

    def forward(
        self,
        mel_spec: torch.Tensor,       # [B, T_a, n_mels]
        prosody_vec: torch.Tensor,    # [B, prosody_dim]  (p_a)
        frames: torch.Tensor,         # [B, T_v, 3, H, W]
        input_ids: torch.Tensor,      # [B, T_t]
        attention_mask: torch.Tensor,  # [B, T_t]  (1=유효 토큰, 0=패딩)
        audio_padding_mask: torch.Tensor | None = None,  # [B, T_a] True=패딩(배치 내 가변 길이 대비)
        visual_padding_mask: torch.Tensor | None = None,  # [B, T_v] True=패딩
        waveform: torch.Tensor | None = None,             # wav2vec2 백본일 때만 사용
        wav_attention_mask: torch.Tensor | None = None,   # [B, T_samples] 1=유효
    ) -> torch.Tensor:
        mel_spec, prosody_vec, frames, input_ids, attention_mask, waveform = self._maybe_drop_modalities(
            mel_spec, prosody_vec, frames, input_ids, attention_mask, waveform
        )

        if self.use_w2v:
            if waveform is None:
                raise ValueError(
                    "wav2vec2 백본은 원본 파형이 필요하다 — "
                    "ManifestEmotionDataset(return_waveform=True)로 만들었는지 확인할 것"
                )
            x_a = self.audio_backbone(waveform, wav_attention_mask=wav_attention_mask)
            # wav2vec2는 자체 stride로 길이를 줄이므로 멜 기준 마스크를 쓸 수 없다.
            audio_padding_mask = (
                self.audio_backbone.frame_padding_mask(wav_attention_mask, x_a.size(1))
                if wav_attention_mask is not None else None
            )
        else:
            x_a = self.audio_backbone(mel_spec, key_padding_mask=audio_padding_mask)  # [B, T_a, d_model]
        x_v = self.visual_backbone(frames, key_padding_mask=visual_padding_mask)       # [B, T_v, d_model]
        x_t = self.text_backbone(input_ids, attention_mask)                            # [B, T_t, d_model]

        t_key_padding_mask = attention_mask == 0  # True=패딩(무시할 위치)

        z_cross_v, z_cross_a, z_cross_t = self.fusion(
            x_v, x_a, x_t,
            v_mask=visual_padding_mask, a_mask=audio_padding_mask, t_mask=t_key_padding_mask,
        )

        # 하이브리드 결합: 교차 어텐션(미세 타이밍) + 단일 모달리티 평균(거시 분위기)
        z_v_final = torch.cat([z_cross_v, mean_pool(x_v, visual_padding_mask)], dim=-1)
        z_t_final = torch.cat([z_cross_t, mean_pool(x_t, t_key_padding_mask)], dim=-1)
        z_audio_hybrid = torch.cat([z_cross_a, mean_pool(x_a, audio_padding_mask)], dim=-1)
        z_audio_final = self.prosody_gate(z_audio_hybrid, prosody_vec)

        logits = self.classifier(z_v_final, z_audio_final, z_t_final)
        return logits
