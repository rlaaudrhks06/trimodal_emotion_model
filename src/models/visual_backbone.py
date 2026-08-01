"""설계 v3 §4 ③: Visual Backbone.

입력은 얼굴 크롭 프레임 시퀀스 [B, T_v, 3, H, W] (ABAW 전처리 관례에 따라
112x112로 정렬·크롭되어 있다고 가정, config.visual.face_size).
프레임별로 CNN을 적용해 임베딩을 얻은 뒤(TimeDistributed 방식),
TemporalConvFrontend에 통과시켜 X_v 시퀀스를 만든다.

8.7절 베이스라인 결과: 사전학습 없이 처음부터 학습하던 소형 CNN은 영상 단독
val_acc 24.03%로 다수 클래스(혐오, 23.86%) 수준에 그쳤다 — 사실상 유의미한
시각 신호를 거의 못 뽑아내고 있었다는 뜻. 그래서 ImageNet 사전학습
MobileNetV3-Small로 교체했다. 얼굴 전용 사전학습(FaceNet 등)은 아니지만,
edge/texture/shape 같은 범용 시각 특징을 이미 갖고 있어 처음부터 학습하는 것보다
훨씬 유리하다. 로봇 실시간 탑재 목표(설계 v3 §3.7 경량화)에도 맞는 경량
아키텍처를 선택했다.

주의: 얼굴 크롭 데이터는 이미 112x112로 캐싱돼 있고(재추출하려면 80k 발화 전체를
몇 시간 걸려 다시 크롭해야 함), MobileNetV3는 원래 224x224로 사전학습됐다.
완전 합성곱 구조(AdaptiveAvgPool로 최종 처리)라 112x112 입력도 동작은 하지만,
사전학습 시점보다 각 단계의 특징 맵이 작아 사전학습 효과가 224 입력만큼 완전하진
않을 수 있다 — 이후 성능이 기대만큼 안 나오면 고려할 지점.
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models

from .common import TemporalConvFrontend

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FrameCNN(nn.Module):
    """단일 프레임 [3,H,W] -> 임베딩 [feat_dim]. ImageNet 사전학습 MobileNetV3-Small 기반.

    입력은 datasets/manifest_dataset.py의 load_face_frames()가 만든 0~1 정규화
    픽셀이다(설계 v3 §7.1.4 참고) — 여기서 ImageNet 평균/표준편차로 한 번 더
    정규화해야 사전학습 가중치가 의도대로 작동한다(§7.1.4에서 "사전학습 백본으로
    교체할 경우 반드시 그 백본의 정규화 방식을 맞춰야 한다"고 이미 언급했던 부분).
    """

    def __init__(self, feat_dim: int = 256, dropout: float = 0.0, freeze_layers: int = 9):
        super().__init__()
        backbone = tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT)
        self.features = backbone.features  # Sequential, 13개 블록, 마지막 채널 576
        native_dim = backbone.classifier[0].in_features  # 576

        # BERT와 같은 패턴(설계 v3 §8.2): 하위 N개 블록은 얼리고 상위 블록만 미세조정 —
        # 데이터 규모(80k)에 비해 사전학습 파라미터가 커서 그대로 전부 학습하면 다시
        # 과적합을 키울 위험이 있다.
        if freeze_layers > 0:
            for block in self.features[:freeze_layers]:
                for p in block.parameters():
                    p.requires_grad = False

        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(native_dim, feat_dim)
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 3, H, W], 0~1 범위 -> ImageNet 정규화 -> 사전학습 특징 추출
        x = (x - self.mean) / self.std
        h = self.features(x)
        h = self.pool(h).flatten(1)
        h = self.dropout(h)
        return self.proj(h)


class VisualBackbone(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, ffn_dim: int, n_layers: int, frame_feat_dim: int = 256,
        dropout: float = 0.1, cnn_dropout: float = 0.0, cnn_freeze_layers: int = 9,
    ):
        super().__init__()
        self.frame_cnn = FrameCNN(feat_dim=frame_feat_dim, dropout=cnn_dropout, freeze_layers=cnn_freeze_layers)
        self.frontend = TemporalConvFrontend(
            in_dim=frame_feat_dim, d_model=d_model, n_heads=n_heads, ffn_dim=ffn_dim, n_layers=n_layers, dropout=dropout
        )

    def forward(self, frames: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # frames: [B, T_v, 3, H, W]
        b, t, c, h, w = frames.shape
        flat = frames.reshape(b * t, c, h, w)
        feats = self.frame_cnn(flat).reshape(b, t, -1)  # [B, T_v, frame_feat_dim]
        return self.frontend(feats, key_padding_mask=key_padding_mask)  # X_v: [B, T_v, d_model]
