"""설계 v3 §4 ③: Visual Backbone.

입력은 얼굴 크롭 프레임 시퀀스 [B, T_v, 3, H, W] (ABAW 전처리 관례에 따라
112x112로 정렬·크롭되어 있다고 가정, config.visual.face_size).
프레임별로 CNN을 적용해 임베딩을 얻은 뒤(TimeDistributed 방식),
TemporalConvFrontend에 통과시켜 X_v 시퀀스를 만든다.

이 백본은 두 번 바뀌었다:
1) 처음엔 사전학습 없이 처음부터 학습하는 소형 CNN — 8.7절 베이스라인 결과
   영상 단독 val_acc 24.03%로 다수 클래스(혐오 23.86%) 수준에 그침.
2) ImageNet 사전학습 MobileNetV3-Small로 교체 — 그런데도 개선이 없었고
   오히려 더 빨리 과적합했다. 원인으로 의심되는 건 해상도 불일치(MobileNetV3는
   224 사전학습인데 우리 크롭은 112)와, ImageNet이 일반 사물 사진이라 얼굴
   도메인과 거리가 있다는 점.
3) 그래서 **얼굴 인식 전용 사전학습 백본(MobileFaceNet, emotiefflib 패키지의
   mbf_va_mtl)**으로 다시 교체했다. 이 체크포인트는 애초에 112x112로
   사전학습돼 있어(우리 크롭 크기와 정확히 일치, 리사이즈 자체가 불필요)
   해상도 불일치 문제가 원천적으로 없고, ImageNet이 아니라 얼굴 데이터로
   학습돼 도메인도 훨씬 가깝다. 게다가 밸런스-각성(valence-arousal) 감정
   회귀를 멀티태스크로 학습한 체크포인트라 감정 관련 특징을 이미 담고 있다.
   출처: https://github.com/HSE-asavchenko/face-emotion-recognition (Savchenko et al.)
"""
import torch
import torch.nn as nn
from emotiefflib.facial_analysis import EmotiEffLibRecognizerTorch

from .common import TemporalConvFrontend

# emotiefflib의 mbf_va_mtl 전처리 스펙(ImageNet mean/std가 아니라 -1~1 정규화) —
# facial_analysis.py의 _preprocess와 반드시 일치시켜야 사전학습 가중치가 의도대로 작동한다.
MBF_MEAN = (0.5, 0.5, 0.5)
MBF_STD = (0.5, 0.5, 0.5)
MBF_NATIVE_DIM = 512  # mbf_va_mtl 백본이 내는 임베딩 차원


class FrameCNN(nn.Module):
    """단일 프레임 [3,H,W] -> 임베딩 [feat_dim]. 얼굴인식 사전학습 MobileFaceNet 기반.

    입력은 datasets/manifest_dataset.py의 load_face_frames()가 만든 0~1 정규화
    픽셀이다 — 여기서 mbf_va_mtl 전용 정규화(mean=std=0.5, 즉 [-1,1] 범위)로
    한 번 더 변환한다. 원본 크롭이 이미 112x112라 별도 리사이즈가 필요 없다.
    """

    def __init__(self, feat_dim: int = 256, dropout: float = 0.0, freeze_layers: int = 9):
        super().__init__()
        # device="cpu"로 받아둔 뒤 전체 모델(TrimodalEmotionModel)의 .to(device) 호출 때
        # 같이 옮겨진다 — 여기서 GPU를 미리 요구할 필요 없음.
        recognizer = EmotiEffLibRecognizerTorch(model_name="mbf_va_mtl", device="cpu")
        self.backbone = recognizer.model  # Sequential(MobileFaceNet, Identity) -> [N, 512]

        # freeze_layers는 다른 백본(BERT 등)과 필드 이름을 맞추려고 int로 뒀지만,
        # MobileFaceNet은 BERT의 encoder layer처럼 깔끔하게 N등분할 구조가 아니라서
        # 여기서는 "0보다 크면 백본 전체 동결, 0이면 전부 미세조정"으로 단순화했다.
        # 처음 시도(사전학습 없음/ImageNet)가 둘 다 과적합으로 실패했으므로, 우선은
        # 안전하게 백본을 통째로 얼리고 위에 얹는 작은 projection만 학습해 이 임베딩
        # 자체가 얼마나 쓸모 있는지부터 확인한다.
        self.freeze_backbone = freeze_layers > 0
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.register_buffer("mean", torch.tensor(MBF_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(MBF_STD).view(1, 3, 1, 1))

        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(MBF_NATIVE_DIM, feat_dim)
        self.feat_dim = feat_dim

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            # BatchNorm이 우리 데이터의 배치 통계로 흘러가지 않도록, 동결 시엔
            # 상위 모델이 train()이어도 백본만 강제로 eval() 유지(고정된 러닝 통계 사용).
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 3, H, W], 0~1 범위 -> mbf_va_mtl 전용 정규화 -> 사전학습 얼굴 특징
        x = (x - self.mean) / self.std
        h = self.backbone(x)  # [N, 512]
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
