"""설계 v3 §4 ③: Visual Backbone.

입력은 얼굴 크롭 프레임 시퀀스 [B, T_v, 3, H, W] (ABAW 전처리 관례에 따라
112x112로 정렬·크롭되어 있다고 가정, config.visual.face_size).
프레임별로 경량 CNN을 적용해 임베딩을 얻은 뒤(TimeDistributed 방식),
TemporalConvFrontend에 통과시켜 X_v 시퀀스를 만든다.

torchvision 사전학습 가중치 의존을 피하기 위해 처음부터 학습하는 소형 CNN을
사용한다 — 실데이터 확보 후 정확도가 부족하면 사전학습 백본(예: ResNet18,
VGGFACE)으로 교체하는 것을 권장한다(한계는 설계 문서 10절 참고).
"""
import torch
import torch.nn as nn

from .common import TemporalConvFrontend


class FrameCNN(nn.Module):
    """단일 프레임 [3,H,W] -> 임베딩 [feat_dim] 소형 CNN."""

    def __init__(self, feat_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        # 사전학습 가중치 없이 처음부터 학습하는 CNN인데 규제가 전혀 없었음(BatchNorm뿐) ->
        # 과적합 대응으로 conv 블록 사이에 Dropout2d(채널 단위 드롭) 추가
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),   # 112->56
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),  # 56->28
            nn.Dropout2d(dropout),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),  # 28->14
            nn.Dropout2d(dropout),
            nn.Conv2d(128, feat_dim, 3, stride=2, padding=1), nn.BatchNorm2d(feat_dim), nn.ReLU(),  # 14->7
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 3, H, W] -> [N, feat_dim]
        h = self.conv(x)
        h = self.pool(h).flatten(1)
        return h


class VisualBackbone(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, ffn_dim: int, n_layers: int, frame_feat_dim: int = 256,
        dropout: float = 0.1, cnn_dropout: float = 0.0,
    ):
        super().__init__()
        self.frame_cnn = FrameCNN(feat_dim=frame_feat_dim, dropout=cnn_dropout)
        self.frontend = TemporalConvFrontend(
            in_dim=frame_feat_dim, d_model=d_model, n_heads=n_heads, ffn_dim=ffn_dim, n_layers=n_layers, dropout=dropout
        )

    def forward(self, frames: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # frames: [B, T_v, 3, H, W]
        b, t, c, h, w = frames.shape
        flat = frames.reshape(b * t, c, h, w)
        feats = self.frame_cnn(flat).reshape(b, t, -1)  # [B, T_v, frame_feat_dim]
        return self.frontend(feats, key_padding_mask=key_padding_mask)  # X_v: [B, T_v, d_model]
