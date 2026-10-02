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
import torch.nn.functional as F
from .third_party.efficientface import LocalFeatureExtractor as _EFLocal
from emotiefflib.facial_analysis import EmotiEffLibRecognizerTorch

from .common import TemporalConvFrontend

# emotiefflib의 mbf_va_mtl 전처리 스펙(ImageNet mean/std가 아니라 -1~1 정규화) —
# facial_analysis.py의 _preprocess와 반드시 일치시켜야 사전학습 가중치가 의도대로 작동한다.
MBF_MEAN = (0.5, 0.5, 0.5)
MBF_STD = (0.5, 0.5, 0.5)
MBF_NATIVE_DIM = 512  # mbf_va_mtl 백본이 내는 임베딩 차원


# EfficientFace가 AffectNet으로 사전학습될 때 쓴 정규화 상수(원저장소 main.py:96~97).
EF_MEAN = (0.57535914, 0.44928582, 0.40079932)
EF_STD = (0.20735591, 0.18981615, 0.18132027)


class _DynamicLocalFeatureExtractor(_EFLocal):
    """원본 LocalFeatureExtractor의 사분면 분할을 입력 크기에 맞춰 일반화한 것.

    원본은 `x[:, :, 0:28, 0:28]`처럼 56x56 특징맵(=224 입력)을 전제로 좌표가 박혀 있다.
    112 입력이면 특징맵이 28x28이라 두 번째 패치가 빈 텐서가 되어 conv가 죽는다.
    파라미터(모두 depthwise conv)는 해상도와 무관하므로 사전학습 가중치를 그대로 쓴다.
    concat 순서는 원본과 동일(세로로 11|21, 12|22를 붙이고 둘을 가로로 붙임).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2] // 2, x.shape[3] // 2
        patches = [x[:, :, :h, :w], x[:, :, h:, :w], x[:, :, :h, w:], x[:, :, h:, w:]]
        outs = []
        for i, patch in enumerate(patches, start=1):
            y = self.relu(getattr(self, f"bn{i}_1")(getattr(self, f"conv{i}_1")(patch)))
            outs.append(self.relu(getattr(self, f"bn{i}_2")(getattr(self, f"conv{i}_2")(y))))
        return torch.cat([torch.cat(outs[:2], dim=2), torch.cat(outs[2:], dim=2)], dim=3)


class EfficientFaceFrameCNN(nn.Module):
    """EfficientFace(AffectNet-7 사전학습)의 **공간 부분만** 써서 프레임 하나 -> 임베딩 (13.20절).

    원본(zengqunzhao/EfficientFace, MIT)은 conv5까지가 공간 특징이고 그 뒤는 분류기다.
    우리는 시간축을 교차 어텐션이 처리하므로 공간 특징만 가져와 평균 풀링 후 proj한다.
    **입력 해상도 주의**: 이 체크포인트는 AffectNet 224x224로 사전학습됐고, 원 저장소
    (katerynaCh)도 sample_size=224로 파인튜닝한다. 게다가 LocalFeatureExtractor.forward는
    56x56 특징맵을 전제로 `x[:, :, 28:56, ...]` 슬라이스가 **하드코딩**돼 있어서, 우리 112
    크롭을 그대로 넣으면 특징맵이 28x28이 되어 그 패치가 빈 텐서가 되고 크래시한다.
    그래서 input_size로 둘 중 하나를 고른다:
      · 224(기본) — 112 크롭을 bilinear 업샘플. 사전학습 해상도와 일치. 연산·메모리는 4배.
      · 112      — 사분면 분할을 H//2로 일반화(_DynamicLocalFeatureExtractor). 비용은 v11e와
                   같지만 동결 백본이 절반 스케일 얼굴을 보게 된다.
    어느 쪽이 나은지는 측정 문제다(POSTER++는 업샘플 쪽에서 프로브에 졌다).
    """

    def __init__(self, feat_dim: int = 256, dropout: float = 0.0, freeze_layers: int = 9,
                 weights: str | None = None, input_size: int = 224):
        super().__init__()
        self.input_size = input_size
        from .third_party.efficientface import InvertedResidual
        from .third_party.modulator import Modulator
        repeats, chans = [4, 8, 4], [29, 116, 232, 464, 1024]
        self.conv1 = nn.Sequential(nn.Conv2d(3, chans[0], 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(chans[0]), nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ic = chans[0]
        for name, rep, oc in zip(["stage2", "stage3", "stage4"], repeats, chans[1:]):
            seq = [InvertedResidual(ic, oc, 2)] + [InvertedResidual(oc, oc, 1) for _ in range(rep - 1)]
            setattr(self, name, nn.Sequential(*seq))
            ic = oc
        self.local = (_EFLocal(29, 116, 1) if input_size == 224
                      else _DynamicLocalFeatureExtractor(29, 116, 1))
        self.modulator = Modulator(116)
        self.conv5 = nn.Sequential(nn.Conv2d(ic, chans[-1], 1, 1, 0, bias=False),
                                   nn.BatchNorm2d(chans[-1]), nn.ReLU(inplace=True))
        if weights:
            ck = torch.load(weights, map_location="cpu", weights_only=False)
            sd = ck.get("state_dict", ck)
            sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
            missing, unexpected = self.load_state_dict(sd, strict=False)
            # fc(분류기)는 안 쓰므로 unexpected에 남는 게 정상이다. missing이 있으면 구조가 다른 것.
            if missing:
                raise ValueError(f"EfficientFace 가중치에 없는 키 {len(missing)}개: {missing[:5]}")
            print(f"[EfficientFace] 적재 — 쓰지 않는 키 {len(unexpected)}개(분류기)", flush=True)

        self.freeze_backbone = freeze_layers > 0
        if self.freeze_backbone:
            for p in self.parameters():
                p.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(chans[-1], feat_dim)
        self.feat_dim = feat_dim
        # AffectNet 사전학습 때 쓴 정규화(zengqunzhao/EfficientFace main.py:96). 동결 백본이라
        # 이걸 빼면 가중치가 본 적 없는 분포가 들어간다 — 프로브에서 val 19.71%로
        # 최다 클래스(27.0%)보다도 낮게 나왔다. katerynaCh 저장소는 0~1을 그대로 넣지만
        # 그쪽은 백본을 통째로 파인튜닝하므로 초기 분포 불일치가 학습으로 흡수된다.
        self.register_buffer("mean", torch.tensor(EF_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(EF_STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            for m in (self.conv1, self.stage2, self.stage3, self.stage4, self.local, self.modulator, self.conv5):
                m.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 원본 EfficientFace.forward_features와 같은 경로. 입력은 0~1 픽셀
        # (원 저장소도 ToTensor(norm_value=255)로 0~1을 그대로 넣는다 — mean/std 정규화 없음).
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std       # 원본 순서와 같다(Resize -> ToTensor -> Normalize)
        x = self.conv1(x)
        x = self.maxpool(x)
        x1 = self.modulator(self.stage2(x))
        x2 = self.local(x)
        x = x1 + x2
        x = self.stage4(self.stage3(x))
        x = self.conv5(x)
        h = x.mean([2, 3])                 # 공간 평균 -> [N, 1024]
        return self.proj(self.dropout(h))


class ScratchFrameCNN(nn.Module):
    """사전학습 없이 처음부터 학습하는 소형 CNN (커밋 0020654 판본 그대로 복원).

    왜 되살리는가(13.18절): 이 CNN이 24.03%로 실패했던 것은 **입력이 얼굴이 아니라 사람
    전신**이었을 때의 기록이다(8.8절). 크롭을 고친 뒤 다시 돌린 적이 없으므로
    "MobileFaceNet이 더 낫다"는 비교는 오염된 입력에서의 비교였다. 같은 조건에서 다시 잰다.
    """

    def __init__(self, feat_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),            # 112->56
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),           # 56->28
            nn.Dropout2d(dropout),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),         # 28->14
            nn.Dropout2d(dropout),
            nn.Conv2d(128, feat_dim, 3, stride=2, padding=1), nn.BatchNorm2d(feat_dim), nn.ReLU(),  # 14->7
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.conv(x)).flatten(1)


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
        backbone_type: str = "mobilefacenet", efficientface_weights: str | None = None,
        efficientface_input_size: int = 224,
    ):
        super().__init__()
        if backbone_type not in ("mobilefacenet", "scratch", "efficientface"):
            raise ValueError("model.visual_backbone는 'mobilefacenet'·'scratch'·'efficientface' 중 "
                             f"하나여야 한다, got {backbone_type!r}")
        self.backbone_type = backbone_type
        if backbone_type == "scratch":
            self.frame_cnn = ScratchFrameCNN(feat_dim=frame_feat_dim, dropout=cnn_dropout)
        elif backbone_type == "efficientface":
            self.frame_cnn = EfficientFaceFrameCNN(feat_dim=frame_feat_dim, dropout=cnn_dropout,
                                                   freeze_layers=cnn_freeze_layers, weights=efficientface_weights,
                                                   input_size=efficientface_input_size)
        else:
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
