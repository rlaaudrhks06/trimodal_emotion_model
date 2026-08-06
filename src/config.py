"""configs/config.yaml을 로드해 각 모듈에서 쓰는 설정 객체로 변환합니다."""
import yaml
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"


@dataclass
class ModelConfig:
    d_model: int = 256
    n_heads: int = 8
    ffn_dim: int = 1024
    backbone_layers: int = 2
    cross_attn_dropout: float = 0.1
    # 과적합 대응(§11.2): 크로스어텐션 블록의 residual 분기를 확률적으로 건너뛰는
    # stochastic depth. 0.0(기본)이면 항등이라 v1~v8과 완전히 동일하게 동작한다.
    cross_attn_drop_path: float = 0.0
    num_classes: int = 8
    prosody_dim: int = 10
    # 과적합 대응(2차 학습에서 추가): 각 모듈별 규제 강도를 config.yaml에서 조절 가능하게 함
    backbone_dropout: float = 0.1       # 오디오/비주얼 TemporalConvFrontend
    visual_cnn_dropout: float = 0.0     # FrameCNN(MobileFaceNet) 백본 출력 후 dropout
    classifier_dropout: float = 0.2     # HybridClassifier 최종 MLP
    text_dropout: float = 0.1           # BERT hidden/attention dropout override
    text_freeze_layers: int = 0         # BERT 하위 N개 encoder layer(+embeddings) 동결
    visual_freeze_layers: int = 9       # 0보다 크면 MobileFaceNet 백본 전체 동결, 0이면 미세조정(8.7~8.8절 대응)
    # v12 보조 헤드(11.2절)의 은닉 차원. 0(기본)이면 헤드를 아예 만들지 않아
    # v1~v11과 파라미터 수까지 완전히 동일하다. 보조 라벨 학습을 켤 때만 값을 준다.
    aux_head_dim: int = 0


@dataclass
class Config:
    model: ModelConfig
    text_pretrained: str
    # 오디오 백본 선택(11.0.1절): "mel"=기존 멜스펙트로그램+자체학습 프론트엔드,
    # "wav2vec2"=사전학습 SSL 모델에서 특징 추출. wav2vec2일 때만 아래 세 필드를 쓴다.
    audio_backbone: str
    audio_pretrained: str
    audio_w2v_layer: int
    audio_w2v_freeze: bool
    audio_sample_rate: int
    audio_n_mels: int
    audio_n_fft: int
    audio_hop_length: int
    visual_face_size: int
    raw: dict = field(default_factory=dict)  # yaml 원본 전체 (train 섹션 등 접근용)


def load_config(path: Path = CONFIG_PATH) -> Config:
    """config.yaml -> Config.

    주의: 라벨 체계의 단일 출처는 `src/datasets/labels.py`의 EMOTION_LABELS다.
    예전엔 config에도 labels: 목록이 있었지만 어느 코드도 읽지 않았고, 실제로
    3개 config가 8클래스 목록을 그대로 들고 있어 코드(7클래스)와 모순된 상태였다.
    혼동의 소지만 있어 제거했다 — num_classes만 labels.py와 맞으면 된다.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    model_cfg = ModelConfig(**raw["model"])
    audio_raw = raw["audio"]
    return Config(
        model=model_cfg,
        text_pretrained=raw["text"]["pretrained_model"],
        audio_backbone=audio_raw.get("backbone", "mel"),
        audio_pretrained=audio_raw.get("pretrained_model", "facebook/wav2vec2-large-xlsr-53"),
        audio_w2v_layer=audio_raw.get("w2v_layer", 12),
        audio_w2v_freeze=audio_raw.get("w2v_freeze", True),
        audio_sample_rate=audio_raw["sample_rate"],
        audio_n_mels=raw["audio"]["n_mels"],
        audio_n_fft=raw["audio"]["n_fft"],
        audio_hop_length=raw["audio"]["hop_length"],
        visual_face_size=raw["visual"]["face_size"],
        raw=raw,
    )
