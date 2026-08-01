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
    num_classes: int = 8
    prosody_dim: int = 10
    # 과적합 대응(2차 학습에서 추가): 각 모듈별 규제 강도를 config.yaml에서 조절 가능하게 함
    backbone_dropout: float = 0.1       # 오디오/비주얼 TemporalConvFrontend
    visual_cnn_dropout: float = 0.0     # FrameCNN conv 블록 사이 Dropout2d
    classifier_dropout: float = 0.2     # HybridClassifier 최종 MLP
    text_dropout: float = 0.1           # BERT hidden/attention dropout override
    text_freeze_layers: int = 0         # BERT 하위 N개 encoder layer(+embeddings) 동결


@dataclass
class Config:
    model: ModelConfig
    text_pretrained: str
    audio_sample_rate: int
    audio_n_mels: int
    audio_n_fft: int
    audio_hop_length: int
    visual_face_size: int
    labels: list
    raw: dict = field(default_factory=dict)  # yaml 원본 전체 (train 섹션 등 접근용)


def load_config(path: Path = CONFIG_PATH) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    model_cfg = ModelConfig(**raw["model"])
    return Config(
        model=model_cfg,
        text_pretrained=raw["text"]["pretrained_model"],
        audio_sample_rate=raw["audio"]["sample_rate"],
        audio_n_mels=raw["audio"]["n_mels"],
        audio_n_fft=raw["audio"]["n_fft"],
        audio_hop_length=raw["audio"]["hop_length"],
        visual_face_size=raw["visual"]["face_size"],
        labels=raw["labels"],
        raw=raw,
    )
