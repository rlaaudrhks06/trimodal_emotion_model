"""실시간 입력을 v11이 학습 때 본 것과 **똑같은 형태**로 만든다.

이 모듈의 존재 이유가 전부 여기에 있다. 학습 전처리와 추론 전처리가 조금이라도
어긋나면 모델은 조용히 엉뚱한 답을 낸다 — 이 프로젝트에서 이미 겪은 종류의 사고다
(XLSR 입력 정규화 누락, 얼굴 크롭이 사실 전신이었던 것, 패딩 포함 정규화).

그래서 새로 구현하지 않고 **학습에 쓴 코드를 그대로 import해서 쓴다**:
  - 얼굴: `src.features.face_align.detect_and_align_face` (mediapipe 검출+정렬)
  - 운율: `src.features.prosody.extract_prosody`
  - 멜:   `src.features.audio_frontend.waveform_to_mel`
프레임 규약도 `manifest_dataset.load_face_frames`와 맞춘다 — BGR→RGB, CHW, /255.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.audio_frontend import waveform_to_mel          # noqa: E402
from src.features.face_align import (                            # noqa: E402
    create_face_detector, detect_and_align_face, ensure_face_detector_model,
)
from src.features.prosody import extract_prosody                 # noqa: E402


class FaceCropper:
    """웹캠 프레임 -> 정렬된 얼굴 크롭 [face_size, face_size, 3] BGR.

    학습 때는 AI Hub가 준 person bbox 안에서 얼굴을 찾았다. 실시간에는 person bbox가
    없으므로 **프레임 전체를 person 영역으로 준다** — `detect_and_align_face`가 그
    안에서 mediapipe로 얼굴을 찾으므로 결과는 같은 성질의 크롭이 된다.
    """

    def __init__(self, face_size: int = 112, min_confidence: float = 0.5):
        self.face_size = face_size
        model_path = ensure_face_detector_model()
        self.detector = create_face_detector(model_path, min_confidence)

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        h, w = frame_bgr.shape[:2]
        return detect_and_align_face(
            frame_bgr, (0, 0, w, h), face_size=self.face_size, detector=self.detector
        )


def frames_to_tensor(faces_bgr: list[np.ndarray], max_frames: int = 32) -> np.ndarray:
    """얼굴 크롭 리스트 -> [T_v, 3, H, W] float32 0~1 (RGB).

    `manifest_dataset.load_face_frames` + `__getitem__`의 변환을 그대로 따른다.
    프레임이 max_frames보다 많으면 균등 샘플링하는 것까지 동일하다.
    """
    if not faces_bgr:
        raise ValueError("얼굴 프레임이 하나도 없다")
    if len(faces_bgr) > max_frames:
        idx = np.linspace(0, len(faces_bgr) - 1, max_frames).astype(int)
        faces_bgr = [faces_bgr[i] for i in idx]

    out = []
    for img in faces_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out.append(img.transpose(2, 0, 1))          # HWC -> CHW
    return np.stack(out).astype(np.float32) / 255.0


def build_batch(
    wav: np.ndarray,
    faces_bgr: list[np.ndarray],
    text: str,
    tokenizer,
    cfg,
    device: torch.device,
    prosody_stats: dict | None = None,
    max_text_len: int = 64,
) -> dict:
    """발화 1건 -> 모델이 바로 먹을 수 있는 배치(크기 1).

    `collate_fn`이 하는 일을 배치 크기 1에 맞춰 옮긴 것이다. 배치가 하나뿐이라
    패딩이 없으므로 마스크는 전부 유효(오디오/시각은 False=유효 규약).
    """
    wav = np.asarray(wav, dtype=np.float32)
    sr = cfg.audio_sample_rate

    mel = waveform_to_mel(wav, sr, n_mels=cfg.audio_n_mels,
                          n_fft=cfg.audio_n_fft, hop_length=cfg.audio_hop_length)
    prosody = extract_prosody(wav, sr)
    if prosody_stats is not None:
        # 학습에서 prosody_stats_path를 준 config라면 같은 통계로 정규화해야 한다.
        lo = np.asarray(prosody_stats["clip_lower"], dtype=np.float32)
        hi = np.asarray(prosody_stats["clip_upper"], dtype=np.float32)
        mu = np.asarray(prosody_stats["mean"], dtype=np.float32)
        sd = np.asarray(prosody_stats["std"], dtype=np.float32)
        prosody = (np.clip(prosody, lo, hi) - mu) / np.where(sd == 0, 1.0, sd)

    frames = frames_to_tensor(faces_bgr)

    enc = tokenizer([text], padding=True, truncation=True,
                    max_length=max_text_len, return_tensors="pt")

    t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32)).unsqueeze(0)
    batch = {
        "mel_spec": t(mel),
        "prosody_vec": t(prosody),
        "frames": t(frames),
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "audio_padding_mask": torch.zeros(1, mel.shape[0], dtype=torch.bool),
        "visual_padding_mask": torch.zeros(1, frames.shape[0], dtype=torch.bool),
    }
    if cfg.audio_backbone == "wav2vec2":
        batch["waveform"] = torch.from_numpy(wav).unsqueeze(0)
        # wav2vec2의 attention_mask는 1=유효 규약(멜 쪽 padding_mask와 반대다).
        batch["wav_attention_mask"] = torch.ones(1, wav.shape[0], dtype=torch.long)

    return {k: v.to(device) for k, v in batch.items()}
