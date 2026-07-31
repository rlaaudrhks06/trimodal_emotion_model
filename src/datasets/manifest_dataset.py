"""매니페스트(CSV) 기반 트리모달 감정 데이터셋.

KEMDy19/20, AI Hub 데이터는 원본 배포 형태(폴더 구조·파일명 규칙)가 실제로
다운로드해봐야 확정된다. 그래서 원본 포맷에 직접 종속되는 대신, 아래 스키마의
매니페스트 CSV 한 장으로 추상화한다 — 실データ를 받은 뒤
`scripts/build_manifest.py`에서 이 CSV를 생성하도록 연결하면 된다.

매니페스트 CSV 필수 컬럼:
    utt_id            : 발화 고유 ID
    label             : datasets.labels.EMOTION_LABELS 중 하나
    wav_path          : 오디오 파일 경로 (wav/flac 등 soundfile이 읽을 수 있는 포맷)
    text              : STT 전사문 (혹은 데이터셋 제공 정답 전사문)
    face_frames_dir   : 발화 구간에 대응하는 얼굴 크롭 프레임(jpg/png) 디렉터리.
                        파일명 오름차순 정렬이 곧 시간 순서라고 가정.
"""
from pathlib import Path

import cv2
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from ..config import Config
from ..features.audio_frontend import waveform_to_mel
from ..features.prosody import extract_prosody
from .labels import LABEL_TO_IDX

REQUIRED_COLUMNS = ["utt_id", "label", "wav_path", "text", "face_frames_dir"]


def load_face_frames(frames_dir: str, face_size: int, max_frames: int = 32) -> np.ndarray:
    """얼굴 크롭 프레임 디렉터리 -> [T_v, 3, H, W] float32 (0~1 정규화).

    프레임 수가 max_frames보다 많으면 균등 샘플링, 적으면 그대로 둔다
    (배치 결합 시 collate_fn에서 0-패딩).
    """
    paths = sorted(Path(frames_dir).glob("*"))
    if len(paths) == 0:
        raise FileNotFoundError(f"얼굴 프레임을 찾을 수 없음: {frames_dir}")

    if len(paths) > max_frames:
        idx = np.linspace(0, len(paths) - 1, max_frames).astype(int)
        paths = [paths[i] for i in idx]

    frames = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.resize(img, (face_size, face_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frames.append(img.transpose(2, 0, 1))  # HWC -> CHW

    if len(frames) == 0:
        raise FileNotFoundError(f"유효한 이미지 프레임이 없음: {frames_dir}")

    return np.stack(frames, axis=0)  # [T_v, 3, H, W]


class ManifestEmotionDataset(Dataset):
    """melspectrogram/운율/얼굴 프레임은 wav·이미지 파일 내용이 바뀌지 않는 한 항상
    같은 값이 나오는 순수 계산이다. 그런데도 매 에폭·매 실행마다 원본에서 다시
    계산하고 있었던 게 실측 결과 가장 큰 병목이었다(운율 추출의 librosa.pyin이
    특히 느림). cache_dir을 주면 발화(utt_id)별로 한 번 계산한 결과를 .npz로
    저장해두고, 다음 접근부터는 그걸 그대로 불러온다 — 값 자체는 100% 동일하고
    속도만 빨라진다(캐싱이 학습 결과에 영향을 주지 않음).
    """

    def __init__(
        self, manifest_csv: str, cfg: Config, max_audio_seconds: float = 8.0,
        max_video_frames: int = 32, cache_dir: str | Path | None = None,
        prosody_stats_path: str | Path | None = None,
    ):
        import pandas as pd

        self.df = pd.read_csv(manifest_csv)
        missing = [c for c in REQUIRED_COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(f"매니페스트에 필수 컬럼이 없습니다: {missing}")

        self.cfg = cfg
        self.max_audio_seconds = max_audio_seconds
        self.max_video_frames = max_video_frames
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # 데이터 전처리 EDA 점검 문서(§1.2)의 최우선 항목: prosody 10차원은 스케일이
        # 서로 완전히 다른데(f0_mean 수백 vs jitter 0.01대) 지금까지 정규화가 전혀
        # 없었다. prosody_stats_path가 주어지면(= scripts/compute_prosody_stats.py로
        # train 세트에서만 fit한 통계) IQR 클리핑 + z-score 정규화를 적용한다.
        # None이면(기본값) 기존과 완전히 동일하게 동작 — 진행 중인 baseline 학습과의
        # A/B 비교를 위해 하위호환을 유지한다.
        self.prosody_mean = self.prosody_std = self.prosody_clip_lo = self.prosody_clip_hi = None
        if prosody_stats_path is not None:
            import json
            with open(prosody_stats_path, encoding="utf-8") as f:
                stats = json.load(f)
            self.prosody_mean = np.array(stats["mean"], dtype=np.float32)
            self.prosody_std = np.array(stats["std"], dtype=np.float32)
            self.prosody_clip_lo = np.array(stats["clip_lower"], dtype=np.float32)
            self.prosody_clip_hi = np.array(stats["clip_upper"], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def _compute_features(self, row) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y, sr = librosa.load(
            row.wav_path, sr=self.cfg.audio_sample_rate, mono=True,
            duration=self.max_audio_seconds,
        )
        mel = waveform_to_mel(
            y, sr, n_mels=self.cfg.audio_n_mels,
            n_fft=self.cfg.audio_n_fft, hop_length=self.cfg.audio_hop_length,
        )  # [T_a, n_mels]
        prosody = extract_prosody(y, sr)  # [prosody_dim]
        frames = load_face_frames(
            row.face_frames_dir, face_size=self.cfg.visual_face_size, max_frames=self.max_video_frames
        )  # [T_v, 3, H, W]
        return mel, prosody, frames

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        utt_id = str(row.utt_id)

        cache_path = self.cache_dir / f"{utt_id}.npz" if self.cache_dir else None
        if cache_path is not None and cache_path.exists():
            cached = np.load(cache_path)
            mel, prosody, frames = cached["mel"], cached["prosody"], cached["frames"]
        else:
            mel, prosody, frames = self._compute_features(row)
            if cache_path is not None:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                # 중간에 프로세스가 죽어도 캐시 파일이 반쯤 쓰인 채로 남지 않도록
                # 임시 파일에 먼저 쓰고 마지막에 원자적으로 이름을 바꾼다.
                # np.savez는 파일명이 .npz로 안 끝나면 자기가 .npz를 덧붙여버리므로,
                # 임시 파일명도 반드시 .npz로 끝나야 한다(안 그러면 rename 대상이 없어서 에러남).
                tmp_path = cache_path.with_name(cache_path.stem + ".tmp.npz")
                np.savez(str(tmp_path), mel=mel, prosody=prosody, frames=frames)
                tmp_path.replace(cache_path)

        if self.prosody_mean is not None:
            # 캐시에는 항상 원본(raw) prosody를 저장하고, 정규화는 읽을 때마다 적용한다
            # -> stats를 나중에 바꿔도 캐시를 무효화할 필요가 없다.
            prosody = np.clip(prosody, self.prosody_clip_lo, self.prosody_clip_hi)
            prosody = (prosody - self.prosody_mean) / self.prosody_std

        label_idx = LABEL_TO_IDX[str(row.label).strip().lower()]

        return {
            "utt_id": utt_id,
            "mel": mel,
            "prosody": prosody,
            "frames": frames,
            "text": str(row.text),
            "label": label_idx,
        }


def _pad_time(arrays: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """[T_i, ...] 리스트 -> 배치 내 최대 T로 0-패딩. 반환: (tensor, key_padding_mask[True=패딩])"""
    max_t = max(a.shape[0] for a in arrays)
    rest_shape = arrays[0].shape[1:]
    out = np.zeros((len(arrays), max_t, *rest_shape), dtype=np.float32)
    mask = np.ones((len(arrays), max_t), dtype=bool)  # True=패딩
    for i, a in enumerate(arrays):
        t = a.shape[0]
        out[i, :t] = a
        mask[i, :t] = False
    return torch.from_numpy(out), torch.from_numpy(mask)


class CollateFn:
    """클로저 대신 모듈 레벨 클래스로 구현 — DataLoader(num_workers>0)가 워커 프로세스를
    spawn 방식으로 띄울 때(macOS 기본값, CUDA 환경에서도 권장되는 방식) pickle이 가능해야
    하는데, 중첩 함수(클로저)는 pickle이 안 돼서 워커가 시작조차 못 하는 문제가 있었다.
    """

    def __init__(self, tokenizer_name: str, max_text_len: int = 64):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_text_len = max_text_len

    def __call__(self, batch: list[dict]) -> dict:
        mel_tensor, audio_mask = _pad_time([b["mel"] for b in batch])
        frames_tensor, visual_mask = _pad_time([b["frames"] for b in batch])
        prosody_tensor = torch.from_numpy(np.stack([b["prosody"] for b in batch]))
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

        text_enc = self.tokenizer(
            [b["text"] for b in batch], padding=True, truncation=True,
            max_length=self.max_text_len, return_tensors="pt",
        )

        return {
            "utt_ids": [b["utt_id"] for b in batch],
            "mel_spec": mel_tensor,
            "audio_padding_mask": audio_mask,
            "prosody_vec": prosody_tensor,
            "frames": frames_tensor,
            "visual_padding_mask": visual_mask,
            "input_ids": text_enc["input_ids"],
            "attention_mask": text_enc["attention_mask"],
            "labels": labels,
        }


def make_collate_fn(tokenizer_name: str, max_text_len: int = 64) -> CollateFn:
    return CollateFn(tokenizer_name, max_text_len)
