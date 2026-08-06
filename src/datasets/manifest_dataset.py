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
from .labels import LABEL_TO_IDX, normalize_label

REQUIRED_COLUMNS = ["utt_id", "label", "wav_path", "text", "face_frames_dir"]

# v12 보조 라벨(11.2절): AI Hub 원본은 발화마다 감정 라벨을 네 개 갖고 있고,
# 우리가 정답으로 쓰는 multimodal 라벨과 나머지 셋의 일치율이 크게 다르다
# (소리 77.35% / 영상 41.69% / 텍스트 30.87%). 각 브랜치가 "자기 입력에 답이 있는"
# 과제를 함께 풀도록 보조 라벨을 실어 나른다. scripts/add_modality_labels.py가 채운다.
#
# 매니페스트 컬럼명은 AI Hub 원본 표기(image/sound/text)를 따르고, 배치 키는 모델의
# 브랜치 이름(visual/audio/text)을 따른다 — 모델 쪽에서 어느 브랜치용인지 헷갈리지 않게.
AUX_LABEL_COLUMNS = {
    "label_image": "aux_visual",
    "label_sound": "aux_audio",
    "label_text": "aux_text",
}

# CrossEntropyLoss가 무시하는 기본 인덱스. 보조 라벨이 비어 있거나(원본 미발견)
# 우리 7클래스 체계로 정규화되지 않는 값이면 이걸 넣어, 그 표본만 보조 손실에서
# 자동으로 빠지게 한다 — 호출부에서 마스킹을 따로 구현할 필요가 없다.
IGNORE_INDEX = -100


def load_face_frames(frames_dir: str, face_size: int, max_frames: int = 32) -> np.ndarray:
    """얼굴 크롭 프레임 디렉터리 -> [T_v, 3, H, W] **uint8**(0~255).

    프레임 수가 max_frames보다 많으면 균등 샘플링, 적으면 그대로 둔다
    (배치 결합 시 collate_fn에서 0-패딩).

    0~1 float32 변환은 여기서 하지 않고 __getitem__에서 한다 — 캐시에 float32로
    저장하면 픽셀 하나가 4바이트가 되어 캐시가 4배로 부푼다(실측 303GB 중 93%가
    이 프레임이었다). uint8로 저장하고 읽을 때 변환하면 값은 완전히 동일하면서
    캐시가 약 201GB 줄어든다.
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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img.transpose(2, 0, 1))  # HWC -> CHW

    if len(frames) == 0:
        raise FileNotFoundError(f"유효한 이미지 프레임이 없음: {frames_dir}")

    return np.stack(frames, axis=0)  # [T_v, 3, H, W] uint8


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
        prosody_stats_path: str | Path | None = None, return_waveform: bool = False,
    ):
        import pandas as pd

        self.df = pd.read_csv(manifest_csv)
        missing = [c for c in REQUIRED_COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(f"매니페스트에 필수 컬럼이 없습니다: {missing}")

        # 보조 라벨 컬럼은 있으면 쓰고 없으면 그냥 지나간다 — 컬럼이 없는 기존
        # 매니페스트(v1~v11)로는 배치에 aux_* 키 자체가 안 생겨 동작이 완전히 동일하다.
        self.aux_columns = {c: k for c, k in AUX_LABEL_COLUMNS.items() if c in self.df.columns}

        self.cfg = cfg
        self.max_audio_seconds = max_audio_seconds
        self.max_video_frames = max_video_frames
        self.cache_dir = Path(cache_dir) if cache_dir else None
        # wav2vec2 백본은 멜스펙트로그램이 아니라 원본 파형을 받는다.
        # 캐시에 넣지 않고 매번 읽는 이유: build_manifest가 이미 16kHz mono로 저장해둬서
        # soundfile로 바로 읽으면 리샘플링이 없어 충분히 빠르고, 파형까지 캐시하면
        # 발화당 256KB가 더 늘어난다. 층 선택을 바꿔가며 실험하기에도 이 편이 유연하다.
        self.return_waveform = return_waveform

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

        # 프레임은 uint8(0~255)로 저장/전달되므로 여기서 0~1 float32로 변환한다.
        # 마이그레이션 도중에는 옛 캐시(float32, 이미 0~1)가 섞여 있을 수 있어 dtype으로 분기 —
        # 둘 다 최종적으로 동일한 값이 된다.
        if frames.dtype == np.uint8:
            frames = frames.astype(np.float32) / 255.0

        if self.prosody_mean is not None:
            # 캐시에는 항상 원본(raw) prosody를 저장하고, 정규화는 읽을 때마다 적용한다
            # -> stats를 나중에 바꿔도 캐시를 무효화할 필요가 없다.
            prosody = np.clip(prosody, self.prosody_clip_lo, self.prosody_clip_hi)
            prosody = (prosody - self.prosody_mean) / self.prosody_std

        # normalize_label: 기존 매니페스트 CSV엔 "contempt" 문자열이 그대로 남아있으므로
        # (8->7클래스 병합, src/datasets/labels.py 참고) 여기서 흡수한다 — CSV 자체는 안 건드림.
        label_idx = LABEL_TO_IDX[normalize_label(str(row.label).strip().lower())]

        item = {
            "utt_id": utt_id,
            "mel": mel,
            "prosody": prosody,
            "frames": frames,
            "text": str(row.text),
            "label": label_idx,
        }
        for col, key in self.aux_columns.items():
            # 빈 값(원본 미발견)이나 우리 체계 밖의 값은 IGNORE_INDEX로 둔다.
            # pandas는 빈 칸을 NaN(float)으로 읽으므로 문자열 변환 후 판정해야 한다.
            raw = str(row[col]).strip().lower()
            item[key] = LABEL_TO_IDX.get(normalize_label(raw), IGNORE_INDEX) if raw and raw != "nan" else IGNORE_INDEX
        if self.return_waveform:
            import soundfile as sf
            wav, sr = sf.read(row.wav_path, dtype="float32", always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            # build_manifest_aihub.py가 16kHz로 저장하므로 리샘플링은 필요 없지만,
            # 다른 경로로 만들어진 데이터가 섞이면 조용히 틀리므로 확인하고 막는다.
            if sr != self.cfg.audio_sample_rate:
                raise ValueError(
                    f"{row.wav_path}: 샘플레이트가 {sr}Hz인데 config는 "
                    f"{self.cfg.audio_sample_rate}Hz를 기대함"
                )
            item["waveform"] = wav[: int(self.max_audio_seconds * sr)]
        return item


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

        out_extra = {}
        if "waveform" in batch[0]:
            # 파형은 [T] 1차원이라 _pad_time([T,...] 가정)을 그대로 못 쓴다.
            # wav2vec2의 attention_mask는 1=유효(멜 쪽 key_padding_mask와 반대 규약)이므로
            # 이름을 wav_attention_mask로 구분해 혼동을 막는다.
            wavs = [b["waveform"] for b in batch]
            max_len = max(len(w) for w in wavs)
            wav_arr = np.zeros((len(wavs), max_len), dtype=np.float32)
            wav_mask = np.zeros((len(wavs), max_len), dtype=np.int64)
            for i, w in enumerate(wavs):
                wav_arr[i, : len(w)] = w
                wav_mask[i, : len(w)] = 1
            out_extra["waveform"] = torch.from_numpy(wav_arr)
            out_extra["wav_attention_mask"] = torch.from_numpy(wav_mask)

        # v12 보조 라벨. 데이터셋이 실어줬을 때만 배치에 들어간다("waveform"과 같은 규약).
        # 값이 IGNORE_INDEX인 표본은 CrossEntropyLoss가 알아서 건너뛴다.
        for key in AUX_LABEL_COLUMNS.values():
            if key in batch[0]:
                out_extra[key] = torch.tensor([b[key] for b in batch], dtype=torch.long)

        return {
            **out_extra,
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
