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


def add_white_noise(y: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    """지정한 SNR(dB)이 되도록 백색 가우시안 잡음을 더한다. **평가 전용.**

    `SNR_dB = 10·log10(P_signal / P_noise)` 이므로 `P_noise = P_signal / 10^(SNR/10)`.
    신호 전력은 이 발화 전체의 평균 제곱으로 잡는다(패딩 전이라 유효 구간뿐이다).

    seed를 발화 ID에서 유도해 **워커 수·배치 순서와 무관하게 재현**되게 한다.
    전역 난수를 쓰면 num_workers에 따라 결과가 달라져 비교가 깨진다.

    한계: 실제 생활 소음(다른 사람 말소리, 가전, 반향)은 백색 잡음이 아니다.
    이 수치는 "소음에 얼마나 버티는가"의 하한 감을 잡는 용도이지 실환경 수치가 아니다.
    """
    p_signal = float(np.mean(y.astype(np.float64) ** 2))
    if p_signal <= 0:
        return y
    p_noise = p_signal / (10.0 ** (snr_db / 10.0))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, np.sqrt(p_noise), size=y.shape).astype(y.dtype)
    return (y + noise).astype(y.dtype)


def _utt_seed(utt_id: str, base: int = 20260808) -> int:
    """발화 ID -> 결정적 시드. 파이썬 hash()는 실행마다 달라져서 못 쓴다."""
    import hashlib
    h = hashlib.sha256(f"{base}:{utt_id}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")


# 소음 시드는 **여기 하나만** 둔다(13.5절). scripts/precompute_noisy_prosody.py가
# 이걸 import해서 미리 운율을 만들고, 학습·평가 때 같은 함수로 파형을 오염시킨다.
# 사본을 만들면 한쪽만 고쳐져서 운율과 파형이 다른 잡음을 겪는데, 그건 에러 없이
# 조용히 틀린 조건이 된다 — 이 프로젝트가 반복해서 당한 유형이다.
NOISE_SEED_BASE = 20260911
# SNR 추첨용 시드는 잡음 실현용과 분리한다. 같은 base를 쓰면 "어느 SNR을 뽑나"와
# "그 SNR에서 어떤 잡음이 오나"가 같은 해시에서 나와 상관이 생긴다.
NOISE_PICK_BASE = 20260912


def noise_seed(utt_id: str, snr_db: float) -> int:
    """(발화, SNR) -> 잡음 실현 시드.

    SNR을 섞는 이유: 같은 발화라도 SNR마다 **독립적인** 잡음 실현을 준다. 섞지 않으면
    numpy가 같은 표준정규 표본을 뽑고 scale만 곱하므로, SNR 20/10/5가 잡음 모양은
    같고 진폭만 다른 1차원 족(族)이 된다(실측: 비율이 상수 0.31622777).

    `:g`로 포맷하는 이유: config가 정수 10을 주고 argparse가 10.0을 주는데, 그냥
    f-string에 넣으면 "10"과 "10.0"이 서로 다른 시드가 된다. 그러면 사전계산과 학습이
    어긋나고, 그 어긋남은 조용하다.
    """
    return _utt_seed(f"{utt_id}@{snr_db:g}", base=NOISE_SEED_BASE)

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
        noise_snr_db: float | None = None,
        noise_aug_snrs: list[float] | None = None,
        noisy_prosody_dir: str | Path | None = None,
        w2v_cache_dir: str | Path | None = None,
        noise_aug_clean_ratio: float | None = None,
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

        # 소음 강건성 평가용(11.1절 A3). **기본값 None이면 아래가 전부 무동작이라
        # 학습·기존 평가 동작이 완전히 동일하다.**
        self.noise_snr_db = noise_snr_db
        if noise_snr_db is not None and self.cache_dir is not None:
            # 캐시에 든 멜·운율은 깨끗한 오디오로 계산해둔 것이다. 그대로 쓰면
            # 파형만 오염되고 멜·운율은 깨끗한 채 남아 조건이 뒤섞인다.
            print(f"[dataset] 소음 주입(SNR {noise_snr_db}dB) — 특징 캐시를 쓰지 않는다"
                  f" (캐시된 멜·운율은 깨끗한 오디오 기준이라 섞이면 안 됨)")
            self.cache_dir = None

        # ── 소음 증강 학습(13.5절). 위 평가 경로와 달리 **캐시를 살린다.**
        #
        # 소음 때문에 실제로 다시 계산해야 하는 건 운율 10차원뿐이다. 얼굴은 소음과
        # 무관하고, 파형은 원래 캐시하지 않으니 잡음 주입이 공짜다. 그래서 운율만
        # scripts/precompute_noisy_prosody.py가 미리 만들어둔 걸 읽어 덮어쓴다.
        # 이게 없으면 매 에폭 librosa.pyin(0.80초/건)과 얼굴 JPEG 216만 장이 다시 돌아
        # 에폭이 24분 -> 60~90분이 된다.
        self.noise_aug_snrs = list(noise_aug_snrs) if noise_aug_snrs else None
        # 깨끗하게 남길 비율. None이면 균등(SNR N개 + 깨끗 1 -> 1/(N+1), v11a가 쓴 값).
        # v11a에서 깨끗 조건이 -0.84%p 빠졌는데(McNemar p=0.024) 그게 깨끗 표본이 25%로
        # 줄어든 탓인지 보려고 올릴 수 있게 한다. 나머지 확률은 SNR들이 균등하게 나눈다.
        if noise_aug_clean_ratio is not None and not (0.0 < noise_aug_clean_ratio < 1.0):
            raise ValueError(f"noise_aug_clean_ratio는 (0, 1) 사이여야 한다: {noise_aug_clean_ratio}")
        self.noise_aug_clean_ratio = noise_aug_clean_ratio
        self.epoch = 0
        self._noisy_idx: dict[float, dict[str, int]] = {}
        self._noisy_arr: dict[float, np.ndarray] = {}
        if self.noise_aug_snrs:
            if noise_snr_db is not None:
                raise ValueError(
                    "noise_snr_db(평가용 고정 SNR)와 noise_aug_snrs(학습용 증강)를 "
                    "같이 줄 수 없다 — 어느 쪽이 이기는지 조용해진다"
                )
            # 멜은 깨끗한 캐시에서 나온다. wav2vec2 경로는 멜을 아예 안 써서(model.py의
            # use_w2v 분기) 무해하지만, 멜 백본에 켜면 '깨끗한 멜 + 오염된 운율'이라는
            # 뒤섞인 조건이 에러 없이 만들어진다. 그래서 막는다.
            if cfg.audio_backbone != "wav2vec2":
                raise ValueError(
                    f"소음 증강은 wav2vec2 오디오 백본에서만 쓸 수 있다"
                    f"(현재 '{cfg.audio_backbone}'). 멜 백본은 캐시의 깨끗한 멜을 받게 되어"
                    f" 운율만 오염된 뒤섞인 조건이 된다"
                )
            if noisy_prosody_dir is None:
                raise ValueError(
                    "noise_aug_snrs를 쓰려면 noisy_prosody_dir이 필요하다 — "
                    "scripts/precompute_noisy_prosody.py 먼저"
                )
            self._load_noisy_prosody(Path(noisy_prosody_dir))

        # ── wav2vec2 출력 캐시(13.6절). 학습 한 스텝의 84%가 동결 wav2vec2인데 같은
        # 파형이면 매 에폭 같은 값을 낸다. scripts/precompute_w2v_cache.py가 조건별
        # ({clean|snr20|...}/{utt}.npy, fp16 [T_a, 1024])로 미리 뽑아두면 여기서 읽어
        # 배치에 audio_feat로 실어 보내고, 모델은 proj·frontend만 돈다.
        #
        # 켜면 파형은 안 읽는다 — 모델이 안 쓰는 256KB를 발화마다 디스크에서 읽을 이유가 없다.
        # 캐시가 하나라도 비면 __init__에서 죽는다. 에폭 중간에 죽거나, 더 나쁘게는 조용히
        # 파형 경로로 흘러가 배치 안에 두 경로가 섞이면 안 된다.
        self.w2v_cache_dir = Path(w2v_cache_dir) if w2v_cache_dir else None
        if self.w2v_cache_dir is not None:
            if cfg.audio_backbone != "wav2vec2":
                raise ValueError(f"w2v_cache_dir은 wav2vec2 백본에서만 의미가 있다(현재 '{cfg.audio_backbone}')")
            if noise_snr_db is not None:
                raise ValueError("평가용 고정 SNR(noise_snr_db)과 w2v_cache_dir은 같이 쓸 수 없다 — "
                                 "평가는 파형 경로로 한다")
            conds = ["clean"] + [f"snr{s:g}" for s in (self.noise_aug_snrs or [])]
            utts = self.df["utt_id"].astype(str)
            for c in conds:
                d = self.w2v_cache_dir / c
                missing = [u for u in utts if not (d / f"{u}.npy").exists()]
                if missing:
                    raise FileNotFoundError(
                        f"wav2vec2 캐시 {d}에 {len(missing):,}/{len(utts):,}건이 없다 "
                        f"(예: {missing[:3]}) — scripts/precompute_w2v_cache.py 먼저"
                    )
            if self.return_waveform:
                print("[dataset] w2v 캐시 사용 — 파형은 읽지 않는다 (audio_feat로 대체)", flush=True)
                self.return_waveform = False
            print(f"[dataset] w2v 캐시 {self.w2v_cache_dir} 조건 {conds} · {len(utts):,}발화 전수 확인", flush=True)

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

    def _load_noisy_prosody(self, d: Path) -> None:
        """SNR별 사전계산 운율을 올린다. **발화가 하나라도 비면 여기서 죽는다.**

        중간에 KeyError로 죽으면 16워커 × 수십 분 뒤에야 알게 된다. 더 나쁜 건
        조용히 깨끗한 운율로 흘러가는 것이다 — 그러면 "소음 증강을 했다"는 실험이
        사실은 안 한 실험이 되고 에러는 안 난다.
        """
        want = set(self.df["utt_id"].astype(str))
        for snr in self.noise_aug_snrs:
            p = d / f"snr{snr:g}.npz"
            if not p.exists():
                raise FileNotFoundError(
                    f"소음 운율 캐시 없음: {p}  "
                    f"(scripts/precompute_noisy_prosody.py --snr {snr:g})"
                )
            z = np.load(p)
            ids = [str(u) for u in z["utt_ids"]]
            missing = want - set(ids)
            if missing:
                raise ValueError(
                    f"{p.name}에 매니페스트 발화 {len(missing)}건이 없다 "
                    f"(예: {sorted(missing)[:3]}). 사전계산을 다른 매니페스트로 돌렸을 "
                    f"가능성이 크다 — 다시 돌릴 것"
                )
            arr = z["prosody"]
            if arr.dtype != np.float32:
                # 깨끗한 운율(extract_prosody)이 float32다. 여기가 float64면 두 경로가
                # 섞인 배치에서 np.stack이 float64로 올라가고, CollateFn은 prosody를
                # 캐스팅하지 않아 그대로 모델까지 간다.
                raise ValueError(
                    f"{p.name}의 운율 dtype이 {arr.dtype}다 — float32여야 한다"
                    f"(extract_prosody와 같아야 배치 dtype이 안 올라간다)"
                )
            self._noisy_idx[snr] = {u: i for i, u in enumerate(ids)}
            self._noisy_arr[snr] = arr
        print(f"[dataset] 소음 증강 SNR {self.noise_aug_snrs} + 깨끗 "
              f"— 발화별로 매 에폭 하나를 뽑는다 ({len(want):,}발화)", flush=True)

    def set_epoch(self, epoch: int) -> None:
        """에폭마다 SNR 추첨을 바꾼다. 학습 루프가 매 에폭 불러야 한다.

        [주의] DataLoader(persistent_workers=True)면 워커가 __init__ 때 받은 **사본**을
        들고 있어서 여기서 바꾼 값이 워커에 안 간다. 그래서 소음 증강을 켤 때는
        train.py가 persistent_workers를 끈다. 조용히 1에폭째 조건으로 15에폭을 도는
        사고를 막기 위한 것이다.
        """
        self.epoch = epoch

    def _snr_for(self, utt_id: str) -> float | None:
        """이 발화가 이번 에폭에 받을 SNR. None이면 깨끗하게 둔다.

        고정 할당(발화 ID만으로 결정)이 아니라 에폭을 섞는 이유: 고정하면 각 발화가
        평생 한 조건만 겪고 깨끗한 표본이 1/(N+1)로 줄어든다. 에폭을 섞으면 같은 발화가
        깨끗할 때도 있고 오염될 때도 있어 그게 증강이다.
        """
        if not self.noise_aug_snrs:
            return self.noise_snr_db          # 평가용 고정 SNR, 또는 None
        h = _utt_seed(f"{utt_id}#{self.epoch}", base=NOISE_PICK_BASE)
        if self.noise_aug_clean_ratio is None:
            choices = [None, *self.noise_aug_snrs]
            return choices[h % len(choices)]
        # 해시를 [0,1)로 펴서 앞 구간은 깨끗, 나머지를 SNR들이 균등 분할. 같은 해시를
        # 쓰므로 clean_ratio를 안 주면 위와 완전히 같은 추첨이 나온다(v11a 재현 유지).
        u = h / 2**32
        if u < self.noise_aug_clean_ratio:
            return None
        k = int((u - self.noise_aug_clean_ratio) / (1.0 - self.noise_aug_clean_ratio) * len(self.noise_aug_snrs))
        return self.noise_aug_snrs[min(k, len(self.noise_aug_snrs) - 1)]

    def _compute_features(self, row, snr_db: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y, sr = librosa.load(
            row.wav_path, sr=self.cfg.audio_sample_rate, mono=True,
            duration=self.max_audio_seconds,
        )
        # 소음은 특징 계산 **전에** 넣는다 — 멜·운율·파형이 모두 같은 오염된
        # 오디오에서 나와야 실환경(마이크가 시끄러운 소리를 받는 상황)과 일치한다.
        if snr_db is not None:
            y = add_white_noise(y, snr_db, seed=noise_seed(str(row.utt_id), snr_db))
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

        # **SNR은 여기서 한 번만 뽑는다.** 운율과 파형이 같은 값을 써야 같은 잡음을
        # 겪는다. 아래 두 곳에서 각자 뽑으면 어긋나고, 그 어긋남은 에러를 안 낸다.
        snr_db = self._snr_for(utt_id)

        cache_path = self.cache_dir / f"{utt_id}.npz" if self.cache_dir else None
        if cache_path is not None and cache_path.exists():
            cached = np.load(cache_path)
            mel, prosody, frames = cached["mel"], cached["prosody"], cached["frames"]
        else:
            # snr_db가 아니라 self.noise_snr_db를 넘긴다 — 헷갈리기 쉬운 곳이라 이유를 적는다.
            #
            # 평가 경로(noise_snr_db)는 위 __init__에서 cache_dir을 None으로 만들었으므로
            # 오염된 특징이 캐시에 남을 일이 없다. 반면 증강 경로(noise_aug_snrs)는 캐시를
            # 살려두는 게 목적이다. 여기에 snr_db를 넘기면 캐시 미스 때 **오염된 멜·운율이
            # 깨끗한 캐시에 저장**되고, 그 뒤 모든 실행이 그걸 깨끗한 값으로 읽는다.
            # 증강 경로의 오염은 아래 운율 덮어쓰기 + 파형 잡음으로만 들어간다.
            mel, prosody, frames = self._compute_features(row, self.noise_snr_db)
            if cache_path is not None:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                # 중간에 프로세스가 죽어도 캐시 파일이 반쯤 쓰인 채로 남지 않도록
                # 임시 파일에 먼저 쓰고 마지막에 원자적으로 이름을 바꾼다.
                # np.savez는 파일명이 .npz로 안 끝나면 자기가 .npz를 덧붙여버리므로,
                # 임시 파일명도 반드시 .npz로 끝나야 한다(안 그러면 rename 대상이 없어서 에러남).
                #
                # **여기서 저장하는 prosody는 반드시 깨끗한 값이어야 한다.** 그래서 소음
                # 증강 덮어쓰기는 이 저장보다 **뒤에** 있다. 순서가 바뀌면(또는 아래
                # 블록이 이 저장을 품으면) 오염된 운율이 깨끗한 캐시에 박히고, 그 뒤
                # 모든 실행이 그걸 깨끗한 값으로 읽는다.
                tmp_path = cache_path.with_name(cache_path.stem + ".tmp.npz")
                np.savez(str(tmp_path), mel=mel, prosody=prosody, frames=frames)
                tmp_path.replace(cache_path)

        # 소음 증강: 운율만 사전계산본으로 갈아끼운다(13.5절). **캐시 저장 뒤**다 —
        # 위 주석 참고. 원본(raw)이라 아래 정규화가 깨끗한 운율과 같은 방식으로 걸린다.
        # dtype도 맞는다 — extract_prosody가 float32고 npz도 float32로 저장한다.
        # (안 맞으면 깨끗·소음이 섞인 배치가 np.stack에서 float64로 올라가고,
        #  CollateFn은 prosody를 캐스팅하지 않으므로 모델까지 그대로 간다.)
        if snr_db is not None and self.noise_aug_snrs:
            i = self._noisy_idx[snr_db].get(utt_id)
            if i is None:
                # __init__에서 전수 검사를 하므로 여기 오면 안 된다. 그래도 조용히
                # 깨끗한 운율로 흘려보내지 않는다 — 그러면 안 한 실험이 한 실험이 된다.
                raise KeyError(f"소음 운율 캐시(SNR {snr_db:g})에 {utt_id}가 없다")
            # copy()가 필요한 이유: 이 배열은 전체 발화를 담은 공유 배열의 **뷰**다.
            # 호출부가 제자리에서 고치면 그 발화의 캐시가 영구히 오염된다. 깨끗한 경로는
            # np.load가 매번 새 배열을 주므로 애초에 이 위험이 없다 — 성질을 맞춰둔다.
            prosody = self._noisy_arr[snr_db][i].copy()

        # wav2vec2 캐시(13.6절): 이번 에폭 조건(snr_db)에 맞는 파일을 읽는다. 위 운율과
        # **같은 snr_db**를 쓰므로 운율·오디오 표현이 같은 잡음 실현을 공유한다.
        audio_feat = None
        if self.w2v_cache_dir is not None:
            cond = "clean" if snr_db is None else f"snr{snr_db:g}"
            audio_feat = np.load(self.w2v_cache_dir / cond / f"{utt_id}.npy")  # fp16 [T_a, 1024]

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
        if audio_feat is not None:
            item["audio_feat"] = audio_feat
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
            wav = wav[: int(self.max_audio_seconds * sr)]
            # 운율과 **같은 시드**로 잡음을 넣는다. 여기서 빠뜨리면 wav2vec2 백본만
            # 깨끗한 파형을 받아 조건이 어긋난다(v11은 파형만 읽는다).
            # snr_db는 __getitem__ 맨 위에서 한 번 뽑은 값이다 — 평가 경로면 고정 SNR,
            # 증강 경로면 이번 에폭 추첨 결과이고, 위 운율 덮어쓰기와 반드시 같은 값이다.
            if snr_db is not None:
                wav = add_white_noise(wav, snr_db, seed=noise_seed(utt_id, snr_db))
            item["waveform"] = wav
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

        if "audio_feat" in batch[0]:
            # 캐시된 wav2vec2 출력 [T_a, 1024] fp16 -> [B, T_max, 1024] float32 + 마스크(True=패딩).
            # _pad_time이 float32로 올려준다 — proj(Linear)가 float32라 여기서 맞춘다.
            feat_tensor, feat_mask = _pad_time([b["audio_feat"] for b in batch])
            out_extra["audio_feat"] = feat_tensor
            out_extra["audio_feat_padding_mask"] = feat_mask

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
