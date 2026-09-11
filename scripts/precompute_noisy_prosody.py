"""소음 증강 학습용 **운율만** 미리 계산한다 — 캐시를 살리면서 소음을 넣기 위해.

## 왜 필요한가 (통합기록 13.5절)

제안서 3.1.5 ②에 소음 증강 학습을 약속했다(SNR 20dB 43.80% / 10dB 36.85%, 깨끗 46.19%;
일반 가정의 음성 환경이 정확히 그 구간이다). 그런데 **지금 구조로는 그냥 켤 수 없다.**

`manifest_dataset.py`는 `noise_snr_db`를 받으면 `cache_dir`을 None으로 만든다. 옳은
처리다 — 캐시에 든 멜·운율은 깨끗한 오디오로 계산한 것이라 섞이면 조건이 뒤죽박죽이
된다. 문제는 그러면 **매 에폭마다** 이게 다시 돈다는 것이다.

  · librosa.pyin 운율 재계산 (0.80초/건 — feature_cache가 존재하는 이유 그 자체)
  · 얼굴 JPEG 재디코딩 (발화당 최대 32장, 전체 216만 장)

24분/에폭이 60~90분이 되고, 15에폭 한 번이 6시간에서 15~22시간이 된다. 시드를 2~3개
돌려야 하는데(효과 크기가 1.3%p 검출 한계에 가깝다) 성립하지 않는다.

## 해결 — 실제로 다시 계산해야 하는 건 운율 10차원뿐이다

  얼굴 사진   소음과 무관              -> 기존 캐시 그대로
  멜          v11은 wav2vec2 경로라 안 씀 -> 계산 안 함
  파형        원래 캐시 안 함(매번 디스크) -> 잡음 주입이 공짜
  **운율**     소음의 영향을 받음         -> **이 스크립트가 미리 만든다**

## 파형과 운율이 **같은 잡음**을 겪게 하는 것이 핵심

`add_white_noise(y, snr, seed)`는 seed가 같으면 같은 잡음을 만든다. 그래서 여기서 쓴
seed를 학습 때도 그대로 쓰면, 미리 계산한 운율과 그때그때 만든 파형이 **동일한 오염**을
공유한다. 어긋나면 조용히 틀린 조건이 된다 — 이 프로젝트가 반복해서 당한 유형이다.

seed는 `(utt_id, snr)`로 묶는다. SNR마다 다른 잡음 실현을 주되, 같은 (발화, SNR)에는
항상 같은 잡음이 오게 한다.

## 저장 형식 — 작은 파일 수십만 개를 만들지 않는다

SNR당 **파일 하나**에 전체 발화를 담는다. 발화별 .npz로 쪼개면 17만 개가 되고, 그건
방금 216만 개 얼굴 프레임 전송에서 겪은 문제를 새로 만드는 것이다.

  data/noisy_prosody/snr{N}.npz   utt_ids (N,) · prosody (N,10) float32

**원본(raw) 운율을 저장한다.** 기존 캐시와 같은 규약이다 — 정규화는 읽을 때 적용하므로
통계를 바꿔도 캐시를 무효화할 필요가 없다(`manifest_dataset.py`의 prosody_mean 정규화 블록).

## train만 만드는 이유

val은 깨끗한 채로 둔다. 체크포인트 선택이 val_accuracy 기준이라, val에 소음을 넣으면
선택 기준 자체가 v11과 달라져 비교가 깨진다. "한 번에 하나만 바꾼다"를 지킨다.
소음 조건 평가는 기존 `evaluate.py --noise-snr` 경로가 이미 올바르게 처리한다.

실행:
    python scripts/precompute_noisy_prosody.py --snr 20 10 5 --workers 16
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config                                       # noqa: E402
from src.datasets.manifest_dataset import add_white_noise, noise_seed    # noqa: E402
from src.features.prosody import extract_prosody                         # noqa: E402

# noise_seed는 **manifest_dataset에만** 있다. 학습·평가가 파형을 오염시킬 때 쓰는
# 그 함수를 그대로 가져와야 여기서 만든 운율과 같은 잡음이 된다. 여기 사본을 두면
# 한쪽만 고쳐지고, 그러면 운율과 파형이 다른 잡음을 겪는데 에러는 안 난다.


def one(args: tuple[str, str, float, int, float]) -> tuple[str, np.ndarray]:
    """발화 하나의 소음 운율. 워커에서 돈다."""
    utt_id, wav_path, snr_db, sr, max_sec = args
    import librosa

    y, _ = librosa.load(wav_path, sr=sr, mono=True, duration=max_sec)
    y = add_white_noise(y, snr_db, seed=noise_seed(utt_id, snr_db))
    return utt_id, extract_prosody(y, sr).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_si_w2v.yaml")
    ap.add_argument("--split", default="train",
                    help="기본 train만. val은 깨끗하게 둔다(체크포인트 선택 기준 유지)")
    ap.add_argument("--snr", type=float, nargs="+", default=[20.0, 10.0, 5.0])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-audio-seconds", type=float, default=8.0,
                    help="ManifestEmotionDataset 기본값과 같아야 한다")
    ap.add_argument("--out-dir", default="data/noisy_prosody")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    manifest = ROOT / cfg.raw["train"][f"{args.split}_manifest"]
    df = pd.read_csv(manifest)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[precompute] {args.split} {len(df):,}발화 × SNR {args.snr} "
          f"· 워커 {args.workers}")
    print(f"[precompute] 출력 {out_dir}")

    for snr in args.snr:
        out = out_dir / f"snr{snr:g}.npz"
        if out.exists():
            print(f"[SNR {snr:g}] 이미 있음 — 건너뜀 ({out.name})")
            continue

        jobs = [(str(r.utt_id), str(r.wav_path), snr,
                 cfg.audio_sample_rate, args.max_audio_seconds)
                for r in df.itertuples()]

        t0 = time.time()
        ids: list[str] = []
        rows: list[np.ndarray] = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, (uid, pros) in enumerate(ex.map(one, jobs, chunksize=32), 1):
                ids.append(uid)
                rows.append(pros)
                if i % 2000 == 0:
                    el = time.time() - t0
                    rate = i / el
                    print(f"[SNR {snr:g}] {i:,}/{len(jobs):,}  {rate:.1f}건/초  "
                          f"남은 {(len(jobs)-i)/rate/60:.0f}분", flush=True)

        arr = np.stack(rows)
        if not np.isfinite(arr).all():
            # 조용히 NaN을 저장하면 학습이 에러 없이 망가진다.
            bad = int((~np.isfinite(arr)).any(axis=1).sum())
            print(f"[SNR {snr:g}] ⚠ 비정상 값이 있는 발화 {bad}건 — 확인 필요", flush=True)

        # 임시 파일에 쓴 뒤 원자적 rename. 중간에 죽어도 반쯤 쓰인 파일이 안 남는다
        # (manifest_dataset.py의 캐시 저장과 같은 규약).
        tmp = out.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, utt_ids=np.array(ids), prosody=arr)
        tmp.replace(out)
        print(f"[SNR {snr:g}] 완료 {len(ids):,}건 {(time.time()-t0)/60:.1f}분 "
              f"-> {out.name} ({out.stat().st_size/1e6:.1f}MB)", flush=True)

    print("\n[precompute] 끝. 학습은 configs/config_noise_aug.yaml로 돌린다 — "
          "noise_aug_snrs/noisy_prosody_dir이 거기 있다.")
    print("[precompute] 검증: python tests/test_noise_aug.py "
          "(운율과 파형이 같은 잡음을 겪는지 본다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
