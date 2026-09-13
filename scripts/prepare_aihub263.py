"""AI Hub "감정 분류를 위한 대화 음성 데이터셋"(dataSetSn=263)을 학습 매니페스트로 만든다.

## 왜 (통합기록 13.12절, 11.3.2 항목 5)

병목은 오디오 브랜치이고 과적합 원인은 화자 외우기다(train 62% vs test 45%). 멀티모달
영상은 배포분 5,600클립을 전부 쓰고 있어(5601-6000.zip이 없다) 같은 데이터는 더 없다.
263은 **다른 화자들의 음성 43,991발화**(48kHz, 3~6초, 전사문 포함, 영상 없음)라
"train 화자 다양성"을 겨냥한다. train에만 넣고 val/test는 그대로 둔다 — 평가는
오염되지 않는다.

## 라벨 — 5명 다수결, 3/5 이상 합의만

CSV에 "상황"(대본 의도)과 5명의 라벨링이 있는데 둘의 일치가 68.1%뿐이다. 멀티모달
영상의 정답도 "사람이 보고 매긴 것"이라 같은 성질인 **다수결**을 쓴다. 2/5 이하(16%)는
사실상 잡음이라 뺀다 → 36,680발화. 다수결 분포는 sad 38%·neutral 18%·angry 18%로
기존 데이터(disgust·happy 우세)와 다르다 — 클래스 가중치가 train 매니페스트에서
다시 계산되므로 자동으로 반영된다.

## 영상 없음

face_frames_dir을 빈 값으로 둔다. ManifestEmotionDataset이 빈 값이면 **검은 프레임 1장**을
준다 — 모달리티 드롭아웃이 영상을 지울 때 모델이 보는 입력과 정확히 같다(frames.zero_()
+ 마스크 유지). 새 개념이 아니라 이미 학습한 "카메라가 죽은 상황"이다.

## 산출물

    data/processed_ext/aihub263/audio/{wav_id}.wav     16kHz mono PCM16
    data/manifests_ext/aihub263.csv                    utt_id,label,wav_path,text,face_frames_dir,agree,age,gender,src
    data/manifests_ext/train_si_plus263.csv            기존 train + 263 (학습용)

실행:
    python scripts/prepare_aihub263.py --raw-dir "/data/work/aihub_download/KETI데이터/2-12.감정_분류를_위한_대화_음성_데이터셋"
"""
from __future__ import annotations

import argparse
import collections
import io
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets.labels import EMOTION_LABELS  # noqa: E402

CSVS = ["4차_14606.csv", "5차_10011.csv", "5차년도_2차_19374.csv"]
ZIPS = ["4차_wav.zip", "5차_wav.zip", "5차년도_2차_19374.zip"]
NORM = {"happiness": "happy", "sadness": "sad", "anger": "angry", "angry": "angry",
        "disgust": "disgust", "fear": "fear", "surprise": "surprise", "neutral": "neutral", "happy": "happy", "sad": "sad"}


def norm_label(s) -> str | None:
    s = str(s).strip().lower()
    return NORM.get(s)


def load_labels(raw: Path, min_agree: int) -> pd.DataFrame:
    frames = []
    for f in CSVS:
        d = pd.read_csv(raw / f, encoding="cp949")
        d["src"] = f.split("_")[0]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    lab_cols = [c for c in df.columns if c.endswith("감정") and c[0].isdigit()]
    assert len(lab_cols) == 5, lab_cols
    votes = df[lab_cols].apply(lambda r: [norm_label(x) for x in r], axis=1)
    bad = votes.map(lambda v: any(x is None for x in v))
    if bad.any():
        print(f"  ⚠ 알 수 없는 라벨 {int(bad.sum())}건 제외: {set(x for v in votes[bad] for x in v)}")
        df, votes = df[~bad], votes[~bad]
    top = votes.map(lambda v: collections.Counter(v).most_common(1)[0])
    df = df.assign(label=top.map(lambda t: t[0]), agree=top.map(lambda t: t[1]),
                   text=df["발화문"].astype(str).str.strip(), age=df["나이"], gender=df["성별"])
    assert set(df.label) <= set(EMOTION_LABELS), set(df.label) - set(EMOTION_LABELS)
    n0 = len(df)
    df = df[df.agree >= min_agree]
    print(f"  라벨 {n0:,} -> 합의 {min_agree}/5 이상 {len(df):,}")
    return df


def _resample_one(args):
    """워커: zip 안의 wav 하나를 16kHz mono PCM16으로 저장. (zip, name, out)"""
    zpath, name, out = args
    import librosa
    with zipfile.ZipFile(zpath) as zf, zf.open(name) as fh:
        y, sr = sf.read(io.BytesIO(fh.read()), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000, res_type="soxr_hq")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.wav")
    sf.write(tmp, y, 16000, subtype="PCM_16")
    tmp.replace(out)
    return out.stem, len(y) / 16000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", default="data/processed_ext/aihub263")
    ap.add_argument("--manifest-dir", default="data/manifests_ext")
    ap.add_argument("--base-train", default="data/manifests_si/train.csv")
    ap.add_argument("--min-agree", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    raw = Path(args.raw_dir); out = ROOT / args.out_dir; mdir = ROOT / args.manifest_dir
    mdir.mkdir(parents=True, exist_ok=True); (out / "audio").mkdir(parents=True, exist_ok=True)

    df = load_labels(raw, args.min_agree)
    print("  라벨 분포:", df.label.value_counts().to_dict())

    # zip 안의 wav 목록 → wav_id -> (zip, 멤버)
    members = {}
    for z in ZIPS:
        with zipfile.ZipFile(raw / z) as zf:
            for n in zf.namelist():
                if n.lower().endswith(".wav"):
                    members[Path(n).stem] = (str(raw / z), n)
    missing = [w for w in df.wav_id if w not in members]
    if missing:
        print(f"  ⚠ zip에 없는 wav {len(missing)}건 제외 (예: {missing[:3]})")
        df = df[~df.wav_id.isin(missing)]

    jobs = [(members[w][0], members[w][1], out / "audio" / f"{w}.wav") for w in df.wav_id
            if not (out / "audio" / f"{w}.wav").exists()]
    print(f"  리샘플 대상 {len(jobs):,} (이미 있음 {len(df)-len(jobs):,})")
    t0 = time.time(); durs = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (wid, dur) in enumerate(ex.map(_resample_one, jobs, chunksize=16), 1):
            durs[wid] = dur
            if i % 2000 == 0:
                print(f"    {i:,}/{len(jobs):,}  {i/(time.time()-t0):.0f}건/초", flush=True)
    print(f"  리샘플 끝 {(time.time()-t0)/60:.1f}분")

    man = pd.DataFrame({
        "utt_id": "k263_" + df.wav_id.astype(str),
        "label": df.label.values,
        "wav_path": [str((out / "audio" / f"{w}.wav").relative_to(ROOT)) for w in df.wav_id],
        "text": df.text.values,
        "face_frames_dir": "",                       # 영상 없음 -> 데이터셋이 검은 프레임 1장
        "agree": df.agree.values, "age": df.age.values, "gender": df.gender.values, "src": df.src.values,
    })
    man.to_csv(mdir / "aihub263.csv", index=False)
    base = pd.read_csv(ROOT / args.base_train)
    plus = pd.concat([base, man[[c for c in base.columns if c in man.columns]]], ignore_index=True)
    for c in base.columns:
        if c not in man.columns:
            plus[c] = plus[c]                        # aux 라벨 컬럼 등은 263 행에서 NaN -> IGNORE_INDEX
    plus.to_csv(mdir / "train_si_plus263.csv", index=False)
    lens = np.array([sf.info(ROOT / p).duration for p in man.wav_path[:2000]])
    print(f"\n  aihub263.csv {len(man):,}행 · 길이 표본 평균 {lens.mean():.2f}s 최대 {lens.max():.2f}s (8초 초과 {(lens>8).mean()*100:.1f}%)")
    print(f"  train_si_plus263.csv {len(plus):,}행 = 기존 {len(base):,} + 263 {len(man):,}")
    print(f"  결합 라벨 분포: {plus.label.str.lower().value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
