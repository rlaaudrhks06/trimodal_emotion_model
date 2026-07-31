"""데이터_전처리_EDA_점검_및_근거.md 2장의 공백을 점검하는 진단 스크립트.

prosody 스케일/이상치/왜도는 compute_prosody_stats.py가 이미 출력하므로 여기서는
다루지 않고, 그 외 항목만 확인한다:
  - 라벨 밸런스가 train/val/test 사이에 비슷한지
  - 특정 화자(person_id)가 특정 감정에 쏠려있는지(confound 위험)
  - 결측/퇴화 샘플(빈 텍스트, 빈 얼굴 프레임 디렉터리)
  - 텍스트 길이 / 오디오 길이 분포(자르는 상한을 얼마나 넘는지)

이건 읽기 전용 진단이다 — 여기서 뭔가를 자동으로 고치거나 데이터를 바꾸지 않는다.
결과를 보고 실제로 문제가 있는 항목만 골라서 반영하는 게 원칙(§4).

사용 예:
    python scripts/eda_report.py --cache-dir data/feature_cache
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features.prosody import PROSODY_FEATURE_NAMES


def label_balance(splits: dict[str, pd.DataFrame]):
    print("\n=== 1. 라벨 밸런스 (train/val/test 비율(%) 비교) ===")
    rows = []
    for name, df in splits.items():
        counts = df["label"].astype(str).str.strip().str.lower().value_counts(normalize=True)
        rows.append(counts.rename(name))
    table = (pd.concat(rows, axis=1).fillna(0.0) * 100).round(2)
    print(table)

    # train 대비 val/test 비율이 크게 벗어나는 클래스 경고 (절대 차이 3%p 이상)
    for other in ["val", "test"]:
        if other not in table.columns:
            continue
        diff = (table[other] - table["train"]).abs()
        flagged = diff[diff > 3.0]
        if len(flagged) > 0:
            print(f"[경고] train 대비 {other} 비율이 3%p 이상 차이나는 클래스:\n{flagged.round(2)}")


def speaker_label_crosstab(df: pd.DataFrame):
    print("\n=== 2. 화자(person_id)-감정 교차표 (train, 발화 수 상위 20명, 비율(%)) ===")
    person_id = df["utt_id"].astype(str).str.split("_").str[1]
    tmp = df.assign(person_id=person_id)
    top_people = tmp["person_id"].value_counts().head(20).index
    tmp = tmp[tmp["person_id"].isin(top_people)]
    ct = (pd.crosstab(tmp["person_id"], tmp["label"], normalize="index") * 100).round(1)
    print(ct)

    max_ratio = ct.max(axis=1)
    skewed = max_ratio[max_ratio > 80]
    if len(skewed) > 0:
        print(f"\n[경고] 특정 감정 하나에 80% 이상 쏠린 화자 {len(skewed)}명:\n{skewed.round(1)}")
    else:
        print("\n특정 감정에 80% 이상 쏠린 화자 없음 (80% 기준)")


def degenerate_samples(df: pd.DataFrame):
    print("\n=== 3. 결측/퇴화 샘플 점검 (train) ===")
    empty_text = df["text"].astype(str).str.strip().eq("").sum()
    print(f"빈 텍스트: {empty_text}개 / {len(df)}개")

    missing_dirs, empty_dirs = 0, 0
    for d in df["face_frames_dir"]:
        p = Path(d)
        if not p.exists():
            missing_dirs += 1
        elif not any(p.iterdir()):
            empty_dirs += 1
    print(f"얼굴 프레임 디렉터리 없음: {missing_dirs}개, 있지만 비어있음: {empty_dirs}개")


def text_length_distribution(df: pd.DataFrame, max_text_len: int = 64):
    print("\n=== 4. 텍스트 길이 분포 (어절 수 기준 근사치) ===")
    lengths = df["text"].astype(str).str.split().str.len()
    print(lengths.describe().round(2))
    # 한국어 어절 1개가 서브워드 토큰 1.5~2개 정도로 쪼개지는 경우가 흔해서,
    # max_text_len(기본 64토큰)의 절반(=32어절) 넘는 비율을 "잘릴 위험"으로 근사한다.
    risk_ratio = (lengths > max_text_len // 2).mean() * 100
    print(f"{max_text_len}토큰 상한에 걸려 잘릴 위험이 있는 발화 비율(근사): {risk_ratio:.1f}%")


def audio_duration_distribution(df: pd.DataFrame, cache_dir: Path | None, max_audio_seconds: float = 8.0):
    print("\n=== 5. 오디오 길이 분포 (초, feature_cache의 duration_sec 재사용) ===")
    if cache_dir is None:
        print("--cache-dir 미지정 -> 건너뜀")
        return
    dur_idx = PROSODY_FEATURE_NAMES.index("duration_sec")
    durations, missing = [], 0
    for utt_id in df["utt_id"].astype(str):
        p = cache_dir / f"{utt_id}.npz"
        if not p.exists():
            missing += 1
            continue
        durations.append(float(np.load(p)["prosody"][dur_idx]))
    if missing:
        print(f"(캐시 없는 발화 {missing}개는 제외)")
    s = pd.Series(durations)
    print(s.describe().round(3))
    near_cap = (s >= max_audio_seconds - 0.5).mean() * 100
    print(f"{max_audio_seconds}초 상한 근처(>= {max_audio_seconds - 0.5}초)라 잘렸을 가능성 있는 비율: {near_cap:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/manifests/train.csv")
    parser.add_argument("--val", default="data/manifests/val.csv")
    parser.add_argument("--test", default="data/manifests/test.csv")
    parser.add_argument("--cache-dir", default=None, help="지정하면 오디오 길이 분포도 확인(feature_cache 재사용)")
    parser.add_argument("--max-text-len", type=int, default=64)
    parser.add_argument("--max-audio-seconds", type=float, default=8.0)
    args = parser.parse_args()

    splits = {"train": pd.read_csv(args.train), "val": pd.read_csv(args.val), "test": pd.read_csv(args.test)}
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    label_balance(splits)
    speaker_label_crosstab(splits["train"])
    degenerate_samples(splits["train"])
    text_length_distribution(splits["train"], args.max_text_len)
    audio_duration_distribution(splits["train"], cache_dir, args.max_audio_seconds)


if __name__ == "__main__":
    main()
