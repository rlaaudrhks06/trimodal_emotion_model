"""prosody 벡터 정규화 통계를 train 세트에서만 계산해 저장.

데이터_전처리_EDA_점검_및_근거.md §1.2 / §4-1,2: prosody 10차원(f0_mean 수백 vs
jitter_approx 0.01대)이 지금까지 스케일링 없이 그대로 쓰이고 있었다. 이 스크립트는
- IQR 기준 이상치 경계(clip_lower/upper)
- (이상치를 clip한 뒤의) 평균/표준편차
- 참고용 왜도(skew), 이상치 개수
를 **train 매니페스트에서만** 계산해 JSON으로 저장한다. val/test는 이 통계를
그대로 transform만 하고 다시 fit하지 않는다 — fit을 train에만 하는 게 4번
노트북(Scaling)의 핵심 원칙이자 여기서 지키려는 것.

이미 feature_cache에 발화별 prosody가 계산돼 있으므로(오디오를 다시 읽지 않고)
캐시된 .npz만 읽어서 빠르게 계산한다.

사용 예:
    python scripts/compute_prosody_stats.py \
        --manifest data/manifests/train.csv \
        --cache-dir data/feature_cache \
        --out data/prosody_stats_train.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features.prosody import PROSODY_FEATURE_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="반드시 train 매니페스트만 (val/test 넣으면 leakage)")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--iqr-multiplier", type=float, default=1.5)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    cache_dir = Path(args.cache_dir)

    vecs, missing = [], 0
    for utt_id in df["utt_id"].astype(str):
        p = cache_dir / f"{utt_id}.npz"
        if not p.exists():
            missing += 1
            continue
        vecs.append(np.load(p)["prosody"])

    if missing:
        print(f"[compute_prosody_stats] 경고: feature_cache에 없는 발화 {missing}개는 통계에서 제외 "
              f"(먼저 train.py나 diagnose_dataset.py를 한 번 돌려 캐시를 채워야 함)")
    if len(vecs) == 0:
        raise RuntimeError("캐시된 prosody가 하나도 없음 — cache-dir 경로 확인 필요")

    arr = np.stack(vecs)  # [N, 10]

    q1 = np.percentile(arr, 25, axis=0)
    q3 = np.percentile(arr, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - args.iqr_multiplier * iqr
    upper = q3 + args.iqr_multiplier * iqr

    is_outlier = (arr < lower) | (arr > upper)
    n_outliers_per_dim = is_outlier.sum(axis=0)

    clipped = np.clip(arr, lower, upper)
    mean = clipped.mean(axis=0)
    std = clipped.std(axis=0) + 1e-6
    skewness = skew(arr, axis=0)

    stats = {
        "feature_names": PROSODY_FEATURE_NAMES,
        "n_samples": int(arr.shape[0]),
        "iqr_multiplier": args.iqr_multiplier,
        "clip_lower": lower.tolist(),
        "clip_upper": upper.tolist(),
        "mean": mean.tolist(),  # clip 이후 평균/표준편차 -> 이 값으로 z-score 정규화
        "std": std.tolist(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[compute_prosody_stats] {arr.shape[0]}개 발화 기준 통계 저장 -> {args.out}\n")
    print(f"{'feature':16s} {'mean':>9s} {'std':>9s} {'skew':>7s} {'outliers':>9s}  판단")
    for i, name in enumerate(PROSODY_FEATURE_NAMES):
        n_out = int(n_outliers_per_dim[i])
        pct_out = 100.0 * n_out / arr.shape[0]
        skew_note = "치우침 큼" if abs(skewness[i]) > 1.0 else ("약간 치우침" if abs(skewness[i]) > 0.5 else "대칭적")
        print(
            f"{name:16s} {mean[i]:9.4f} {std[i]:9.4f} {skewness[i]:7.3f} "
            f"{n_out:6d}({pct_out:4.1f}%)  {skew_note}"
        )


if __name__ == "__main__":
    main()
