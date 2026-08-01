"""얼굴 크롭 품질/정렬 눈으로 확인용 진단 스크립트.

8.7절 베이스라인 + 두 차례 사전학습 백본(ImageNet MobileNetV3, 얼굴인식
MobileFaceNet) 교체가 모두 실패한 뒤, "크롭이 랜드마크 정렬 없이 바운딩박스만
잘라낸 거라 얼굴인식 사전학습 백본이 기대하는 입력과 다르다"는 가설을 눈으로
검증하기 위한 용도. train manifest에서 감정별로 몇 개씩 샘플링해, 각 발화의
첫/중간/마지막 프레임을 모아 하나의 큰 그리드 PNG로 저장한다.

사용법:
    python scripts/inspect_face_crops.py --manifest data/manifests/train.csv \
        --out face_crop_inspection.png --per-class 3
"""
import argparse
import random

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path


def pick_frames(frames_dir: str) -> list[np.ndarray]:
    paths = sorted(Path(frames_dir).glob("*"))
    if not paths:
        return []
    idx = sorted(set([0, len(paths) // 2, len(paths) - 1]))
    imgs = []
    for i in idx:
        img = cv2.imread(str(paths[i]))
        if img is None:
            continue
        imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    df = pd.read_csv(args.manifest)

    rows = []
    for label, grp in df.groupby("label"):
        n = min(args.per_class, len(grp))
        rows.append(grp.sample(n=n, random_state=args.seed))
    sample = pd.concat(rows, ignore_index=True)

    # 각 발화당 최대 3프레임(처음/중간/끝) x 발화 수 만큼의 그리드
    cols = 3
    rows_data = []
    for _, r in sample.iterrows():
        frames = pick_frames(r.face_frames_dir)
        while len(frames) < cols:
            frames.append(np.zeros((112, 112, 3), dtype=np.uint8))
        rows_data.append((r.label, r.utt_id, frames[:cols]))

    n_rows = len(rows_data)
    fig, axes = plt.subplots(n_rows, cols, figsize=(cols * 2.2, n_rows * 2.4))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, (label, utt_id, frames) in enumerate(rows_data):
        for j in range(cols):
            ax = axes[i, j]
            ax.imshow(frames[j])
            ax.axis("off")
            if j == 0:
                ax.set_title(f"{label}\n{utt_id}", fontsize=8, loc="left")

    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"[inspect_face_crops] 저장 완료: {args.out} ({n_rows}개 발화 x {cols}프레임)")


if __name__ == "__main__":
    main()
