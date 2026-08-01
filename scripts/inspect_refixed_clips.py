"""refix_face_crops.py로 방금 재처리한 특정 클립들의 얼굴 크롭을 눈으로 확인.

inspect_face_crops.py는 manifest에서 무작위 샘플링이라 아직 안 고친 클립이 섞여
나온다. 이건 --clip-ids로 지정한 클립들의 face_frames_dir만 모아서 그리드로 보여준다.

사용법:
    python scripts/inspect_refixed_clips.py --frames-out data/processed_full \
        --clip-ids clip_1 clip_10 clip_100 clip_101 clip_102 \
        --out face_crop_refixed_v1.png --per-clip 3
"""
import argparse
import random

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-out", required=True, type=Path)
    ap.add_argument("--clip-ids", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-clip", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    faces_root = args.frames_out / "faces"

    rows_data = []
    for clip_id in args.clip_ids:
        utt_dirs = sorted(faces_root.glob(f"{clip_id}_*"))
        if not utt_dirs:
            print(f"[inspect_refixed_clips] {clip_id}: 발화 디렉터리 없음, 건너뜀")
            continue
        n = min(args.per_clip, len(utt_dirs))
        chosen = random.sample(utt_dirs, n)
        for d in chosen:
            paths = sorted(d.glob("frame_*.jpg"))
            if not paths:
                continue
            idx = sorted(set([0, len(paths) // 2, len(paths) - 1]))
            frames = []
            for i in idx:
                img = cv2.imread(str(paths[i]))
                if img is not None:
                    frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            rows_data.append((clip_id, d.name, frames))

    cols = 3
    n_rows = len(rows_data)
    fig, axes = plt.subplots(n_rows, cols, figsize=(cols * 2.2, n_rows * 2.4))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, (clip_id, utt_id, frames) in enumerate(rows_data):
        for j in range(cols):
            ax = axes[i, j]
            if j < len(frames):
                ax.imshow(frames[j])
            ax.axis("off")
            if j == 0:
                ax.set_title(f"{clip_id}\n{utt_id}", fontsize=7, loc="left")

    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"[inspect_refixed_clips] 저장 완료: {args.out} ({n_rows}개 발화)")


if __name__ == "__main__":
    main()
