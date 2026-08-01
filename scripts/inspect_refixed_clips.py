"""refix_face_crops.py로 방금 재처리한 발화들의 얼굴 크롭을 눈으로 확인.

inspect_face_crops.py는 manifest에서 무작위 샘플링이라 아직 안 고친 클립이 섞여
나온다. clip_id는 JSON 내부 필드값이라 파일명(clip_1 등)과 다를 수 있어 매칭이
불안정하므로, 대신 "최근 N분 안에 수정된" 얼굴 프레임 파일을 찾아서 그리드로 보여준다.

(v2: 파이썬으로 전체 face 디렉터리를 순회하며 매 파일 stat()을 뜨면 데이터셋 전체
규모(수만 개 발화) 때문에 너무 느려서, OS의 find(-newermt)를 그대로 호출하는
방식으로 바꿈 — find는 단일 C 프로세스라 훨씬 빠르다.)

사용법:
    python scripts/inspect_refixed_clips.py --frames-out data/processed_full \
        --recent-minutes 15 --out face_crop_refixed_v1.png --sample 15
"""
import argparse
import random
import subprocess

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-out", required=True, type=Path)
    ap.add_argument("--recent-minutes", type=float, default=15.0, help="이 시간 안에 수정된 프레임만 대상")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=15, help="눈으로 확인할 발화 수")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    faces_root = args.frames_out / "faces"

    result = subprocess.run(
        [
            "find", str(faces_root), "-maxdepth", "2", "-name", "frame_*.jpg",
            "-newermt", f"-{args.recent_minutes} minutes",
        ],
        capture_output=True, text=True, check=True,
    )
    recent_files = [Path(p) for p in result.stdout.splitlines() if p]
    recent_dirs = sorted({p.parent for p in recent_files})
    print(f"[inspect_refixed_clips] 최근 {args.recent_minutes}분 안에 수정된 발화 디렉터리 {len(recent_dirs)}개 발견")
    if not recent_dirs:
        print("[inspect_refixed_clips] 대상 없음 -> --recent-minutes 값을 늘려보세요.")
        return

    n = min(args.sample, len(recent_dirs))
    chosen = random.sample(recent_dirs, n)

    rows_data = []
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
        rows_data.append((d.name, frames))

    cols = 3
    n_rows = len(rows_data)
    fig, axes = plt.subplots(n_rows, cols, figsize=(cols * 2.2, n_rows * 2.4))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, (utt_id, frames) in enumerate(rows_data):
        for j in range(cols):
            ax = axes[i, j]
            if j < len(frames):
                ax.imshow(frames[j])
            ax.axis("off")
            if j == 0:
                ax.set_title(utt_id, fontsize=7, loc="left")

    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"[inspect_refixed_clips] 저장 완료: {args.out} ({n_rows}개 발화)")


if __name__ == "__main__":
    main()
