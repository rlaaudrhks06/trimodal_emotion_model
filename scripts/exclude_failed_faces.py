"""§8.8 대응: 얼굴 재검출에 실패해 옛(전신 크롭) 이미지가 그대로 남은 발화를
train/val/test 매니페스트에서 제외한다.

refix_face_crops.py는 클립 단위 집계(성공/실패 개수)만 로그에 남기고 개별 실패
utt_id는 기록하지 않았다. 대신 "실패한 발화의 face_frames_dir는 재처리 기간 동안
단 한 파일도 새로 안 써졌다"는 사실을 이용해, --failed-ids 파일(find -newermt로
미리 뽑아둔 목록)에 있는 utt_id 행만 매니페스트에서 제거한다.

사용법:
    # 1) 실패한 utt_id 목록 먼저 뽑기(재처리 시작 이전 시각 기준):
    comm -23 <(ls data/processed_full/faces | sort) \
        <(find data/processed_full/faces -maxdepth 2 -name "frame_*.jpg" \
          -newermt "2026-08-02 00:00:00" | sed 's|/frame_[^/]*$||' | xargs -n1 basename | sort -u) \
        > failed_utt_ids.txt

    # 2) 매니페스트에서 제외 (원본은 .bak로 백업):
    python scripts/exclude_failed_faces.py --failed-ids failed_utt_ids.txt \
        --manifests data/manifests/train.csv data/manifests/val.csv data/manifests/test.csv
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--failed-ids", required=True, type=Path)
    ap.add_argument("--manifests", nargs="+", required=True, type=Path)
    args = ap.parse_args()

    failed = {line.strip() for line in args.failed_ids.read_text(encoding="utf-8").splitlines() if line.strip()}
    print(f"[exclude_failed_faces] 제외 대상 utt_id {len(failed)}개 로드")

    total_removed = 0
    for manifest_path in args.manifests:
        df = pd.read_csv(manifest_path)
        before = len(df)
        df = df[~df["utt_id"].astype(str).isin(failed)]
        removed = before - len(df)
        total_removed += removed

        backup_path = manifest_path.with_suffix(manifest_path.suffix + ".bak")
        shutil.copy2(manifest_path, backup_path)
        df.to_csv(manifest_path, index=False)
        print(f"[exclude_failed_faces] {manifest_path}: {before} -> {len(df)}행 ({removed}개 제외, 원본은 {backup_path}에 백업)")

    print(f"[exclude_failed_faces] 완료. 총 {total_removed}개 발화 제외")


if __name__ == "__main__":
    main()
