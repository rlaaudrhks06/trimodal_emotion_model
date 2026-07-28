"""매니페스트 CSV를 clip_id 기준으로 train/val/test로 분할.

utt_id가 "{clip_id}_{person_id}_{start}_{end}" 형식이므로 첫 번째 '_' 이전을
clip_id로 취급한다. 같은 클립(같은 두 화자·같은 장면)이 여러 발화로 쪼개져
train/val/test에 걸쳐 나뉘면 정보 누수가 생기므로, 클립 단위로 통째로 분할한다
(설계 v3 §7.1 subject-independent 원칙과 동일한 취지).
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_clip = defaultdict(list)
    for row in rows:
        clip_id = row["utt_id"].split("_")[0]
        by_clip[clip_id].append(row)

    clip_ids = list(by_clip.keys())
    random.Random(args.seed).shuffle(clip_ids)

    n = len(clip_ids)
    n_val = max(1, int(n * args.val_ratio))
    n_test = max(1, int(n * args.test_ratio))
    val_clips = set(clip_ids[:n_val])
    test_clips = set(clip_ids[n_val:n_val + n_test])
    train_clips = set(clip_ids[n_val + n_test:])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = rows[0].keys()

    for split_name, clips in [("train", train_clips), ("val", val_clips), ("test", test_clips)]:
        out_path = args.out_dir / f"{split_name}.csv"
        split_rows = [r for r in rows if r["utt_id"].split("_")[0] in clips]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows)
        print(f"[split_manifest] {split_name}: {len(clips)}클립 / {len(split_rows)}발화 -> {out_path}")


if __name__ == "__main__":
    main()
