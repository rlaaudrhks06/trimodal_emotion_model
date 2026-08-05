"""매니페스트 CSV를 train/val/test로 분할.

utt_id는 "{clip_id}_{person_id}_{start}_{end}" 형식이다.

**분할 단위 두 가지 (--unit)**

1. `clip` (기존 방식, v1~v9가 쓴 것): clip_id 단위로 통째로 분할.
   같은 클립이 여러 발화로 쪼개져 train/val/test에 걸치는 것을 막는다.

2. `block` (권장, 8.14절): 40클립 블록 = 원본 영상 1편 단위로 분할.

   `clip` 방식이 화자 독립성을 보장하지 못한다는 것이 뒤늦게 밝혀졌다(통합기록 8.14절).
   AI Hub "멀티모달 영상" 데이터는 40개 클립이 한 블록(원본 영상 1편)을 이루고,
   person_id는 그 블록에 등장하는 특정 인물에게 전역적으로 고유하게 부여된 번호다
   (실측: person_id=114 -> 클립 2241~2280, person_id=90 -> 클립 1761~1800,
   각각 정확히 40클립 연속 구간이며 서로 겹치지 않음).

   따라서 클립을 무작위로 섞으면 같은 사람의 발화가 train/val/test에 모두 흩어진다.
   실제로 기존 분할에서는 test에 등장하는 화자 278명이 **전원** train에도 존재했다 —
   즉 "처음 보는 화자"가 한 명도 없었고, 설계 문서 3.8.1절이 명시한
   subject-independent 조건이 전혀 지켜지지 않은 상태였다.

   블록 단위로 나누면 화자뿐 아니라 배경·조명·촬영 조건까지 함께 분리되므로,
   person_id 단위 분할보다 안전하다(한 클립에 화자가 둘 이상일 때 클립이 쪼개지는
   문제도 없다).

분할 후에는 `--verify`로 실제 화자 중복 여부를 반드시 확인할 것.

사용 예 (화자 독립 분할):
    python scripts/split_manifest.py --manifest data/manifests/all.csv \\
        --out-dir data/manifests_speaker_independent --unit block --verify
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

# AI Hub 멀티모달 영상 데이터의 블록 크기(원본 영상 1편당 클립 수). 8.14.1절 실측 근거.
CLIPS_PER_BLOCK = 40


def parse_utt_id(utt_id: str) -> tuple[str, str]:
    """utt_id -> (clip_id, person_id)."""
    parts = utt_id.split("_")
    return parts[0], (parts[1] if len(parts) > 1 else "")


def group_key(utt_id: str, unit: str) -> str:
    clip_id, _ = parse_utt_id(utt_id)
    if unit == "clip":
        return clip_id
    # block: clip_id가 1부터 시작하는 연속 번호라고 보고 40개씩 묶는다.
    # 숫자가 아닌 clip_id(다른 데이터셋 등)는 그 자체를 하나의 그룹으로 둔다.
    try:
        return str((int(clip_id) - 1) // CLIPS_PER_BLOCK)
    except ValueError:
        return clip_id


def verify_speaker_independence(splits: dict[str, list[dict]]) -> bool:
    """train/val/test 간 person_id가 겹치는지 확인하고 결과를 출력한다."""
    speakers = {
        name: {parse_utt_id(r["utt_id"])[1] for r in rows}
        for name, rows in splits.items()
    }
    print("\n[verify] split별 화자(person_id) 수:")
    for name, s in speakers.items():
        print(f"  {name:5s}: {len(s)}명")

    ok = True
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = speakers[a] & speakers[b]
        status = "OK" if not overlap else "겹침!"
        print(f"  {a} ∩ {b}: {len(overlap)}명  [{status}]")
        if overlap:
            ok = False

    if ok:
        print("\n[verify] 통과 — 화자가 세 split에 걸쳐 중복되지 않음 (subject-independent)")
    else:
        print("\n[verify] 실패 — 화자가 중복됨. --unit block 을 썼는지, "
              "clip_id 번호 체계가 가정(1부터 연속)과 맞는지 확인 필요")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--unit", choices=["clip", "block"], default="clip",
        help="분할 단위. clip=기존 방식(v1~v9), block=40클립 블록 단위(화자 독립, 8.14절 권장)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="분할 후 train/val/test 간 화자(person_id) 중복 여부를 검사",
    )
    args = parser.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_group = defaultdict(list)
    for row in rows:
        by_group[group_key(row["utt_id"], args.unit)].append(row)

    group_ids = list(by_group.keys())
    random.Random(args.seed).shuffle(group_ids)

    n = len(group_ids)
    n_val = max(1, int(n * args.val_ratio))
    n_test = max(1, int(n * args.test_ratio))
    assigned = {
        "val": set(group_ids[:n_val]),
        "test": set(group_ids[n_val:n_val + n_test]),
        "train": set(group_ids[n_val + n_test:]),
    }

    unit_label = "클립" if args.unit == "clip" else f"블록({CLIPS_PER_BLOCK}클립)"
    print(f"[split_manifest] 분할 단위: {args.unit} — 총 {n}개 {unit_label}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = rows[0].keys()
    splits = {}

    for split_name in ["train", "val", "test"]:
        groups = assigned[split_name]
        split_rows = [r for r in rows if group_key(r["utt_id"], args.unit) in groups]
        splits[split_name] = split_rows
        out_path = args.out_dir / f"{split_name}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows)
        print(f"[split_manifest] {split_name}: {len(groups)}개 {unit_label} / {len(split_rows)}발화 -> {out_path}")

    if args.verify:
        verify_speaker_independence(splits)


if __name__ == "__main__":
    main()
