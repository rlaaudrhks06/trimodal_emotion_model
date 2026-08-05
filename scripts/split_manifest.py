"""매니페스트 CSV를 train/val/test로 분할.

utt_id는 "{clip_id}_{person_id}_{start}_{end}" 형식이다.

**분할 단위 세 가지 (--unit)**

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

3. `speaker` (가장 엄격, 권장): `block`으로 묶은 뒤, **화자를 공유하는 블록들을
   하나의 그룹으로 병합**해서 분할한다.

   실측 결과 `block` 단위만으로는 화자 중복이 278명 -> 4명으로 줄지만 0이 되지는
   않았다. 같은 person_id가 두 개 이상의 블록에 등장하는 경우가 남아 있기 때문이다
   (같은 배우가 여러 원본 영상에 출연했거나 번호가 재사용된 경우 — 어느 쪽이든
   "그 번호가 train과 test 양쪽에 있다"는 사실 자체가 문제다).

   블록을 노드로, "같은 person_id를 공유함"을 간선으로 보는 그래프의 **연결 요소
   (connected component)** 단위로 분할하면, 원인과 무관하게 화자 중복이 0이 된다.

분할 후에는 `--verify`로 실제 화자 중복 여부를 반드시 확인할 것.

사용 예 (화자 독립 분할):
    python scripts/split_manifest.py --manifest data/manifests/all.csv \\
        --out-dir data/manifests_si --unit speaker --verify
"""
import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.datasets.labels import normalize_label

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


def merge_groups_sharing_speakers(rows: list[dict], base_unit: str = "block") -> dict[str, str]:
    """화자를 공유하는 그룹들을 하나로 병합한 매핑(원래 그룹 -> 병합된 그룹)을 만든다.

    union-find로 "같은 person_id가 등장하는 그룹들"을 이어붙인다. 결과적으로
    한 화자의 발화는 반드시 하나의 병합 그룹 안에만 존재하게 되어,
    그룹 단위로 나누면 화자가 split 간에 겹칠 수 없다.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # 경로 압축
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # person_id -> 그 화자가 등장하는 그룹들
    by_speaker = defaultdict(set)
    all_groups = set()
    for row in rows:
        g = group_key(row["utt_id"], base_unit)
        all_groups.add(g)
        find(g)
        _, person_id = parse_utt_id(row["utt_id"])
        by_speaker[person_id].add(g)

    n_merged_speakers = 0
    for person_id, groups in by_speaker.items():
        if len(groups) > 1:
            n_merged_speakers += 1
            it = iter(groups)
            first = next(it)
            for g in it:
                union(first, g)

    mapping = {g: find(g) for g in all_groups}
    n_components = len(set(mapping.values()))
    print(f"[split_manifest] {base_unit} {len(all_groups)}개 -> 화자 공유 병합 후 {n_components}개 그룹")
    if n_merged_speakers:
        print(f"[split_manifest]   (여러 {base_unit}에 걸쳐 등장하는 화자 {n_merged_speakers}명 때문에 병합 발생)")
    return mapping


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
        print("\n[verify] 실패 — 화자가 중복됨. --unit speaker 로 다시 시도할 것 "
              "(블록을 넘나드는 화자가 있으면 block 단위만으로는 완전 분리가 안 됨)")

    # 클래스 분포도 같이 본다: 분할 단위가 커질수록(클립 3,937개 -> 화자그룹 138개)
    # 무작위 배정만으로 클래스 비율이 틀어질 여지가 커진다. 비율이 크게 어긋나면
    # 다른 split로 학습한 결과끼리 비교할 때 그 차이가 성능 차이로 오해된다.
    #
    # normalize_label을 거쳐야 한다 — 매니페스트 CSV엔 "contempt"가 그대로 남아있지만
    # 모델은 이를 disgust로 병합해서 본다(8.10절). 원본 라벨로 세면 모델이 실제로
    # 마주하는 분포와 다른 숫자를 보게 된다.
    print("\n[verify] split별 클래스 비율(%) — 모델이 보는 라벨 기준(contempt는 disgust에 병합):")
    dists = {}
    for name, rows in splits.items():
        counts = Counter(normalize_label(str(r["label"]).strip().lower()) for r in rows)
        total = sum(counts.values())
        dists[name] = {k: 100 * v / total for k, v in counts.items()}
    all_labels = sorted({k for d in dists.values() for k in d})
    print(f"  {'label':10s}" + "".join(f"{n:>9s}" for n in splits) + f"{'최대차':>9s}")
    worst = 0.0
    for lab in all_labels:
        vals = [dists[n].get(lab, 0.0) for n in splits]
        gap = max(vals) - min(vals)
        worst = max(worst, gap)
        print(f"  {lab:10s}" + "".join(f"{v:>8.2f} " for v in vals) + f"{gap:>8.2f} ")
    verdict = "양호" if worst < 3 else ("주의" if worst < 6 else "심함 — seed를 바꿔 재분할 검토")
    print(f"  -> 클래스별 최대 편차 {worst:.2f}%p ({verdict})")

    # 다수 클래스 기준선: "무조건 최다 클래스만 찍기"로 얻는 정확도.
    # split이 바뀌면 이 값도 바뀌므로, 모델 정확도를 해석할 기준점으로 함께 기록해야 한다.
    for name in splits:
        top_label, top_pct = max(dists[name].items(), key=lambda kv: kv[1])
        print(f"  {name:5s} 다수 클래스 기준선: {top_pct:.2f}% ({top_label})")

    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--unit", choices=["clip", "block", "speaker"], default="clip",
        help=("분할 단위. clip=기존 방식(v1~v9), block=40클립 블록, "
              "speaker=화자를 공유하는 블록들을 병합(가장 엄격, 8.14절 권장)"),
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="분할 후 train/val/test 간 화자(person_id) 중복 여부를 검사",
    )
    args = parser.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.unit == "speaker":
        # 블록으로 1차 묶은 뒤, 화자를 공유하는 블록들을 연결 요소로 병합
        merge_map = merge_groups_sharing_speakers(rows, base_unit="block")
        key_of = lambda utt_id: merge_map[group_key(utt_id, "block")]
    else:
        key_of = lambda utt_id: group_key(utt_id, args.unit)

    by_group = defaultdict(list)
    for row in rows:
        by_group[key_of(row["utt_id"])].append(row)

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

    unit_label = {"clip": "클립", "block": f"블록({CLIPS_PER_BLOCK}클립)",
                  "speaker": "화자그룹"}[args.unit]
    print(f"[split_manifest] 분할 단위: {args.unit} — 총 {n}개 {unit_label}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = rows[0].keys()
    splits = {}

    for split_name in ["train", "val", "test"]:
        groups = assigned[split_name]
        split_rows = [r for r in rows if key_of(r["utt_id"]) in groups]
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
