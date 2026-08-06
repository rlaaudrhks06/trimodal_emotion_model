"""AI Hub 원본 JSON에서 모달리티별 감정 라벨을 뽑아 매니페스트에 컬럼으로 붙인다.

**왜 필요한가 (v12의 출발점)**

AI Hub 원본은 발화마다 감정 라벨을 네 개 갖고 있다 — image / sound / text / multimodal.
우리는 지금까지 multimodal 하나만 정답으로 써왔는데, 5,637개를 집계해보니 나머지 셋이
그것과 얼마나 일치하는지가 극단적으로 달랐다:

    소리 라벨  77.35% 일치
    영상 라벨  41.69% 일치
    텍스트 라벨 30.87% 일치
    넷 다 일치  10.20%

즉 우리가 쓰는 정답은 사실상 "소리로 판단한 감정"에 가깝다. 그런데 모델은 세 브랜치
모두에게 이 정답 하나를 맞히라고 요구한다. 영상 브랜치는 **화면이 58% 반박하는 답**을
내놓아야 하므로 배울 근거가 없고, 남는 방법은 촬영 맥락(배경·조명·인물)을 외우는 것뿐이다.
실제로 프로빙에서 세 브랜치 모두 감정보다 화자/촬영분 정보를 약 3배 더 담고 있었다.

v12는 각 브랜치에 **자기 입력에 답이 있는 보조 과제**를 준다. 이 스크립트는 그 보조
라벨을 매니페스트에 채워 넣는다.

**기존 컬럼은 건드리지 않는다.** label(=multimodal)은 그대로 두고 뒤에 세 컬럼만 붙이므로,
이 파일을 읽는 기존 코드는 전부 그대로 동작한다.

원본을 못 찾은 발화는 빈 값으로 남긴다(5201-5600 배치처럼 원본이 소실됐다 복구된 경우
등). 학습 쪽에서 빈 값은 보조 손실에서 제외한다.

실행 예:
    python scripts/add_modality_labels.py \
        --raw-dir data/raw/aihub_full \
        --manifests data/manifests/all.csv \
                    data/manifests_si/train.csv \
                    data/manifests_si/val.csv \
                    data/manifests_si/test.csv
"""
import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.labels import EMOTION_LABELS, normalize_label

MODALITIES = ["image", "sound", "text"]
NEW_COLUMNS = [f"label_{m}" for m in MODALITIES]


def read_json(path: Path) -> dict | None:
    """AI Hub JSON을 읽는다. 일부 클립이 CP949로 저장돼 있어 UTF-8 실패 시 재시도한다
    (build_manifest_aihub.py:69-75와 같은 처리)."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp949")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def collect_labels(raw_dir: Path) -> dict[str, dict[str, str]]:
    """원본 전체를 훑어 {utt_id: {image/sound/text: 라벨}} 사전을 만든다.

    utt_id 조립 규칙은 build_manifest_aihub.py:217과 반드시 같아야 한다
    ("{clip_id}_{person_id}_{script_start}_{script_end}"). 다르면 매니페스트와
    한 건도 안 맞는데, 그건 "원본을 못 찾음"과 겉보기가 같아서 조용히 실패한다.
    그래서 main()에서 매칭률을 반드시 출력하고 낮으면 경고한다.
    """
    files = sorted(raw_dir.glob("**/clip_*/clip_*.json"))
    if not files:  # 샘플 배포 레이아웃(라벨데이터/clip_13.json)도 지원
        files = sorted(raw_dir.glob("**/라벨데이터/clip_*.json"))
    print(f"[add_labels] JSON {len(files):,}개 스캔", flush=True)

    out: dict[str, dict[str, str]] = {}
    # 실패는 **발화 단위로** 센다. 같은 발화가 여러 프레임에 반복 등장하는데, 첫 시도가
    # 실패하면 out에 안 들어가서 다음 프레임에서 또 시도하고 또 세어진다. 단순 카운터로
    # 세면 "13만 개 제외"처럼 실제보다 몇 배 부풀려진 수가 찍혀 데이터를 크게 잃은 것으로
    # 오해하게 된다(실측: 카운터 130,515 vs 실제 수집 78,848 + 매칭률 98.3%).
    failed_utts: set[str] = set()
    # 일부 모달리티만 없는 경우도 따로 센다 — 어느 모달리티가 얼마나 비는지 보여야
    # 보조 손실에서 그 항이 얼마나 빠지는지 가늠할 수 있다.
    per_modality_missing: Counter = Counter()
    n_bad_json = 0
    for i, fp in enumerate(files, 1):
        d = read_json(fp)
        if d is None:
            n_bad_json += 1
            continue
        clip_id = d.get("clip_id")
        for objs in d.get("data", {}).values():
            for o in objs.values():
                tx = o.get("text")
                if not isinstance(tx, dict) or not tx.get("script"):
                    continue
                try:
                    utt_id = (f"{clip_id}_{o['person_id']}_"
                              f"{int(tx['script_start'])}_{int(tx['script_end'])}")
                except (KeyError, TypeError, ValueError):
                    continue
                if utt_id in out:
                    continue  # 같은 발화가 여러 프레임에 반복 등장 — 첫 것만
                em = o.get("emotion", {})

                # **모달리티마다 독립적으로 판정한다.** 처음엔 셋을 한꺼번에 읽어 하나라도
                # 실패하면 통째로 버렸는데, 실측해보니 실패는 전부 `image` 필드 부재였고
                # `sound`·`text`는 멀쩡히 있었다. 즉 정답과 가장 잘 맞는 소리 라벨(74%)까지
                # image가 없다는 이유로 버리고 있었다. 있는 것만 채우고 없는 것만 비운다.
                labs = {}
                for m in MODALITIES:
                    node = em.get(m)
                    if not isinstance(node, dict) or "emotion" not in node:
                        continue
                    v = normalize_label(node["emotion"])
                    # 병합/별칭을 거쳤는데도 우리 체계에 없는 값은 그 모달리티만 비운다.
                    if v in EMOTION_LABELS:
                        labs[m] = v
                if not labs:
                    failed_utts.add(utt_id)  # 셋 다 못 얻은 발화만 실패로 센다
                    continue
                per_modality_missing.update(m for m in MODALITIES if m not in labs)
                out[utt_id] = labs
        if i % 500 == 0:
            print(f"    {i:,}/{len(files):,} 클립  (누적 발화 {len(out):,})", flush=True)

    if n_bad_json:
        print(f"[add_labels] 경고: JSON 파싱 실패 {n_bad_json}개")
    # 성공 뒤 다른 프레임에서 실패한 경우도 있으므로, 끝내 못 얻은 것만 남긴다
    failed_utts -= out.keys()
    if failed_utts:
        print(f"[add_labels] 세 모달리티 라벨을 하나도 못 얻은 발화 {len(failed_utts):,}개 제외")
    if per_modality_missing:
        detail = ", ".join(f"{m} {n:,}개" for m, n in per_modality_missing.most_common())
        print(f"[add_labels] 일부 모달리티만 빈 발화: {detail} (나머지 모달리티는 정상 사용)")
    return out


def annotate(manifest: Path, table: dict[str, dict[str, str]], backup: bool) -> tuple[int, int]:
    """매니페스트에 컬럼 3개를 붙여 제자리에서 다시 쓴다. 반환: (전체, 채워진 수)."""
    with open(manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0, 0

    fieldnames = [c for c in rows[0] if c not in NEW_COLUMNS] + NEW_COLUMNS
    filled = 0
    for r in rows:
        labs = table.get(str(r["utt_id"]))
        if labs:
            filled += 1
        for m in MODALITIES:
            # labs는 얻은 모달리티만 담고 있다(부분 성공 허용). 없는 것은 빈 값으로 둔다.
            r[f"label_{m}"] = labs.get(m, "") if labs else ""

    if backup:
        shutil.copy2(manifest, manifest.with_suffix(manifest.suffix + ".bak"))
    # 임시 파일에 먼저 쓰고 원자적으로 교체 — 중간에 죽어도 원본이 반쯤 덮이지 않는다.
    tmp = manifest.with_suffix(manifest.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(manifest)
    return len(rows), filled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--manifests", nargs="+", type=Path, required=True)
    parser.add_argument("--no-backup", action="store_true",
                        help="원본 .bak 백업을 만들지 않는다(기본은 만든다)")
    parser.add_argument("--dry-run", action="store_true",
                        help="파일을 쓰지 않고 매칭률과 일치율만 출력")
    args = parser.parse_args()

    table = collect_labels(args.raw_dir)
    print(f"[add_labels] 원본에서 발화 {len(table):,}개 수집\n")

    for mf in args.manifests:
        if not mf.exists():
            print(f"[add_labels] {mf}: 없음, 건너뜀")
            continue
        with open(mf, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        hit = sum(1 for r in rows if str(r["utt_id"]) in table)
        rate = 100 * hit / len(rows) if rows else 0
        print(f"[add_labels] {mf}: {len(rows):,}행 중 {hit:,}건 매칭 ({rate:.1f}%)")

        # 매칭률이 낮으면 원본 부재가 아니라 utt_id 조립 규칙이 어긋났을 가능성이 크다.
        # 그 경우 컬럼이 전부 빈 채로 조용히 채워지므로 여기서 반드시 멈춘다.
        if rate < 50:
            print(f"    ⚠ 매칭률이 50% 미만이다. 원본이 실제로 없는 건지, utt_id 조립"
                  f" 규칙이 어긋난 건지 확인할 것 (매니페스트 예: {rows[0]['utt_id']})")
            sample = next(iter(table)) if table else "(없음)"
            print(f"      원본에서 만든 utt_id 예: {sample}")
            if not args.dry_run:
                raise SystemExit("[add_labels] 중단 — --dry-run으로 먼저 확인하세요")

        if not args.dry_run:
            n, filled = annotate(mf, table, backup=not args.no_backup)
            print(f"    -> 컬럼 {NEW_COLUMNS} 추가, {filled:,}/{n:,}건 채움"
                  f"{' (.bak 백업 생성)' if not args.no_backup else ''}")

        # 정답(multimodal)과 각 모달리티 라벨의 일치율 — v12 설계의 근거 수치다.
        matched = [r for r in rows if str(r["utt_id"]) in table]
        if matched:
            print(f"    정답 대비 일치율:", end="")
            for m in MODALITIES:
                # 그 모달리티 라벨이 있는 발화만 분모로 쓴다 — 없는 것까지 분모에 넣으면
                # 일치율이 실제보다 낮게 나온다.
                have = [r for r in matched if m in table[str(r["utt_id"])]]
                if not have:
                    print(f"  {m} -", end="")
                    continue
                agree = sum(
                    1 for r in have
                    if table[str(r["utt_id"])][m]
                    == normalize_label(str(r["label"]).strip().lower())
                )
                print(f"  {m} {100*agree/len(have):.2f}%", end="")
            print()
            dist = Counter(v["sound"] for v in (table[str(r["utt_id"])] for r in matched) if "sound" in v)
            top = ", ".join(f"{k} {100*v/len(matched):.1f}%" for k, v in dist.most_common(3))
            print(f"    소리 라벨 분포 상위: {top}")
        print()


if __name__ == "__main__":
    main()
