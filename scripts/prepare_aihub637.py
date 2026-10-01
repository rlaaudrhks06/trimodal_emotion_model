"""AI Hub 637 「감정 음성 데이터셋」(SKT) -> 16kHz wav + 학습 매니페스트.

## 쓸 수 있는 것과 없는 것 (실측)

    large.tar  성우 8명 × 18,960발화 · script.txt 있음  → **쓴다**
    small.tar  일반인 501명 × 160발화 · 라벨/대본 **없음** → 못 쓴다 (wav만 들어 있다)

## script.txt 형식 (4줄 1발화)

    F0001_000001 NEUTRAL #지문
    전화가 끊어지자||HL 한숨을 내쉬는||M 대규||||HL      ← 전사문(끊어읽기 태그 섞임)
    저놔가 끄너지자||HL ...                              ← 발음 전사
    (빈 줄)

## 감정 16종 중 우리 7클래스로 **깨끗하게** 매핑되는 것만 쓴다

    NEUTRAL→neutral  SAD→sad  ANGRY→angry  JOY→happy  SURPRISE→surprise
    FEAR→fear        UNPLEASURE→disgust
    KIND·TEASE·DRY·SERIOUS·DOUBT·ANXIOUS·SHY·HURRY·HESITATE 는 대응이 없다 — 버린다.
    (ANXIOUS를 fear로 밀어넣는 식의 억지 매핑은 라벨 잡음이다. 1차에서 경멸을 억지로
     붙잡고 있다가 7클래스로 되돌린 경험이 있다.)

## 영상 없음 — 263과 같은 경로

    face_frames_dir을 빈 값으로 둔다. 데이터셋이 검은 프레임 1장 + 유효 마스크를 주고,
    이는 모달리티 드롭아웃이 영상을 지울 때의 입력과 torch.equal로 같다.

## 주의 (결과 해석에 필요)

  · 화자가 **8명뿐**이다. 263(43,991발화·다수 화자)과 성격이 다르다.
  · NEUTRAL은 전부 #지문(나레이션 낭독)이라 대화체 중립과 분포가 다르다 —
    `--drop-narration`으로 뺄 수 있게 해뒀다.
  · 8명이 **같은 문장**을 읽었다. 텍스트는 8번 중복되고 음성만 다르다.

    python scripts/prepare_aihub637.py --large "/data/work/aihub637/f51986/SKT데이터/large.tar" \
        --out-dir data/processed_ext/aihub637 --manifest data/manifests_ext/aihub637.csv --workers 16
"""
from __future__ import annotations

import argparse, collections, re, sys, tarfile, tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.datasets.labels import normalize_label  # noqa: E402

EMO_MAP = {"NEUTRAL": "neutral", "SAD": "sad", "ANGRY": "angry", "JOY": "happy",
           "SURPRISE": "surprise", "FEAR": "fear", "UNPLEASURE": "disgust"}
HEAD = re.compile(r"^([FM]\d+_\d+)\s+([A-Z]+)\s*(#\S+)?\s*$")
TAG = re.compile(r"\|+[A-Z]*")   # 끊어읽기 태그: "||HL", "|||M", "||||LHL"
SR = 16000


def parse_script(text: str) -> dict[str, tuple[str, str, str]]:
    """script.txt -> {utt: (감정 대분류, 소분류, 전사문)}. 전사문의 끊어읽기 태그는 지운다."""
    out, lines = {}, text.split("\n")
    for i, ln in enumerate(lines):
        m = HEAD.match(ln)
        if not m:
            continue
        raw = lines[i + 1] if i + 1 < len(lines) else ""
        out[m.group(1)] = (m.group(2), m.group(3) or "", re.sub(r"\s+", " ", TAG.sub(" ", raw)).strip())
    return out


def one_speaker(tar_path: str, member: str, out_dir: str, drop_narration: bool) -> tuple[list[dict], dict]:
    """large.tar 안의 화자 tar 하나 -> 행 목록. 48kHz wav를 16kHz로 리샘플해 저장한다."""
    stat = collections.defaultdict(int)
    rows = []
    with tarfile.open(tar_path) as outer:
        tmp = Path(tempfile.gettempdir()) / Path(member).name
        with outer.extractfile(member) as src, open(tmp, "wb") as dst:
            while (b := src.read(1 << 22)):
                dst.write(b)
    try:
        with tarfile.open(tmp) as t:
            names = t.getnames()
            spk = names[0].split("/")[0]
            sc = [n for n in names if n.endswith("script.txt")]
            if not sc:
                stat["script없음"] += 1
                return rows, dict(stat)
            script = parse_script(t.extractfile(sc[0]).read().decode("utf-8", "replace"))
            wdir = Path(out_dir) / "audio"
            wdir.mkdir(parents=True, exist_ok=True)
            for n in names:
                if not n.endswith(".wav"):
                    continue
                uid = Path(n).stem
                meta = script.get(uid)
                if meta is None:
                    stat["대본에없음"] += 1
                    continue
                emo, sub, txt = meta
                if drop_narration and sub == "#지문":
                    stat["지문제외"] += 1
                    continue
                if emo not in EMO_MAP:
                    stat[f"매핑없음:{emo}"] += 1
                    continue
                if not txt:
                    stat["전사문없음"] += 1
                    continue
                y, sr = sf.read(t.extractfile(n), dtype="float32")
                if y.ndim > 1:
                    y = y.mean(axis=1)
                if sr != SR:
                    y = librosa.resample(y, orig_sr=sr, target_sr=SR, res_type="soxr_hq")
                if len(y) < SR * 0.3:
                    stat["너무짧음"] += 1
                    continue
                wav_path = wdir / f"{uid}.wav"
                sf.write(wav_path, y, SR, subtype="PCM_16")
                rows.append({"utt_id": f"k637_{uid}", "label": normalize_label(EMO_MAP[emo]),
                             "wav_path": str(wav_path), "text": txt, "face_frames_dir": "",
                             "emo_raw": emo, "sub": sub, "speaker": spk, "dur": round(len(y) / SR, 2)})
                stat["성공"] += 1
    finally:
        tmp.unlink(missing_ok=True)
    return rows, dict(stat)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--large", required=True, help="large.tar 경로")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--drop-narration", action="store_true", help="#지문(나레이션 낭독) 제외")
    args = ap.parse_args()

    with tarfile.open(args.large) as t:
        members = [n for n in t.getnames() if n.endswith(".tar")]
    print(f"[prep637] 화자 tar {len(members)}개: {[Path(m).stem for m in members]}", flush=True)

    rows, stat = [], collections.defaultdict(int)
    with ProcessPoolExecutor(max_workers=min(args.workers, len(members))) as ex:
        for r, s in ex.map(one_speaker, [args.large] * len(members), members,
                           [args.out_dir] * len(members), [args.drop_narration] * len(members)):
            rows += r
            for k, v in s.items():
                stat[k] += v
            print(f"[prep637] 누적 발화 {len(rows):,}", flush=True)

    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.manifest, index=False)
    print(f"[prep637] {args.manifest} · 발화 {len(df):,}", flush=True)
    print("[prep637] 집계:", dict(sorted(stat.items(), key=lambda kv: -kv[1])), flush=True)
    if len(df):
        print("[prep637] 라벨:", df["label"].value_counts().to_dict(), flush=True)
        print(f"[prep637] 화자 {df['speaker'].nunique()}명 · 총 {df['dur'].sum()/3600:.1f}시간 "
              f"· 길이 중앙값 {df['dur'].median():.1f}초", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
