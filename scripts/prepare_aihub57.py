"""AI Hub 57 「멀티모달」(2018) 원본 zip -> 발화 단위 매니페스트 + 16kHz wav + 정렬된 얼굴 프레임.

58(지금 쓰는 데이터)과 같은 형식으로 만들어 train에만 더한다. val·test는 건드리지 않는다.

원본 구조 (zip 안, 무압축 store라 직접 읽는다):
    {clip}/{clip}.mp4
    {clip}/{clip}_interpolation.json          ← 마스터. 작업자별 json은 분업본이라 한쪽만 채워져 있다
    {clip}/{clip}/{shot}/KM_{frame_id:010d}.jpg
JSON:
    common_info.frame_size                     클립 총 프레임 수
    dialogue_infos[]  start_time/end_time("HH:MM:SS.mmm") · speaker_id(배우 이름) · utterance
    shot_infos[].visual_infos[]  frame_id · persons[].person_info{face_rect, emotion{8종 0~10}}

58과 다른 점 셋:
  ① **face_rect를 직접 준다**(58은 인물 전신 bbox뿐이라 8.8절 사건이 났다). 그래도 그대로 쓰지 않고
     그 영역 안에서 mediapipe로 다시 검출·정렬한다 — 58과 입력 분포를 맞추려면 같은 경로여야 한다.
  ② 감정이 8종 강도(0~10)다. 발화 구간에서 **화자 본인의** 강도를 프레임마다 더해 최대값을 라벨로 쓴다.
     경멸→혐오 병합은 labels.normalize_label이 처리한다(§11).
  ③ 화자가 배우 이름 문자열이고 'None'·공백이 섞여 있다. 정규화 후 전역 번호를 붙인다.

    python scripts/prepare_aihub57.py --zips "/data/work/aihub57/f*/09.멀티모달/*.zip" \
        --out-dir data/processed_ext/aihub57 --manifest data/manifests_ext/aihub57.csv --workers 16
"""
import argparse, csv, glob, hashlib, json, re, sys, tempfile, zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.datasets.labels import normalize_label  # noqa: E402
from src.features.face_align import create_face_detector, detect_and_align_face, ensure_face_detector_model  # noqa: E402

# 57의 감정 키 -> 우리 표기. contempt는 normalize_label이 disgust로 흡수한다.
EMO = {"happiness": "happy", "sadness": "sad", "anger": "angry", "surprise": "surprise",
       "afraid": "fear", "contempt": "contempt", "disgust": "disgust", "neutral": "neutral"}
MAX_FRAMES = 24          # 58과 같은 발화당 최대 프레임 수
SR = 16000


def norm_speaker(s) -> str | None:
    """배우 이름 정규화. 'None'·'none'·빈 값은 화자 없음으로 본다."""
    s = re.sub(r"\s+", "", str(s or ""))
    return None if s.lower() in ("", "none", "nan") else s


def t2s(t: str) -> float:
    """'HH:MM:SS.mmm' -> 초."""
    h, m, s = str(t).split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def decode_audio(mp4_path: Path) -> tuple[np.ndarray, float]:
    """mp4 -> (16kHz 모노 float32, 영상 fps). ffmpeg 바이너리 없이 PyAV로만 한다."""
    import av
    with av.open(str(mp4_path)) as c:
        vs = c.streams.video[0]
        fps = float(vs.average_rate) if vs.average_rate else 0.0
        if not c.streams.audio:
            return np.zeros(0, dtype=np.float32), fps
        a = c.streams.audio[0]
        rs = av.AudioResampler(format="s16", layout="mono", rate=SR)
        chunks = []
        for frame in c.decode(a):
            for out in rs.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in rs.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    wav = np.concatenate(chunks).astype(np.float32) / 32768.0 if chunks else np.zeros(0, dtype=np.float32)
    return wav, fps


def one_clip(zip_path: str, clip: str, out_dir: str, face_size: int, model_path: str) -> tuple[list[dict], dict]:
    """클립 하나 -> 발화 행 목록. 실패 사유는 카운터로 돌려준다(조용히 사라지지 않게)."""
    stat = defaultdict(int)
    rows = []
    z = zipfile.ZipFile(zip_path)
    try:
        d = json.loads(z.read(f"{clip}/{clip}_interpolation.json").decode("utf-8-sig"))
    except KeyError:
        stat["json없음"] += 1
        return rows, dict(stat)
    dialogues = d.get("dialogue_infos") or []
    if not dialogues:
        stat["대사없음"] += 1
        return rows, dict(stat)

    # frame_id -> (jpg 경로, 그 프레임의 화자별 face_rect·emotion)
    frame_info: dict[int, dict] = {}
    for si in (d.get("shot_infos") or []):
        shot = si.get("image_folder")
        for vi in (si.get("visual_infos") or []):
            fid = vi.get("frame_id")
            if fid is None:
                continue
            per = {}
            for p in (vi.get("persons") or []):
                spk = norm_speaker(p.get("person_id"))
                pi = p.get("person_info") or {}
                if spk and pi.get("face_rect"):
                    per[spk] = (pi["face_rect"], pi.get("emotion") or {})
            frame_info[fid] = {"jpg": f"{clip}/{clip}/{shot}/KM_{fid:010d}.jpg", "persons": per}

    tmp_mp4 = Path(tempfile.gettempdir()) / f"{clip}.mp4"
    try:
        with z.open(f"{clip}/{clip}.mp4") as src, open(tmp_mp4, "wb") as dst:
            while (b := src.read(1 << 22)):
                dst.write(b)
        wav, fps = decode_audio(tmp_mp4)
    except Exception:
        stat["mp4실패"] += 1
        tmp_mp4.unlink(missing_ok=True)
        return rows, dict(stat)
    finally:
        tmp_mp4.unlink(missing_ok=True)
    if fps <= 0 or wav.size == 0:
        stat["fps/오디오없음"] += 1
        return rows, dict(stat)

    detector = create_face_detector(Path(model_path))
    out = Path(out_dir)
    for dl in dialogues:
        spk = norm_speaker(dl.get("speaker_id"))
        text = str(dl.get("utterance") or "").strip()
        if not spk or not text:
            stat["화자/대사없음"] += 1
            continue
        try:
            t0, t1 = t2s(dl["start_time"]), t2s(dl["end_time"])
        except Exception:
            stat["시간파싱실패"] += 1
            continue
        if t1 - t0 < 0.3:
            stat["너무짧음"] += 1
            continue
        f0, f1 = int(round(t0 * fps)), int(round(t1 * fps))
        fids = [f for f in range(f0, f1 + 1) if f in frame_info and spk in frame_info[f]["persons"]]
        if not fids:
            stat["구간에화자프레임없음"] += 1
            continue

        # 라벨: 구간 안 화자 본인의 감정 강도를 합산해 최대값
        acc = defaultdict(float)
        for f in fids:
            for k, v in frame_info[f]["persons"][spk][1].items():
                if k in EMO:
                    acc[EMO[k]] += float(v or 0)
        if not acc or max(acc.values()) <= 0:
            stat["감정전부0"] += 1
            continue
        label = normalize_label(max(acc, key=acc.get))

        # utt_id는 58과 같은 "{clip}_{person}_{start}_{end}" 4토막 형식이어야 한다
        # (split_manifest·check_dataset_overlap이 그 규칙으로 파싱한다).
        # 화자 번호는 hash()를 쓰면 프로세스마다 달라져 재현이 깨지므로 이름의 md5 앞자리를 쓴다.
        spk_num = int(hashlib.md5(spk.encode()).hexdigest()[:6], 16) % 100000
        utt = f"57{int(re.sub(r'[^0-9]', '', clip)):d}_{spk_num}_{int(t0*1000)}_{int(t1*1000)}"
        fdir = out / "faces" / utt
        step = max(1, len(fids) // MAX_FRAMES)
        saved = 0
        for f in fids[::step][:MAX_FRAMES]:
            try:
                buf = np.frombuffer(z.read(frame_info[f]["jpg"]), np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            except KeyError:
                continue
            if img is None:
                continue
            r = frame_info[f]["persons"][spk][0]
            box = (int(r["min_x"]), int(r["min_y"]), int(r["max_x"]), int(r["max_y"]))
            face = detect_and_align_face(img, box, face_size=face_size, detector=detector)
            if face is None:
                continue
            fdir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(fdir / f"frame_{saved:03d}.jpg"), face)
            saved += 1
        if saved == 0:
            stat["얼굴검출0"] += 1
            continue

        seg = wav[int(t0 * SR): int(t1 * SR)]
        if seg.size < SR * 0.3:
            stat["오디오짧음"] += 1
            continue
        wdir = out / "audio"
        wdir.mkdir(parents=True, exist_ok=True)
        sf.write(str(wdir / f"{utt}.wav"), seg, SR)
        rows.append({"utt_id": utt, "label": label, "wav_path": str(wdir / f"{utt}.wav"),
                     "text": text, "face_frames_dir": str(fdir),
                     "label_image": "", "label_sound": "", "label_text": "",
                     "speaker_name": spk, "clip": clip, "n_frames": saved})
        stat["성공"] += 1
    return rows, dict(stat)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", required=True, help="glob 패턴")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-clips", type=int, default=0, help="0이면 전부 (소규모 시험용)")
    ap.add_argument("--face-size", type=int, default=112)
    args = ap.parse_args()

    model_path = str(ensure_face_detector_model())
    zips = sorted(glob.glob(args.zips))
    print(f"[prep57] zip {len(zips)}개", flush=True)
    tasks = []
    for zp in zips:
        with zipfile.ZipFile(zp) as z:
            clips = sorted({n.split("/")[0] for n in z.namelist()
                            if n.endswith("_interpolation.json") and not n.split("/")[-1].startswith("._")})
        tasks += [(zp, c) for c in clips]
    if args.limit_clips:
        tasks = tasks[:args.limit_clips]
    print(f"[prep57] 클립 {len(tasks)}개 처리 시작 (워커 {args.workers})", flush=True)

    rows, stat, done = [], defaultdict(int), 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one_clip, zp, c, args.out_dir, args.face_size, model_path): c for zp, c in tasks}
        for fu in as_completed(futs):
            try:
                r, s = fu.result()
            except Exception as e:
                stat[f"예외:{type(e).__name__}"] += 1
                r, s = [], {}
            rows += r
            for k, v in s.items():
                stat[k] += v
            done += 1
            if done % 25 == 0 or done == len(tasks):
                print(f"[prep57] {done}/{len(tasks)} 클립 · 발화 {len(rows):,}", flush=True)

    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    cols = ["utt_id", "label", "wav_path", "text", "face_frames_dir",
            "label_image", "label_sound", "label_text", "speaker_name", "clip", "n_frames"]
    with open(args.manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[prep57] 매니페스트 {args.manifest} · 발화 {len(rows):,}", flush=True)
    print("[prep57] 집계:", dict(sorted(stat.items(), key=lambda kv: -kv[1])), flush=True)
    if rows:
        import pandas as pd
        d = pd.DataFrame(rows)
        print("[prep57] 라벨 분포:", d["label"].value_counts().to_dict(), flush=True)
        print(f"[prep57] 화자 {d["speaker_name"].nunique()}명 · 클립 {d["clip"].nunique()}개 "
              f"· 프레임 중앙값 {int(d["n_frames"].median())}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
