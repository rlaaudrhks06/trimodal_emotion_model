"""AI Hub "멀티모달 영상"(dataSetSn=58) 원본 -> 매니페스트 CSV 변환 (실제 동작 버전).

공식 데이터 설명서(11. [미디어 영역] 멀티모달 영상(상황,행동).pdf)와 실제 샘플
(2019-01-005.멀티모달영상_sample)을 직접 뜯어본 뒤 확인한 실제 JSON 구조를
기준으로 작성했다.

입력 폴더 구조 (AI Hub 표준):
    raw_dir/
      라벨데이터/clip_13.json, clip_30.json, ...
      원천데이터/0013.MP4, 0030.MP4, ...

JSON 핵심 구조:
    data[frame_key][obj_key] = {
        "person_id", "xtl","ytl","xbr","ybr",   # 인물 bounding box (해당 프레임 순간)
        "text": {"script","script_start","script_end", ...},  # 발화 정보(있으면 발화 구간)
        "emotion": {"image":{...}, "sound":{...}, "text":{...}, "multimodal":{"emotion":...}},
    }
같은 발화가 script_start~script_end 구간의 모든 프레임 키에 걸쳐 반복 등장하므로
(person_id, script_start, script_end)로 중복 제거해야 발화 단위 1건이 된다.

라벨: emotion.multimodal.emotion을 정답으로 사용(트리모달 융합 목표이므로 4개 모달리티
중 이미 사람이 종합 판단한 라벨이 우리 학습 타깃과 가장 잘 맞는다). AI Hub 원본은 혐오를
"dislike"로 표기하므로 labels.RAW_LABEL_ALIASES로 "disgust"에 맞춘다.

사용 예:
    python scripts/build_manifest_aihub.py \
        --raw-dir "../2019-01-005.멀티모달영상_sample" \
        --out data/manifests/sample.csv \
        --frames-out data/processed
"""
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 개인 노트북에서 다른 작업과 같이 돌릴 때만 CPU를 낮게 제한한다.
# THROTTLE_CPU=1 환경변수를 줘야 활성화됨 — 전용 서버(예: A100)에서는 기본값(끔)으로
# 모든 코어를 다 써서 빠르게 처리한다.
THROTTLE_CPU = os.environ.get("THROTTLE_CPU") == "1"
if THROTTLE_CPU:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")

import cv2
import numpy as np
import soundfile as sf
import av

if THROTTLE_CPU:
    cv2.setNumThreads(2)

from src.datasets.labels import RAW_LABEL_ALIASES
from src.features.face_align import create_face_detector, ensure_face_detector_model, extract_aligned_frames

TARGET_SR = 16000
MAX_FACE_FRAMES = 24
SLEEP_BETWEEN_UTTERANCES_SEC = 0.3 if THROTTLE_CPU else 0.0  # 노트북에서만 짬짬이 쉬어감


def load_json_utterances(json_path: Path) -> list[dict]:
    import json

    # 대부분 UTF-8이지만 일부 클립(특히 "-수정본" 접미사 없는 원본)은 CP949로 저장되어 있어
    # UTF-8 디코딩이 바이트 단위로 실패한다. UTF-8 먼저 시도하고 실패하면 CP949로 재시도.
    raw = json_path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp949")
    d = json.loads(text)
    clip_id = d["clip_id"]

    seen = {}
    for frame_key, objs in d["data"].items():
        for obj_key, obj in objs.items():
            text = obj.get("text")
            if not isinstance(text, dict) or not text.get("script"):
                continue
            # 일부 클립은 script_start/end를 문자열로 기록해둬서 int로 통일해야 함
            try:
                script_start = int(text["script_start"])
                script_end = int(text["script_end"])
            except (TypeError, ValueError):
                continue  # 프레임 번호를 못 읽으면 이 발화는 건너뜀
            key = (obj["person_id"], script_start, script_end)
            frame_num = int(obj["frame"])
            # 대표 엔트리로 script_start와 가장 가까운(bbox가 가장 대표성 있는) 프레임을 선택
            if key not in seen or abs(frame_num - script_start) < seen[key]["_dist"]:
                seen[key] = {
                    "clip_id": clip_id,
                    "person_id": obj["person_id"],
                    "script": text["script"],
                    "script_start": script_start,
                    "script_end": script_end,
                    "xtl": float(obj["xtl"]), "ytl": float(obj["ytl"]),
                    "xbr": float(obj["xbr"]), "ybr": float(obj["ybr"]),
                    "emotion_raw": obj["emotion"]["multimodal"]["emotion"],
                    "_dist": abs(frame_num - script_start),
                }

    return list(seen.values())


def extract_face_frames(
    video_path: Path, start_frame: int, end_frame: int, bbox, out_dir: Path,
    face_size: int = 112, detector=None,
) -> int:
    """person bbox 영역 안에서 mediapipe로 실제 얼굴을 검출+정렬해 저장한다.

    §8.8 대응: 예전엔 AI Hub가 제공하는 person(인물) bbox를 그대로 잘라 저장했는데,
    이게 얼굴이 아니라 사람 전신이라서 사전학습 얼굴 인식 백본이 전부 무용지물이었다
    (scripts/refix_face_crops.py로 기존 5,200클립을 사후 수정한 이력 참고). 새로
    추가되는 배치(예: 5201-5600 복구분)는 처음부터 이 방식으로 뽑아 재작업이 없게 한다.
    """
    bbox_int = [max(0, int(v)) for v in bbox]
    cap = cv2.VideoCapture(str(video_path))
    saved = extract_aligned_frames(
        cap, start_frame, end_frame, bbox_int, out_dir, detector,
        face_size=face_size, max_frames=MAX_FACE_FRAMES,
    )
    cap.release()
    return saved


def extract_audio_segment(video_path: Path, start_frame: int, end_frame: int, fps: float, out_wav: Path) -> bool:
    start_sec = start_frame / fps
    end_sec = end_frame / fps

    container = av.open(str(video_path))
    audio_streams = [s for s in container.streams if s.type == "audio"]
    if not audio_streams:
        return False
    stream = audio_streams[0]

    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=TARGET_SR)
    samples = []
    container.seek(int(start_sec * av.time_base), any_frame=False)
    for frame in container.decode(stream):
        if frame.time is not None and frame.time > end_sec:
            break
        if frame.time is not None and frame.time < start_sec - 0.5:
            continue
        for resampled in resampler.resample(frame):
            samples.append(resampled.to_ndarray())

    container.close()
    if not samples:
        return False

    audio = np.concatenate(samples, axis=1).flatten().astype(np.float32) / 32768.0
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), audio, TARGET_SR)
    return True


def discover_json_video_pairs(raw_dir: Path) -> list[tuple[Path, Path]]:
    """AI Hub 배포 레이아웃이 두 가지라 둘 다 지원한다.

    레이아웃 1 (샘플 배포): raw_dir/라벨데이터/clip_13.json, raw_dir/원천데이터/0013.MP4
    레이아웃 2 (정식 zip, 예: 5201-5600.zip): raw_dir/**/clip_5355/clip_5355.json + clip_5355.mp4 (같은 폴더)
    """
    label_dir = raw_dir / "라벨데이터"
    video_dir = raw_dir / "원천데이터"
    if label_dir.exists() and video_dir.exists():
        pairs = []
        for json_path in sorted(label_dir.glob("*.json")):
            clip_num = json_path.stem.replace("clip_", "")
            video_path = video_dir / f"{int(clip_num):04d}.MP4"
            if video_path.exists():
                pairs.append((json_path, video_path))
            else:
                print(f"[build_manifest_aihub] 영상 없음, 건너뜀: {video_path}")
        return pairs

    # 레이아웃 2: clip_XXXX 폴더마다 json+mp4가 같이 있음
    pairs = []
    for json_path in sorted(raw_dir.glob("**/clip_*/clip_*.json")):
        video_path = json_path.with_suffix(".mp4")
        if not video_path.exists():
            video_path = json_path.with_suffix(".MP4")
        if video_path.exists():
            pairs.append((json_path, video_path))
        else:
            print(f"[build_manifest_aihub] 영상 없음, 건너뜀: {json_path.stem}")
    return pairs


def _process_one_clip(json_path: Path, video_path: Path, frames_out: Path, face_model_path: Path) -> tuple[str, list, int, str | None]:
    """워커 프로세스에서 클립 하나를 통째로 처리한다.

    실측 결과(12시간에 2152/5200) 클립을 하나씩 순차 처리하는 게 병목이었다 —
    16코어 서버인데 1코어만 쓰고 있었음. CSV 파일 쓰기만 메인 프로세스가 전담하고,
    (느린) 오디오/영상 추출은 여러 프로세스가 클립 단위로 나눠서 병렬 수행한다.
    반환: (클립이름, 매니페스트 행 목록, 실패한 발화 수, 에러 메시지(있으면))
    """
    try:
        utterances = load_json_utterances(json_path)
        clip_id = utterances[0]["clip_id"] if utterances else json_path.stem.replace("clip_", "")
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
    except Exception as e:
        return json_path.stem, [], 0, f"클립 로드 실패: {e}"

    detector = create_face_detector(face_model_path)
    rows, skipped = [], 0
    try:
        for u in utterances:
            try:
                label = RAW_LABEL_ALIASES.get(u["emotion_raw"], u["emotion_raw"])
                utt_id = f"{clip_id}_{u['person_id']}_{u['script_start']}_{u['script_end']}"
                wav_path = frames_out / "audio" / f"{utt_id}.wav"
                face_dir = frames_out / "faces" / utt_id

                ok_audio = extract_audio_segment(video_path, u["script_start"], u["script_end"], fps, wav_path)
                n_frames = extract_face_frames(
                    video_path, u["script_start"], u["script_end"],
                    (u["xtl"], u["ytl"], u["xbr"], u["ybr"]), face_dir,
                    detector=detector,
                )
            except Exception:
                skipped += 1
                continue
            finally:
                time.sleep(SLEEP_BETWEEN_UTTERANCES_SEC)

            if not ok_audio or n_frames == 0:
                skipped += 1
                continue

            rows.append([utt_id, label, str(wav_path), u["script"], str(face_dir)])
    finally:
        detector.close()

    return json_path.stem, rows, skipped, None


def _already_done_clip_ids(out_csv: Path) -> set[str]:
    """기존 out_csv에 이미 기록된 클립 id 집합. --resume 시 이 클립들은 다시 처리하지 않는다."""
    if not out_csv.exists():
        return set()
    done = set()
    with open(out_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add(row["utt_id"].split("_")[0])
    return done


def build_manifest(
    raw_dir: Path, out_csv: Path, frames_out: Path, limit_clips: int | None = None,
    resume: bool = False, num_workers: int | None = None,
) -> None:
    n_rows = 0
    skipped = 0

    # 워커들이 동시에 다운로드를 시도하면 파일이 손상될 수 있으므로, 풀 생성 전에
    # 메인 프로세스에서 얼굴 검출 모델을 한 번만 받아둔다.
    face_model_path = ensure_face_detector_model()

    pairs = discover_json_video_pairs(raw_dir)
    print(f"[build_manifest_aihub] {len(pairs)}개 (json, video) 쌍 발견")

    already_done = _already_done_clip_ids(out_csv) if resume else set()
    if already_done:
        before = len(pairs)
        pairs = [(j, v) for j, v in pairs if j.stem.replace("clip_", "") not in already_done]
        print(f"[build_manifest_aihub] --resume: 이미 처리된 {before - len(pairs)}개 클립 건너뜀 (남은 {len(pairs)}개)")

    if limit_clips is not None:
        pairs = pairs[:limit_clips]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not (resume and out_csv.exists())
    # 클립 단위로 곧바로 파일에 쓰고 flush — 장시간 작업 중 중간에 실패해도 그때까지의 결과는 남긴다.
    f = open(out_csv, "a" if resume and out_csv.exists() else "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if write_header:
        writer.writerow(["utt_id", "label", "wav_path", "text", "face_frames_dir"])

    num_workers = num_workers or max(1, (os.cpu_count() or 4) - 2)
    print(f"[build_manifest_aihub] 프로세스 {num_workers}개로 병렬 처리 시작")

    # 클립 하나하나는 서로 독립적이라(다른 클립 결과에 영향 안 줌) 워커 프로세스로 나눠 돌리고,
    # CSV 쓰기만 메인 프로세스가 전담해서 파일 손상 없이 순서 상관없이 flush한다.
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_process_one_clip, j, v, frames_out, face_model_path): j for j, v in pairs}
        for done_count, future in enumerate(as_completed(futures), 1):
            json_path = futures[future]
            try:
                clip_name, rows, clip_skipped, err = future.result()
            except Exception as e:
                print(f"[build_manifest_aihub] ({done_count}/{len(pairs)}) {json_path.stem}: 예외 발생 ({e})")
                continue

            if err:
                print(f"[build_manifest_aihub] ({done_count}/{len(pairs)}) {clip_name}: {err}")
                continue

            for row in rows:
                writer.writerow(row)
                n_rows += 1
            skipped += clip_skipped
            f.flush()
            print(f"[build_manifest_aihub] ({done_count}/{len(pairs)}) {clip_name}: 발화 {len(rows)}건")

    f.close()
    print(f"[build_manifest_aihub] {n_rows}개 발화 -> {out_csv} ({skipped}개는 오디오/프레임 추출 실패로 제외)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames-out", type=Path, required=True)
    parser.add_argument("--limit-clips", type=int, default=None, help="테스트용: 앞의 N개 클립만 처리")
    parser.add_argument("--resume", action="store_true", help="기존 --out 파일에 이미 있는 클립은 건너뛰고 이어서 처리")
    parser.add_argument("--num-workers", type=int, default=None, help="병렬 프로세스 수 (기본: CPU 코어 수 - 2)")
    args = parser.parse_args()
    build_manifest(
        args.raw_dir, args.out, args.frames_out,
        limit_clips=args.limit_clips, resume=args.resume, num_workers=args.num_workers,
    )
