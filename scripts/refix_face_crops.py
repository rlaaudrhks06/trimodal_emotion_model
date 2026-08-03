"""§8.8 대응: person bbox로 잘못 저장된 face_frames_dir를 실제 얼굴 검출+정렬로 재생성.

build_manifest_aihub.py가 만든 face_frames_dir 프레임은 AI Hub의 "인물(person)"
bbox를 그대로 잘라낸 것으로, 실제로는 얼굴이 아니라 사람 전신이 찍힌 이미지였다
(face_crop_inspection.png로 확인, MobileFaceNet 등 얼굴 인식 사전학습 백본이 3연속
실패한 근본 원인). 매니페스트 CSV/오디오/텍스트/라벨은 이미 정상이므로 그대로 두고,
같은 utt_id의 face_frames_dir 안 이미지 파일만 mediapipe 검출+정렬 결과로 덮어쓴다.

원본 JSON/비디오가 필요하다는 점은 build_manifest_aihub.py와 동일하다(person bbox와
발화 구간(script_start/end)을 다시 읽어야 하므로).

사용법 (build_manifest_aihub.py를 만들 때 쓴 것과 같은 --raw-dir/--frames-out):
    python scripts/refix_face_crops.py --raw-dir /data/aihub_download/... \
        --frames-out data/processed_full --num-workers 12
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from scripts.build_manifest_aihub import MAX_FACE_FRAMES, discover_json_video_pairs, load_json_utterances
from src.features.face_align import create_face_detector, ensure_face_detector_model, extract_aligned_frames


def _refix_one_clip(json_path: Path, video_path: Path, frames_out: Path, face_size: int, model_path: Path) -> tuple[str, int, int, str | None]:
    try:
        utterances = load_json_utterances(json_path)
        clip_id = utterances[0]["clip_id"] if utterances else json_path.stem.replace("clip_", "")
    except Exception as e:
        return json_path.stem, 0, 0, f"클립 로드 실패: {e}"

    detector = create_face_detector(model_path)
    cap = cv2.VideoCapture(str(video_path))
    n_ok = n_fail = 0
    try:
        for u in utterances:
            utt_id = f"{clip_id}_{u['person_id']}_{u['script_start']}_{u['script_end']}"
            face_dir = frames_out / "faces" / utt_id
            if not face_dir.exists():
                # build_manifest_aihub.py에서 이미 스킵된 발화(매니페스트에도 없음) -> 건너뜀
                continue

            start, end = u["script_start"], u["script_end"]
            bbox = (int(u["xtl"]), int(u["ytl"]), int(u["xbr"]), int(u["ybr"]))

            saved = extract_aligned_frames(
                cap, start, end, bbox, face_dir, detector,
                face_size=face_size, max_frames=MAX_FACE_FRAMES,
            )

            if saved > 0:
                # 새로 저장한 것보다 뒤에 남아있는 옛(전신 크롭) 파일은 삭제
                for old in sorted(face_dir.glob("frame_*.jpg"))[saved:]:
                    old.unlink()
                n_ok += 1
            else:
                # 얼굴을 한 프레임도 못 찾음 -> 기존(잘못된) 크롭을 그대로 남겨둠(파이프라인 안 깨지게).
                # 이런 발화는 로그에 남으니 나중에 매니페스트에서 제외할지 검토 필요.
                n_fail += 1
    finally:
        cap.release()
        detector.close()

    return json_path.stem, n_ok, n_fail, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--frames-out", type=Path, required=True)
    ap.add_argument("--face-size", type=int, default=112)
    ap.add_argument("--limit-clips", type=int, default=None, help="테스트용: 앞의 N개 클립만 처리")
    ap.add_argument("--num-workers", type=int, default=None)
    args = ap.parse_args()

    # 워커들이 동시에 다운로드를 시도하면 파일이 손상될 수 있으므로, 풀 생성 전에
    # 메인 프로세스에서 한 번만 받아둔다.
    model_path = ensure_face_detector_model()

    pairs = discover_json_video_pairs(args.raw_dir)
    print(f"[refix_face_crops] {len(pairs)}개 (json, video) 쌍 발견")
    if args.limit_clips is not None:
        pairs = pairs[: args.limit_clips]

    num_workers = args.num_workers or max(1, (os.cpu_count() or 4) - 2)
    print(f"[refix_face_crops] 프로세스 {num_workers}개로 병렬 처리 시작")

    total_ok = total_fail = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_refix_one_clip, j, v, args.frames_out, args.face_size, model_path): j
            for j, v in pairs
        }
        for done, future in enumerate(as_completed(futures), 1):
            json_path = futures[future]
            try:
                clip_name, n_ok, n_fail, err = future.result()
            except Exception as e:
                print(f"[refix_face_crops] ({done}/{len(pairs)}) {json_path.stem}: 예외 발생 ({e})")
                continue
            if err:
                print(f"[refix_face_crops] ({done}/{len(pairs)}) {clip_name}: {err}")
                continue
            total_ok += n_ok
            total_fail += n_fail
            print(f"[refix_face_crops] ({done}/{len(pairs)}) {clip_name}: 재생성 성공 {n_ok} / 얼굴 미검출 {n_fail}")

    print(f"[refix_face_crops] 완료. 발화 {total_ok}개 재생성 성공, {total_fail}개 얼굴 미검출(기존 크롭 유지)")


if __name__ == "__main__":
    main()
