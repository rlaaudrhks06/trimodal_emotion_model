"""mediapipe 기반 실제 얼굴 검출 + 정렬.

§8.8 진단: build_manifest_aihub.py가 face_frames_dir로 저장해온 크롭은 AI Hub의
"인물(person)" bbox를 그대로 잘라낸 것으로, 얼굴이 아니라 책상에 앉은 사람 전신이
찍힌 이미지였다(face_crop_inspection.png로 확인). 사전학습 얼굴 인식 백본
(MobileFaceNet 등)이 3연속 실패한 근본 원인이 여기 있었다.

이 모듈은 그 person bbox 영역 안에서 mediapipe로 실제 얼굴을 다시 찾고, 두 눈
keypoint를 수평으로 맞추도록 회전시킨 뒤 얼굴 bbox 기준으로 크롭한다.

mediapipe 1.0.0부터 레거시 `mp.solutions` API가 완전히 제거되고 Tasks API
(`mediapipe.tasks`)만 남았다 — 이 API는 .tflite 모델 파일을 직접 준비해야 해서,
ensure_face_detector_model()이 최초 1회 자동으로 내려받아 캐싱한다.
"""
import urllib.request
from pathlib import Path

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

_RIGHT_EYE, _LEFT_EYE = 0, 1  # BlazeFace 6-keypoint 순서(레거시/Tasks API 공통)

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
_DEFAULT_MODEL_PATH = Path.home() / ".cache" / "mediapipe_models" / "blaze_face_short_range.tflite"


def ensure_face_detector_model(dest_path: Path = _DEFAULT_MODEL_PATH) -> Path:
    """얼굴 검출용 .tflite 모델을 최초 1회만 내려받아 캐싱, 이후엔 캐시 재사용.

    ProcessPoolExecutor로 병렬 처리하기 전에 메인 프로세스에서 한 번만 호출해야
    한다(워커들이 동시에 다운로드를 시도하면 파일이 손상될 수 있음).
    """
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return dest_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(".tmp")
    print(f"[face_align] 얼굴 검출 모델 다운로드 중: {_MODEL_URL}")
    urllib.request.urlretrieve(_MODEL_URL, str(tmp_path))
    tmp_path.replace(dest_path)
    return dest_path


def create_face_detector(model_path: Path, min_detection_confidence: float = 0.5):
    """워커(클립)마다 한 번만 만들어서 재사용할 FaceDetector 인스턴스."""
    options = vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        min_detection_confidence=min_detection_confidence,
    )
    return vision.FaceDetector.create_from_options(options)


def detect_and_align_face(
    frame_bgr: np.ndarray,
    person_bbox: tuple[int, int, int, int],
    face_size: int = 112,
    margin: float = 0.4,
    detector=None,
) -> np.ndarray | None:
    """원본 프레임 + person bbox -> 정렬된 얼굴 크롭 [face_size,face_size,3] BGR.

    검출 실패 시 None. detector는 create_face_detector()로 미리 만들어 재사용해야
    한다(프레임마다 새로 만들면 초기화 비용 때문에 느려짐).
    """
    xtl, ytl, xbr, ybr = person_bbox
    h, w = frame_bgr.shape[:2]
    xtl, ytl = max(0, xtl), max(0, ytl)
    xbr, ybr = min(w, xbr), min(h, ybr)
    region = frame_bgr[ytl:ybr, xtl:xbr]
    if region.size == 0:
        return None

    rh, rw = region.shape[:2]
    rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = detector.detect(mp_image)
    if not result.detections:
        return None

    best = max(result.detections, key=lambda d: d.categories[0].score)
    kp = best.keypoints  # NormalizedKeypoint, x/y는 region 기준 0~1 상대좌표(공통)
    right_eye = np.array([kp[_RIGHT_EYE].x * rw, kp[_RIGHT_EYE].y * rh])
    left_eye = np.array([kp[_LEFT_EYE].x * rw, kp[_LEFT_EYE].y * rh])

    # 두 눈을 잇는 선이 수평이 되도록 반대 각도로 회전
    dy, dx = left_eye[1] - right_eye[1], left_eye[0] - right_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    eye_center = tuple((right_eye + left_eye) / 2)
    rot_mat = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    rotated = cv2.warpAffine(region, rot_mat, (rw, rh))

    # bounding_box는 Tasks API에서 region 기준 절대 픽셀 좌표(origin_x/y, width/height)
    bb = best.bounding_box
    bx, by, bw, bh = bb.origin_x, bb.origin_y, bb.width, bb.height
    # 얼굴 bbox의 네 꼭짓점도 같은 회전 변환을 적용해야 회전된 좌표계에서 정확히 크롭된다
    corners = np.array([[bx, by], [bx + bw, by], [bx, by + bh], [bx + bw, by + bh]], dtype=np.float64)
    corners_h = np.hstack([corners, np.ones((4, 1))])
    rotated_corners = corners_h @ rot_mat.T
    fx1, fy1 = rotated_corners[:, 0].min(), rotated_corners[:, 1].min()
    fx2, fy2 = rotated_corners[:, 0].max(), rotated_corners[:, 1].max()

    mw, mh = (fx2 - fx1) * margin, (fy2 - fy1) * margin
    fx1, fy1 = max(0, int(fx1 - mw)), max(0, int(fy1 - mh))
    fx2, fy2 = min(rw, int(fx2 + mw)), min(rh, int(fy2 + mh))
    if fx2 <= fx1 or fy2 <= fy1:
        return None

    face = rotated[fy1:fy2, fx1:fx2]
    if face.size == 0:
        return None
    return cv2.resize(face, (face_size, face_size))


def extract_aligned_frames(
    cap: cv2.VideoCapture,
    start_frame: int,
    end_frame: int,
    bbox: tuple[int, int, int, int],
    out_dir: Path,
    detector,
    face_size: int = 112,
    max_frames: int = 24,
) -> int:
    """이미 열린 VideoCapture에서 [start_frame,end_frame] 구간을 max_frames개로
    균등 샘플링해, bbox 영역 안에서 얼굴을 검출+정렬 후 out_dir/frame_NNN.jpg로 저장.

    build_manifest_aihub.py(신규 배치 최초 추출)와 refix_face_crops.py(기존 배치
    사후 보정)가 거의 동일한 프레임 샘플링+검출+저장 루프를 각자 갖고 있었던 걸
    이 함수 하나로 합쳤다. cap의 열기/닫기는 호출자 책임(스크립트마다 열고 닫는
    시점이 달라서 — build_manifest는 발화마다, refix는 클립마다 재사용).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    step = max(1, (end_frame - start_frame + 1) // max_frames)
    saved = 0
    for frame_no in range(start_frame, end_frame + 1, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok:
            continue
        face = detect_and_align_face(frame, bbox, face_size=face_size, detector=detector)
        if face is None:
            continue
        cv2.imwrite(str(out_dir / f"frame_{saved:03d}.jpg"), face)
        saved += 1
    return saved
