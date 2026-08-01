"""mediapipe 기반 실제 얼굴 검출 + 정렬.

§8.8 진단: build_manifest_aihub.py가 face_frames_dir로 저장해온 크롭은 AI Hub의
"인물(person)" bbox를 그대로 잘라낸 것으로, 얼굴이 아니라 책상에 앉은 사람 전신이
찍힌 이미지였다(face_crop_inspection.png로 확인). 사전학습 얼굴 인식 백본
(MobileFaceNet 등)이 3연속 실패한 근본 원인이 여기 있었다.

이 모듈은 그 person bbox 영역 안에서 mediapipe로 실제 얼굴을 다시 찾고, 두 눈
keypoint를 수평으로 맞추도록 회전시킨 뒤 얼굴 bbox 기준으로 크롭한다.
"""
import cv2
import numpy as np

_RIGHT_EYE, _LEFT_EYE = 0, 1  # mediapipe FaceDetection relative_keypoints 순서


def detect_and_align_face(
    frame_bgr: np.ndarray,
    person_bbox: tuple[int, int, int, int],
    face_size: int = 112,
    margin: float = 0.4,
    detector=None,
) -> np.ndarray | None:
    """원본 프레임 + person bbox -> 정렬된 얼굴 크롭 [face_size,face_size,3] BGR.

    검출 실패 시 None. detector는 호출자가 미리 만들어 재사용해야 한다
    (프레임마다 새로 만들면 초기화 비용 때문에 느려짐).
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
    results = detector.process(rgb)
    if not results.detections:
        return None

    best = max(results.detections, key=lambda d: d.score[0])
    kp = best.location_data.relative_keypoints
    right_eye = np.array([kp[_RIGHT_EYE].x * rw, kp[_RIGHT_EYE].y * rh])
    left_eye = np.array([kp[_LEFT_EYE].x * rw, kp[_LEFT_EYE].y * rh])

    # 두 눈을 잇는 선이 수평이 되도록 반대 각도로 회전
    dy, dx = left_eye[1] - right_eye[1], left_eye[0] - right_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    eye_center = tuple((right_eye + left_eye) / 2)
    rot_mat = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    rotated = cv2.warpAffine(region, rot_mat, (rw, rh))

    bbox = best.location_data.relative_bounding_box
    bx, by = bbox.xmin * rw, bbox.ymin * rh
    bw, bh = bbox.width * rw, bbox.height * rh
    # 얼굴 bbox의 네 꼭짓점도 같은 회전 변환을 적용해야 회전된 좌표계에서 정확히 크롭된다
    corners = np.array([[bx, by], [bx + bw, by], [bx, by + bh], [bx + bw, by + bh]])
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
