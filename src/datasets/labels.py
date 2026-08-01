"""8클래스 라벨 체계 (Ekman 6 + 경멸(contempt) + 중립).

애초 설계 v3 초안은 7클래스(Ekman 6 + neutral)였으나, 실제 AI Hub "멀티모달 영상"
데이터셋(우리의 유일한 진짜 트리모달 동기화 소스 — 영상까지 포함된 건 이것뿐)의
공식 데이터 설명서를 확인한 결과 폴 에크먼의 8종 분류(경멸 포함)를 쓰고 있어
8클래스로 확장했다. 경멸(contempt)과 혐오(disgust)는 심리학적으로 구분되는
감정이라 병합하지 않는다.

KEMDy19/20은 7클래스(경멸 없음)라서 해당 데이터로는 "contempt" 라벨이 그냥
등장하지 않을 뿐, 이 8클래스 체계 위에 문제없이 얹을 수 있다.

주의: AI Hub 원본 JSON은 "혐오"를 영어로 "dislike"라고 표기한다. 학술 관례상
"disgust"를 표준 표기로 쓰므로, 매니페스트 생성 시 dislike -> disgust로
맞춰줘야 한다(scripts/build_manifest_aihub.py 참고).
"""

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "contempt", "neutral"]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}

# AI Hub 원본 라벨 표기 -> 우리 표준 표기 (매니페스트 생성 시 사용)
RAW_LABEL_ALIASES = {
    "dislike": "disgust",  # AI Hub 원본 표기
}

# 참고용: 한국어 표기 매핑 (기존 robot_project의 텍스트 감정 라벨과 대조용)
KOREAN_LABELS = {
    "angry": "분노", "disgust": "혐오", "fear": "공포", "happy": "행복",
    "sad": "슬픔", "surprise": "놀람", "contempt": "경멸", "neutral": "중립",
}
