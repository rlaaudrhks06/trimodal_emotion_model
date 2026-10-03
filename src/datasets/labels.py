"""7클래스 라벨 체계 (Ekman 6 + 중립) — §11(통합기록) 경멸→혐오 병합 결정.

애초 설계 v3 초안은 7클래스였다가, AI Hub "멀티모달 영상" 데이터셋이 폴 에크먼의
8종 분류(경멸 포함)를 쓰고 있어 8클래스로 확장했었다(그 시절 근거는 §3.3.3 참고).
그런데 §8.9까지 여러 차례 검증한 결과, 경멸(contempt)은:
  - 데이터가 2천 개 넘게 있어도(§7.3) recall이 4~8%에 그치고
  - person bbox 버그(§8.8) 수정 후에도(v5) 여전히 30%가 혐오로 오분류되고
  - 즉 데이터량 문제도, 영상 브랜치 버그 문제도 아니라 특징 표현 자체의 한계로 보여
심리학적으로는 구분되는 감정이지만, 모델 관점에서 실질적으로 구분이 안 되는 클래스를
붙잡고 있는 것보다 원래 초안대로 혐오에 병합하는 게 낫다고 판단해 7클래스로 되돌린다.

KEMDy19/20은 원래 7클래스(경멸 없음)라서 이 병합 이후에는 AI Hub와 KEMDy19/20이
동일한 라벨 체계를 쓰게 된다(§3.11 한계1의 2차 검증 데이터 호환성도 개선됨).

주의: AI Hub 원본 JSON은 "혐오"를 영어로 "dislike"라고 표기한다. 학술 관례상
"disgust"를 표준 표기로 쓰므로, 매니페스트 생성 시 dislike -> disgust로
맞춰줘야 한다(scripts/build_manifest_aihub.py 참고).
"""

# 라벨 집합은 환경변수 EMOTION_LABEL_SET으로 고른다. 기본값은 지금까지와 똑같은 7클래스라
# 기존 config·결과·스크립트는 한 줄도 영향받지 않는다.
#
# "4"를 주면 분노·행복·보통·슬픔만 쓴다 — emotion4 프로젝트(변유철 2024 논문과 조건을 맞춘
# 4클래스 과제)의 기준선을 **환산이 아니라 직접 학습으로** 재기 위한 것이다. 7클래스 모델의
# 확률을 4개만 더해 환산한 값(test 63.79 ± 0.71)은 4클래스로 학습한 모델과 다를 수 있다.
# 놀람·공포·혐오·경멸은 매핑하지 않고 버린다(매니페스트 단계에서 이미 빠져 있다).
import os as _os

_LABEL_SETS = {
    "7": ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"],
    "4": ["angry", "happy", "neutral", "sad"],
}
_SET = _os.environ.get("EMOTION_LABEL_SET", "7")
if _SET not in _LABEL_SETS:
    raise ValueError(f"EMOTION_LABEL_SET은 {sorted(_LABEL_SETS)} 중 하나여야 한다, got {_SET!r}")
EMOTION_LABELS = _LABEL_SETS[_SET]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}

# AI Hub 원본 라벨 표기 -> 우리 표준 표기 (매니페스트 생성 시 사용)
RAW_LABEL_ALIASES = {
    "dislike": "disgust",  # AI Hub 원본 표기
}

# 8클래스 -> 7클래스 병합 규칙. RAW_LABEL_ALIASES와 별도로 두는 이유: 저건
# "같은 개념의 표기 차이"(dislike==disgust)이고, 이건 "원래 다른 개념이지만
# 모델링 관점에서 병합하기로 한 결정"이라 성격이 다르다 — 나중에 이 결정을
# 재검토할 때(예: 데이터가 훨씬 늘어난 뒤 다시 8클래스로) 여기만 비우면 된다.
LABEL_MERGE = {
    "contempt": "disgust",
}


def normalize_label(raw: str) -> str:
    """원본 라벨 문자열 -> 최종 학습용 표준 라벨(EMOTION_LABELS 중 하나).

    1) RAW_LABEL_ALIASES: 데이터셋 원본 표기 차이 흡수(dislike -> disgust)
    2) LABEL_MERGE: 8->7클래스 병합(contempt -> disgust)
    두 매핑 다 없으면 원본 그대로 반환(이미 표준 표기인 경우).
    """
    label = RAW_LABEL_ALIASES.get(raw, raw)
    label = LABEL_MERGE.get(label, label)
    return label


# 참고용: 한국어 표기 매핑 (기존 robot_project의 텍스트 감정 라벨과 대조용)
KOREAN_LABELS = {
    "angry": "분노", "disgust": "혐오", "fear": "공포", "happy": "행복",
    "sad": "슬픔", "surprise": "놀람", "neutral": "중립",
}
