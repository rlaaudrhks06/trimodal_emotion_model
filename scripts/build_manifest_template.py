"""KEMDy19/20 (또는 AI Hub 멀티모달 영상) 원본 데이터 -> 매니페스트 CSV 변환 스크립트 템플릿.

** 이 스크립트는 아직 완성본이 아닙니다 **
ETRI Nanum(KEMDy19/20)과 AI Hub는 계정 가입·이용 동의 후 브라우저로 수동
다운로드해야 하는 배포 방식이라, 실제 파일이 로컬에 없는 상태에서는 정확한
폴더/파일명 규칙을 단정할 수 없습니다. 데이터를 내려받은 뒤 아래 TODO 부분을
실제 폴더 구조에 맞게 채워 넣어 사용하세요.

목표 출력 스키마 (src/datasets/manifest_dataset.py의 REQUIRED_COLUMNS와 동일):
    utt_id, label, wav_path, text, face_frames_dir

사용 예:
    python scripts/build_manifest_template.py \
        --raw-dir ../data/raw/KEMDy19 \
        --out ../data/manifests/train.csv
"""
import argparse
import csv
from pathlib import Path

from src.datasets.labels import EMOTION_LABELS


def build_manifest(raw_dir: Path, out_csv: Path) -> None:
    rows = []

    # TODO(데이터 다운로드 후 작성):
    #   1) raw_dir 아래 세션/발화 폴더를 순회하며 각 발화의
    #      (wav 경로, 라벨, 전사문, 얼굴 프레임 디렉터리)를 찾는다.
    #   2) 라벨 문자열을 EMOTION_LABELS 표기(angry/disgust/fear/happy/sad/surprise/neutral)로 매핑한다.
    #      - KEMDy19/20 원문 라벨이 한국어이거나 표기가 다를 수 있으므로 매핑 테이블이 필요할 수 있음.
    #   3) 얼굴 프레임이 없는 경우(§2.2 옵션 B로 폴백 시)에는 별도 FER 데이터셋 경로를 사용.
    #
    # 예시 골격:
    # for session_dir in sorted(raw_dir.glob("Session*")):
    #     for wav_path in sorted(session_dir.glob("**/*.wav")):
    #         utt_id = wav_path.stem
    #         label = ...  # 세션 내 라벨 파일(csv/txt)에서 조회
    #         text = ...   # 세션 내 전사문 파일에서 조회
    #         face_frames_dir = ...  # 영상 데이터 사용 시 사전 크롭 프레임 폴더
    #         rows.append([utt_id, label, str(wav_path), text, str(face_frames_dir)])

    if not rows:
        raise NotImplementedError(
            "위 TODO를 실제 데이터 폴더 구조에 맞게 구현한 뒤 다시 실행하세요. "
            f"(참고: 유효 라벨 집합 = {EMOTION_LABELS})"
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["utt_id", "label", "wav_path", "text", "face_frames_dir"])
        writer.writerows(rows)
    print(f"[build_manifest] {len(rows)}개 발화 -> {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build_manifest(args.raw_dir, args.out)
