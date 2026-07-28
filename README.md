# trimodal_emotion_model

반려 로봇용 감정인식 모델. 음성 운율(pitch/jitter/shimmer) · STT 텍스트 · 얼굴 표정(영상)을
계층적 교차 어텐션으로 융합해 8클래스 감정(분노·혐오·공포·행복·슬픔·놀람·경멸·중립)을 예측한다.
설계 배경은 `감정인식_모델_설계_v3.md` 참고.

## 구조

```
src/
  config.py                 # configs/config.yaml 로더
  features/
    audio_frontend.py        # 파형 -> 멜스펙트로그램
    prosody.py                # 파형 -> 운율 벡터 p_a (F0/jitter/shimmer/HNR 등)
  models/
    common.py                  # PositionalEncoding, TemporalConvFrontend (공용)
    audio_backbone.py         # 멜스펙트로그램 -> X_a
    visual_backbone.py        # 얼굴 프레임 -> X_v
    text_backbone.py          # klue/bert-base -> X_t
  fusion/
    cross_attention.py        # 양방향 Multi-Head Cross-Attention 블록
    hierarchical_fusion.py    # 설계 §5.1: 2단계(A<->T, 이후 V<->AT) 4블록 융합
    gated_prosody.py           # 설계 §5.2: p_a 게이트 결합
    classifier.py               # 설계 §5.3: 하이브리드 concat + MLP
  model.py                     # TrimodalEmotionModel (전체 조립)
  datasets/
    labels.py                    # 8클래스 라벨 정의 (경멸 포함)
    manifest_dataset.py         # CSV 매니페스트 기반 Dataset + collate_fn
scripts/
  build_manifest_aihub.py     # AI Hub 원본(clip_XXXX/*.json+*.mp4) -> 매니페스트 CSV
  split_manifest.py            # 매니페스트를 clip 단위로 train/val/test 분할
  train.py                     # 학습 루프 (모달리티 드롭아웃, num_workers 지원)
  evaluate.py                  # Accuracy/weighted-F1/Confusion Matrix
tests/
  test_forward_smoke.py       # 합성 텐서로 차원 정합 검증
  test_real_data_smoke.py     # 실제 데이터로 전체 파이프라인 검증
configs/config.yaml           # 하이퍼파라미터 전체 (num_workers 등)
```

## GPU 서버에서 실행하기 (예: A100)

원본 클립(`clip_XXXX/clip_XXXX.json` + `clip_XXXX.mp4` 형태 폴더)이 서버에 이미 있다는 전제.

```bash
# 1. 클론 + 가상환경
git clone <이 저장소 URL> trimodal_emotion_model
cd trimodal_emotion_model
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. GPU 인식 확인
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 3. 원본 클립 -> 매니페스트 CSV (오디오/얼굴 프레임 추출 포함, 클립 수에 비례해 시간 걸림)
python scripts/build_manifest_aihub.py \
  --raw-dir /data/aihub_download/18.멀티모달영상/5201-5600-수정본 \
  --out data/manifests/aihub_full.csv \
  --frames-out data/processed_full

# 4. train/val/test 분할 (clip 단위로 나눠서 정보 누수 방지)
python scripts/split_manifest.py \
  --manifest data/manifests/aihub_full.csv \
  --out-dir data/manifests \
  --val-ratio 0.15 --test-ratio 0.15

# 5. configs/config.yaml에서 num_workers를 서버 CPU 코어 수에 맞게 올리기 (예: 8~16)
#    (기본값 0은 노트북 기준 — 서버에서 0으로 두면 GPU가 놀게 됨)

# 6. 학습 (nohup으로 세션 끊겨도 계속 돌게)
nohup python scripts/train.py --config configs/config.yaml > train.log 2>&1 &

# 7. 진행 확인
tail -f train.log

# 8. 평가
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --manifest data/manifests/test.csv
```

**주의**: `--raw-dir` 경로는 실제 서버에 데이터를 넣어둔 위치로 맞출 것. 스크립트가
`라벨데이터/원천데이터` 분리 배포(샘플용)와 `clip_XXXX/` 폴더에 json+mp4가 같이 있는 정식
배포 두 형태를 자동으로 구분해서 처리한다.

## 로컬(개발용) 실행

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tests/test_forward_smoke.py       # 실데이터 없이 아키텍처 검증
python tests/test_real_data_smoke.py     # data/manifests/test.csv 있으면 실행
```

## 참고

- 라벨 체계, 데이터 신뢰성 근거, 아키텍처 설계 이유는 `../감정인식_모델_설계_v3.md` §0~§10 참고.
- 학습 중 발생한 이슈와 조치 내역은 `error.md` 참고 (예: MPS `enable_nested_tensor` 이슈).
