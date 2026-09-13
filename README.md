# trimodal_emotion_model

반려 로봇용 한국어 감정인식 모델. **음성 파형 · 얼굴 표정 · 발화 내용**을 2단계 계층적
교차 어텐션으로 융합해 7클래스 감정을 예측하고, **확신이 낮으면 판단을 보류한다.**

2026 인공지능 루키 결선 과제 「May the Force be with you」(팀 모두를 위한 하나)의
감정인식 파트다. 로봇은 서버 없이 Jetson Orin Nano 한 대에서 동작한다.

| 지표 | 값 |
|---|---|
| **7클래스 정확도** | **46.19%** (다수 클래스 기준선 27.00% 대비 +19.19%p) — 시드 42. 3시드 평균 45.18 ± 0.96%. 소음 증강 v11a는 44.59 ± 0.68 / SNR 10dB 44.34 ± 0.72 |
| 3클래스(긍정/부정/중립) — 제품이 쓰는 단위 | **67.34%** (재학습 없이 출력만 묶은 값) |
| 판단 보류 적용 (확신도 0.5) | 응답률 40.4%, 응답한 것의 정확도 **60.66%** |
| 평가 조건 | **화자 독립** — test 화자 40명이 train에 없다 |
| 모델 규모 | 학습 876만 / 전체 4억 3,688만 (98.0%가 동결된 사전학습 가중치) |

채택 모델은 **v11**이다. 전체 사양·성능·한계·재현 방법은 [`docs/v11_모델카드.md`](docs/v11_모델카드.md)에 있다.

---

## ⚠️ 데이터 라이선스 — 이 저장소에 데이터가 없는 이유

학습 데이터는 AI Hub 「멀티모달 영상」(dataSetSn=58)이며 이용정책상 **제3자 제공·재배포
금지, 비상업적 연구·개발 한정**이다. 따라서 다음은 **이 저장소에 없고, 올려서도 안 된다.**

```
data/processed_full/     얼굴 크롭·오디오 (27GB)
data/manifests_*/*.csv   매니페스트
data/feature_cache/      특징 캐시
checkpoints/*.pt         체크포인트 (개당 약 1.6GB)
models/                  사전학습 백본 사본
```

`results/predictions/*.csv`는 `utt_id`·정답·예측·확률만 담고 원문 텍스트가 없어 포함했다.

---

## 지금 상태

**2026-09-10 기준으로 결선 2차 기간이 시작됐다.** 현재 상황·서버 사양·이번 기간 계획은
[`docs/감정인식_프로젝트_통합기록.md`](docs/감정인식_프로젝트_통합기록.md) **13장**에 있다.

| 문서 | 용도 |
|---|---|
| `docs/감정인식_프로젝트_통합기록.md` | **단일 근거 기록.** 설계 전문 + 실험 서사 + 판단 근거. 13장이 현재 국면 |
| `docs/v11_모델카드.md` | 채택 모델의 사양·성능·한계·재현 — 사실 위주 |
| `docs/감정인식_프로젝트_쉬운설명.md` | 비전공자용 설명 |
| `docs/감정인식_프로젝트_공부가이드.md` | 이 프로젝트를 이해하기 위한 학습 순서 |
| `docs/서버_회수_기록.md` | 서버 반납 시 무엇을 어떻게 회수·검증했는가 |

---

## 구조

```
src/
  config.py                    configs/*.yaml 로더
  model.py                     TrimodalEmotionModel (전체 조립)
  model_single_modality.py     단일 모달리티 베이스라인
  eval_report.py               지표 계산·저장 (eval JSON, 발화별 예측 CSV)
  features/
    audio_frontend.py          파형 -> 멜스펙트로그램
    prosody.py                 파형 -> 운율 10차원 (F0/jitter/shimmer/HNR 등)
    face_align.py              person bbox 안에서 얼굴 재검출·정렬 (MediaPipe)
  models/
    common.py                  PositionalEncoding, TemporalConvFrontend (공용)
    audio_backbone_w2v.py      wav2vec2-XLSR-53 -> X_a   ← v11이 쓰는 것
    audio_backbone.py          멜스펙트로그램 -> X_a      (v1~v10 경로, 유지)
    visual_backbone.py         MobileFaceNet -> X_v
    text_backbone.py           klue/bert-base -> X_t
  fusion/
    cross_attention.py         양방향 Multi-Head Cross-Attention 블록
    hierarchical_fusion.py     2단계 계층 융합(4블록) + self-attention 베이스라인
    gated_prosody.py           운율 게이트 / concat 대조군
    classifier.py              하이브리드 concat + MLP
  datasets/
    labels.py                  **7클래스** 라벨 정의 (contempt -> disgust 병합)
    manifest_dataset.py        매니페스트 기반 Dataset + collate_fn + 특징 캐시

scripts/     32개. 주요한 것만:
  train.py                     학습 루프
  evaluate.py                  평가 (--drop-modality / --noise-snr / --save-predictions)
  split_manifest.py            매니페스트 분할  ← --unit 주의, 아래 참고
  compute_prosody_stats.py     운율 정규화 통계 (train에서만 fit)
  build_manifest_aihub.py      AI Hub 원본 -> 매니페스트 (원본이 있을 때만)
  export_onnx.py               ONNX 변환 + PyTorch 동등성 검증
  inspect_prosody_gate.py      운율 게이트가 실제로 작동하는지 측정
  analyze_predictions.py       저장된 예측으로 사후 분석 (모델 재실행 없음)
  probe_embeddings.py          표현에 감정/화자 정보가 얼마나 있는지 프로빙

robot/
  demo.py                      실시간 데모 (카메라·마이크 -> 감정 판정)
  brain/engine.py              추론 엔진 — 온도 보정 + 판단 보류까지
  brain/preprocess.py          실시간 입력을 학습 때와 **똑같은 형태**로
  scripts/benchmark_inference.py   구간별 추론 속도 측정

configs/     17개. v11은 config_si_w2v.yaml
results/     학습 곡선 CSV · 평가 JSON · 발화별 예측 CSV (git 포함)
tests/       스모크 테스트
docs/        위 표 참고
```

---

## 서버에서 학습 재개하기

원본 클립에서 시작하지 않는다. **`processed_full`(전처리 완료본)을 올려서 쓴다.**
얼굴 재검출 파이프라인은 이미 한 번 돌렸고, 다시 돌리면 AI Hub 원본 15개 zip(약 120GB)을
다시 받아야 한다.

### 1. 코드와 환경

```bash
git clone https://github.com/rlaaudrhks06/trimodal_emotion_model.git
cd trimodal_emotion_model
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# GPU 인식 확인 — CPU 버전이 깔리는 사고가 흔하다
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**디스크 위치를 먼저 정한다.** `feature_cache`만 84GB이므로 여유가 큰 파티션에 둔다.
홈 파티션이 100GB 이하면 학습 도중 가득 차서 죽는다.

### 2. 데이터 업로드 — rsync 말고 병렬 tar

얼굴 프레임이 **216만 개의 작은 파일**이라 rsync는 파일마다 프로토콜 왕복을 한다.
tar 스트리밍을 병렬로, 압축 없이(JPG·WAV는 이미 압축돼 있다) 보내면 **약 47배 빠르다**
— 실측값과 방법은 `docs/서버_회수_기록.md`에 있다.

올릴 것: `data/processed_full/` · `data/manifests_si/` · `data/prosody_stats_train_si.json`
· `models/wav2vec2-large-xlsr-53/` · `checkpoints/v11_best.pt`

### 3. 무결성 검증 — 개수만 맞추면 안 된다

파일 개수·총 바이트·매니페스트 참조 누락 0을 확인하고, **마지막에 실제 추론을 돌려
저장된 예측(`results/predictions/v11.csv`)과 일치하는지 본다.** 개수와 용량이 맞아도
예측이 재현되지 않으면 데이터·체크포인트·코드 중 하나가 어긋난 것이다.

### 4. 학습

```bash
nohup python scripts/train.py --config configs/config_si_w2v.yaml > train.log 2>&1 &
tail -f train.log
```

`num_workers`는 config에서 서버 CPU 코어 수에 맞춘다. 첫 에폭은 특징 캐시를 만드느라
느리다(캐시 없이 0.80초/건). 학습 전에 DataLoader를 한 바퀴 돌려 캐시를 채워두면
GPU가 놀지 않는다.

### 5. 평가

```bash
python scripts/evaluate.py --config configs/config_si_w2v.yaml \
    --checkpoint checkpoints_si_w2v/best_model.pt --save-as v11 --save-predictions
```

`--manifest`를 생략하면 config의 `test_manifest`를 쓴다 — 다른 분할로 학습한 모델을
옛 test셋으로 평가하는 사고를 막기 위한 기본값이다.

**`--save-predictions`는 선택이 아니다.** 발화별 예측을 저장하지 않으면 나중에 짝지어
검정(McNemar)을 할 수 없다. 실제로 v11과 v12b는 이걸 안 해서 지금도 검정을 못 한다.

---

## ⚠️ 매니페스트를 다시 나눈다면 — `--unit speaker`

`split_manifest.py`의 `--unit` **기본값은 `clip`이고, 그건 화자 누수를 만든다.**

AI Hub 데이터는 40클립이 원본 영상 한 편을 이루고 `person_id`가 전역 고유 번호라,
클립 단위로 섞으면 **같은 사람이 train과 test에 모두 들어간다.** 실제로 이전 분할에서는
test 화자 278명이 전원 train에도 있었고, 그 조건에서 나온 v1~v9의 수치는 전부 부풀려져
있었다. 화자 독립으로 다시 나누자 실질 **6.51%p**가 빠졌다.

```bash
python scripts/split_manifest.py --manifest data/manifests/all.csv \
    --out-dir data/manifests_si --unit speaker --seed 2026 --verify
```

`--verify`가 train/val/test 간 화자 중복 0을 확인한다. 경위는 통합기록 8.14·8.17절.

---

## 로컬(개발용) 실행

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tests/test_forward_smoke.py     # 실데이터 없이 아키텍처·플래그 검증
python tests/test_real_data_smoke.py   # 매니페스트·데이터가 있으면 실행
```

실시간 데모(카메라·마이크 필요):

```bash
python robot/demo.py --checkpoint checkpoints/v11_best.pt
```

---

## 진행 중인 실험 — 융합 설계 애블레이션

전부 v11(`config_si_w2v.yaml`)에서 **한 항목만** 바꾼 config다. 근거와 판정 기준은
통합기록 11.3.2·13.4.3절에 있다.

| config | 바뀌는 것 | 묻는 것 |
|---|---|---|
| `config_fusion_av.yaml` | `fusion_order: audio_visual` | 텍스트의 기여가 **정보**인가 **구조**인가 |
| `config_fusion_self.yaml` | `fusion_type: self` | 교차 어텐션이 하는 일이 정말 *교차*인가 |
| `config_prosody_concat.yaml` | `prosody_fusion: concat` | **게이트 기구**가 값을 하는가 |
| `config_prosody_none.yaml` | `prosody_fusion: none` | **운율 정보**가 값을 하는가 |

---

## 젯슨 배포 — ONNX

`ONNX -> TensorRT -> FP16/INT8` 순이고 TensorRT는 ONNX만 입력으로 받는다.
첫 칸은 GPU 없이 된다.

```bash
python scripts/export_onnx.py --config configs/config_si_w2v.yaml \
    --checkpoint checkpoints/v11_best.pt --out checkpoints/onnx/v11.onnx
```

변환은 tracing이라 발화 길이·프레임 수·토큰 수가 상수로 굳어도 **에러 없이 그럴듯한
답이 나온다.** 그래서 이 스크립트는 변환 후 PyTorch와 대조하고, 판정 기준을 로짓의
소수점이 아니라 **로봇이 실제로 쓰는 결정**(7클래스 argmax·3클래스·응답/보류)으로 둔다.
**검증을 통과하지 못하면 파일을 남기지 않는다.**

---

## 비교 규칙

수치를 비교할 때 이 저장소가 지키는 것들이다. 전부 한 번씩 데이고 세운 규칙이다.

- **1.3%p 미만 차이로는 결론을 내지 않는다.** test 11,464개에서 두 모델 *차이*의
  표준오차가 약 0.66%p다. 단일 모델의 표준오차(0.47%p)로 나누면 "유의하다"와
  "구분 안 된다"가 뒤집힌다 — 실제로 한 번 뒤집혔다(8.29.1절).
- **평가할 때 발화별 예측을 저장한다.** 안 하면 짝지어 검정을 영영 못 한다.
- **수치를 손으로 옮기지 않는다.** 표는 스크립트가 원본에서 읽어 만든다.
- **모달리티를 끄는 코드는 학습 쪽 구현을 그대로 따른다.** 다르게 끄면 재는 것이
  "기여도"가 아니라 "낯선 입력에 대한 반응"이 된다.
- **모델에 영향이 가는 수정은 4단계 검토를 거친다** — 1차 발견만, 2차 수정,
  3차 수정 검토, 4차 전체 재검토. 실제로 3차가 2차의 수정을 잡은 적이 있다.

---

## 생성물 (git에 없다 — 스크립트로 다시 만든다)

| 산출물 | 만드는 법 |
|---|---|
| `docs/*.pdf` | `python scripts/build_docs_pdf.py` |
| `results/dashboard.html` | `python scripts/build_dashboard.py` |
| `docs/assets/architecture_*.png` | `python scripts/build_architecture_diagrams.py` |
