# 학습 중 발생한 에러 기록

## 1. MPS 백엔드 미지원 연산 크래시 (2026-07-25)

**증상**: `scripts/train.py` 첫 실행이 첫 배치 forward pass에서 아래 에러로 죽음.

```
NotImplementedError: The operator 'aten::_nested_tensor_from_mask_left_aligned'
is not currently implemented for the MPS device.
```

**원인**: `src/models/common.py`의 `TemporalConvFrontend`가 오디오/영상 백본에서
`nn.TransformerEncoder`에 `src_key_padding_mask`를 넘길 때, PyTorch가 자동으로
"nested tensor" 최적화 경로를 타는데 이 경로 안의 한 연산이 MPS(애플 GPU)에서
아직 구현되어 있지 않음. 학습이 `device="mps"`로 잡혀서(Apple Silicon 자동 감지)
이 문제가 발생함.

이 문제 때문에 첫 에폭이 24분 넘게 걸리다가(패딩이 있는 배치를 만나기 전까지는
정상 진행) 결국 크래시로 종료됨 — 느린 게 아니라 언젠가 죽을 상황이었음.

**조치**: `nn.TransformerEncoder` 생성 시 `enable_nested_tensor=False`를 명시해
문제의 최적화 경로 자체를 비활성화함 (`src/models/common.py`). 정확도에는 영향
없고, 이 최적화가 원래 주는 속도 이득만 포기하는 것 — 데이터 규모가 작아 실질적
영향은 미미함.

**상태**: 수정 완료 및 검증됨 — 실제 패딩이 있는 배치로 MPS에서 forward+backward 재현 테스트 통과. 사용자가 사전 승인한 범위(학습 정상화)라 별도 확인 없이 수정 후 재실행함.
