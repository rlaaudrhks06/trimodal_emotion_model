"""SWA(체크포인트 가중치 평균) 스크립트. 통합기록 §11.1 A-1.

같은 실험(같은 config)의 여러 체크포인트 state_dict를 파라미터별로 단순 평균해서
새 체크포인트 하나를 만든다. 손실 지형의 넓고 평평한 최솟값 쪽으로 이동시켜
단일 체크포인트보다 일반화가 좋아지는 경우가 많다는 게 근거
(Izmailov et al., "Averaging Weights Leads to Wider Optima and Better
Generalization", UAI 2018 — 통합기록 §12 참고문헌 17번).

영상 백본(MobileFaceNet)은 모든 config에서 항상 동결(visual_freeze_layers>0)
상태로 학습되므로 BatchNorm running 통계가 모든 체크포인트에서 동일하다 —
재추정 없이 바로 평균해도 안전하다. 정수형 버퍼(BatchNorm num_batches_tracked
등)는 평균 대신 값이 같은지 확인 후 그대로 사용한다.

실행 예 (v7 기준, val_acc가 안정된 구간의 스냅샷들을 평균):
    python scripts/swa_average.py \\
        --checkpoints checkpoints_periodic_bert_frozen/checkpoint_epoch40.pt \\
                       checkpoints_periodic_bert_frozen/checkpoint_epoch60.pt \\
                       checkpoints_bert_frozen/best_model.pt \\
        --output checkpoints_bert_frozen/swa_avg.pt

이후 평가:
    python scripts/evaluate.py --config configs/config_bert_frozen.yaml \\
        --checkpoint checkpoints_bert_frozen/swa_avg.pt --manifest data/manifests/test.csv
"""
import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True, help="평균낼 state_dict 파일 경로 2개 이상 (같은 config로 학습한 것이어야 함)")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    if len(args.checkpoints) < 2:
        raise ValueError("체크포인트를 2개 이상 지정해야 평균의 의미가 있음")

    state_dicts = [torch.load(p, map_location="cpu") for p in args.checkpoints]
    keys = state_dicts[0].keys()
    for sd, path in zip(state_dicts[1:], args.checkpoints[1:]):
        assert sd.keys() == keys, f"{path}의 파라미터 키가 첫 체크포인트와 다름 — 같은 config로 학습한 것인지 확인"

    averaged = {}
    for key in keys:
        tensors = [sd[key] for sd in state_dicts]
        if tensors[0].is_floating_point():
            averaged[key] = torch.stack(tensors, dim=0).mean(dim=0)
        else:
            first = tensors[0]
            if not all(torch.equal(t, first) for t in tensors[1:]):
                print(f"  [경고] {key}: 정수형 버퍼 값이 체크포인트 간 다름(동결 가정 위반) — 첫 체크포인트 값 사용")
            averaged[key] = first

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(averaged, args.output)
    print(f"[swa_average] {len(args.checkpoints)}개 체크포인트 평균 완료 -> {args.output}")
    for p in args.checkpoints:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
