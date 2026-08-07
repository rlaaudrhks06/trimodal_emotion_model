"""트리모달 감정 모델의 추론 속도를 브랜치별로 잰다.

**왜 이것부터 하는가**: v11은 4.37억 파라미터다. 로봇에서 한 발화를 판단하는 데 몇 초
걸리는지 한 번도 잰 적이 없는데, 이 값이 시스템 설계 전체를 좌우한다.

    0.2초 이하  -> 대화 흐름에 무리 없음. 그대로 간다
    0.5~1초     -> 발화 종료 후 약간의 지연. 대기 표현으로 덮을 수 있다
    2초 이상    -> 실시간 대화에 못 쓴다. 경량화나 서버 분리가 선행돼야 한다

**브랜치별로 나눠 재는 게 핵심이다.** 전체 시간만 알면 무엇을 줄여야 할지 모른다.
wav2vec2가 90%를 먹는지 MobileFaceNet이 프레임 수만큼 곱해지는지에 따라 대응이 다르다.

가중치는 필요 없다 — 연산량은 구조와 입력 크기로 정해지므로 무작위 초기화로 재도 같다.
실제 체크포인트는 1.6GB라 서버에만 있다.

실행 예:
    python scripts/benchmark_inference.py \
        --config ../trimodal_emotion_model/configs/config_si_w2v.yaml
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

# 트리모달 모델 코드를 빌려 쓴다. 로봇 프로젝트가 그 저장소에 의존하는 유일한 지점이라
# 여기서만 경로를 잡고, 나머지 모듈은 brain/ 안에서 자족적으로 동작하게 둔다.
# robot/ 이 트리모달 저장소 안에 있으므로 두 단계 위가 저장소 루트다.
DEFAULT_TRIMODAL = Path(__file__).resolve().parent.parent.parent


def make_batch(cfg, device, n_frames: int, audio_sec: float, n_tokens: int):
    """실사용에 가까운 크기의 가짜 입력 한 건.

    발화 하나를 처리하므로 배치 크기는 1이다 — 학습 때의 128과 달라서, 배치로 상각되던
    고정 비용이 그대로 드러난다. 그게 실사용 조건이다.
    """
    sr = cfg.audio_sample_rate
    n_samples = int(audio_sec * sr)
    hop = cfg.audio_hop_length
    t_mel = n_samples // hop + 1

    b = dict(
        mel_spec=torch.randn(1, t_mel, cfg.audio_n_mels, device=device),
        prosody_vec=torch.randn(1, cfg.model.prosody_dim, device=device),
        frames=torch.rand(1, n_frames, 3, cfg.visual_face_size, cfg.visual_face_size, device=device),
        input_ids=torch.randint(0, 1000, (1, n_tokens), device=device),
        attention_mask=torch.ones(1, n_tokens, dtype=torch.long, device=device),
    )
    if cfg.audio_backbone == "wav2vec2":
        b["waveform"] = torch.randn(1, n_samples, device=device)
        b["wav_attention_mask"] = torch.ones(1, n_samples, dtype=torch.long, device=device)
    return b


def get_sync(device):
    if device.type == "cuda":
        return torch.cuda.synchronize
    if device.type == "mps":
        return torch.mps.synchronize
    return lambda: None


def measure_all(fns: dict, n: int, warmup: int, device) -> dict:
    """여러 구간을 **라운드로빈으로** 번갈아 재고 {이름: (중앙값, 표준편차)}를 돌려준다.

    구간을 하나씩 몰아서 재면 안 된다. 처음엔 그렇게 했다가 브랜치 합(398ms)이 전체
    (130ms)보다 큰 값이 나왔다 — 앞선 측정이 남긴 메모리 단편화·캐시 상태가 뒤 측정에만
    영향을 줘서, 나중에 잰 구간이 부풀려진 것이다(오디오 표준편차가 중앙값과 맞먹었다).

    매 반복마다 모든 구간을 한 번씩 재면 그런 드리프트가 전 구간에 고르게 퍼져 서로
    비교 가능해진다. 워밍업도 모든 구간에 대해 먼저 돌린다.
    """
    sync = get_sync(device)
    for _ in range(warmup):
        for fn in fns.values():
            fn()
    sync()

    samples = {k: [] for k in fns}
    for _ in range(n):
        for k, fn in fns.items():
            t0 = time.perf_counter()
            fn()
            sync()
            samples[k].append((time.perf_counter() - t0) * 1000)
    return {k: (statistics.median(v), statistics.stdev(v) if len(v) > 1 else 0.0)
            for k, v in samples.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trimodal", type=Path, default=DEFAULT_TRIMODAL,
                    help="trimodal_emotion_model 저장소 경로")
    ap.add_argument("--config", type=Path, default=None,
                    help="생략 시 <trimodal>/configs/config_si_w2v.yaml (v11)")
    ap.add_argument("--frames", type=int, default=16, help="얼굴 프레임 수")
    ap.add_argument("--audio-sec", type=float, default=3.0, help="발화 길이(초)")
    ap.add_argument("--tokens", type=int, default=20, help="STT 토큰 수")
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--device", default=None, help="cuda/mps/cpu (생략 시 자동)")
    args = ap.parse_args()

    sys.path.insert(0, str(args.trimodal))
    from src.config import load_config
    from src.model import TrimodalEmotionModel

    cfg_path = args.config or (args.trimodal / "configs" / "config_si_w2v.yaml")
    cfg = load_config(cfg_path)

    # config의 pretrained_model은 트리모달 저장소 기준 상대경로다("models/wav2vec2-...").
    # 다른 폴더에서 실행하면 못 찾고, 그 변환본은 gitignore 대상이라 서버에만 있다.
    # 절대경로로 풀어보고, 없으면 HF 허브 id로 대체한다(가중치는 벤치마크에 무관하고
    # 구조만 같으면 되므로 원본 저장소를 써도 결과가 같다).
    if cfg.audio_backbone == "wav2vec2" and not Path(cfg.audio_pretrained).is_absolute():
        local = args.trimodal / cfg.audio_pretrained
        if local.exists():
            cfg.audio_pretrained = str(local)
        else:
            fallback = "facebook/wav2vec2-large-xlsr-53"
            print(f"[bench] {cfg.audio_pretrained} 없음 -> HF 허브 {fallback} 사용 "
                  f"(구조 동일, 속도 측정에 영향 없음)")
            cfg.audio_pretrained = fallback

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available()
                              else ("mps" if torch.backends.mps.is_available() else "cpu"))

    print(f"[bench] config  {cfg_path.name}  (오디오 백본 {cfg.audio_backbone})")
    print(f"[bench] device  {device}")
    print(f"[bench] 입력    발화 {args.audio_sec}초 / 얼굴 {args.frames}프레임 / 토큰 {args.tokens}개")
    print(f"[bench] 측정    워밍업 {args.warmup}회 후 {args.repeat}회 중앙값\n")

    t0 = time.perf_counter()
    model = TrimodalEmotionModel(cfg).to(device).eval()
    load_s = time.perf_counter() - t0
    n_param = sum(p.numel() for p in model.parameters())
    print(f"[bench] 모델 로드 {load_s:.1f}초, 파라미터 {n_param:,}개 "
          f"({n_param * 4 / 1024**3:.2f}GB @ fp32)\n")

    batch = make_batch(cfg, device, args.frames, args.audio_sec, args.tokens)

    with torch.no_grad():
        fns = {"__total__": lambda: model(**batch)}
        if cfg.audio_backbone == "wav2vec2":
            fns["오디오 (wav2vec2)"] = lambda: model.audio_backbone(
                batch["waveform"], wav_attention_mask=batch["wav_attention_mask"])
        else:
            fns["오디오 (멜)"] = lambda: model.audio_backbone(batch["mel_spec"])
        fns["시각 (MobileFaceNet)"] = lambda: model.visual_backbone(batch["frames"])
        fns["텍스트 (KLUE-BERT)"] = lambda: model.text_backbone(
            batch["input_ids"], batch["attention_mask"])

        res = measure_all(fns, args.repeat, args.warmup, device)

    total, total_sd = res.pop("__total__")
    branch_sum = sum(m for m, _ in res.values())

    print(f"{'구간':26} {'중앙값':>10} {'표준편차':>10} {'비중':>8}")
    print("-" * 58)
    for name, (m, sd) in res.items():
        print(f"{name:26} {m:>9.1f}ms {sd:>9.1f}ms {100*m/total:>7.1f}%")
    rest = total - branch_sum
    print(f"{'융합 + 분류기 (나머지)':26} {rest:>9.1f}ms {'':>9} {100*rest/total:>7.1f}%")
    print("-" * 58)
    print(f"{'전체':26} {total:>9.1f}ms {total_sd:>9.1f}ms")
    print()

    # 정합성 검사 — 브랜치 합이 전체를 크게 넘으면 측정이 신뢰할 수 없다는 뜻이다.
    # (라운드로빈 이전 방식에서 합 398ms > 전체 130ms, 3배가 나온 적이 있다)
    #
    # 10% 여유를 두는 이유: 각 구간을 따로 호출하면 전체 forward에는 없는 호출 오버헤드가
    # 조금씩 붙고, 중앙값끼리 더한 값이 전체 중앙값과 정확히 일치할 이유도 없다.
    # 반복이 적을수록 이 오차가 커진다. 실제로 3회 반복에서 1% 초과로 오탐이 났다.
    TOL = 1.10
    if branch_sum > total * TOL:
        over = 100 * (branch_sum / total - 1)
        print(f"⚠ 브랜치 합 {branch_sum:.1f}ms 가 전체 {total:.1f}ms 를 {over:.0f}% 초과 — "
              f"측정 신뢰 불가.")
        print("  --repeat을 늘리거나 --device cpu로 다시 재볼 것. 아래 판정은 참고만.")
        print()

    sec = total / 1000
    if sec <= 0.2:
        verdict = "대화 흐름에 무리 없다. 현 구조 그대로 진행 가능"
    elif sec <= 1.0:
        verdict = "발화 종료 후 약간의 지연. 대기 표현(끄덕임 등)으로 덮을 수 있다"
    elif sec <= 2.0:
        verdict = "체감되는 지연. 경량화를 검토해야 한다"
    else:
        verdict = "실시간 대화에 쓸 수 없다. 경량화나 서버 분리가 선행돼야 한다"
    print(f"[bench] 발화 1건 {sec:.2f}초  ->  {verdict}")

    top = max(res.items(), key=lambda kv: kv[1][0])
    print(f"[bench] 최대 병목: {top[0]} ({100*top[1][0]/total:.0f}%) — 줄인다면 여기부터")


if __name__ == "__main__":
    main()
