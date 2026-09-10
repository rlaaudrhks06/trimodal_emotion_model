"""운율 게이트가 실제로 작동하는지 본다 — 재학습 없이, GPU 없이.

모델카드 13.3이 이렇게 열어뒀다:

> **주의**: 이 게이트가 실제로 그렇게 작동하는지는 검증하지 않았다.
> `g`의 분포를 들여다본 적이 없다.

설계 v3 §5.2의 취지는 "음성 신호가 잡음이 많을 때 안정적인 운율 통계 쪽 비중을
자동으로 높인다"였다. 구현은 이렇다(`src/fusion/gated_prosody.py`):

    g = Sigmoid(W_g · [z_audio_hybrid ; p_a])            원소별 0~1
    z_audio_final = g ⊙ z_audio_hybrid + (1-g) ⊙ Linear(p_a)

즉 **g가 1에 붙어 있으면 운율 항이 사라진다.** 8.30.3절에서 소음 조건의 운율 기여가
0으로 측정됐는데(파형에만 잡음을 넣어 운율을 깨끗하게 남겨줘도 회복 −9.01 vs
−9.30%p, z=0.45), 그 원인이 "게이트가 애초에 운율을 안 본다"일 수 있다.

이걸 먼저 재는 이유: 11.3.2 항목 3번(운율 게이트 애블레이션)은 재학습이 필요한데,
게이트가 이미 죽어 있다면 그 실험의 결과를 미리 알 수 있다. **A100을 쓰기 전에
공짜로 얻을 수 있는 정보다.**

실행:
    python scripts/inspect_prosody_gate.py --checkpoint checkpoints/v11_best.pt -n 96
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config                                       # noqa: E402
from src.datasets.manifest_dataset import (                              # noqa: E402
    ManifestEmotionDataset, make_collate_fn,
)
from src.model import TrimodalEmotionModel                               # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_si_w2v.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/v11_best.pt")
    ap.add_argument("--manifest", default=None, help="생략하면 config의 test_manifest")
    ap.add_argument("-n", "--num", type=int, default=96, help="볼 발화 수")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--save", default=None, metavar="PATH",
                    help="g 원자료를 npz로 저장 (예: results/embeddings/prosody_gate_v11.npz). "
                         "CPU에서 발화당 수 초라 다시 재는 비용이 크다.")
    ap.add_argument("--noise-snr", type=float, default=None, metavar="dB",
                    help="주면 **같은 발화**를 깨끗한 조건과 이 SNR에서 각각 재서 "
                         "게이트가 운율 쪽으로 옮겨가는지 짝지어 비교한다. "
                         "설계 v3 §5.2의 약속을 직접 재는 조건이다(8.30.3절 참고).")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    if cfg.model.prosody_fusion != "gate":
        raise SystemExit(f"이 config는 prosody_fusion={cfg.model.prosody_fusion!r}라 게이트가 없다")

    model = TrimodalEmotionModel(cfg).eval()
    model.load_state_dict(torch.load(ROOT / args.checkpoint, map_location="cpu"))

    # 게이트 출력을 가로챈다. Sequential(Linear, Sigmoid)의 출력이 곧 g다.
    captured: list[torch.Tensor] = []
    model.prosody_gate.gate.register_forward_hook(
        lambda mod, inp, out: captured.append(out.detach().cpu())
    )

    train_cfg = cfg.raw["train"]
    manifest = args.manifest or train_cfg["test_manifest"]

    def measure(noise_snr_db: float | None) -> np.ndarray:
        """조건 하나로 앞에서부터 args.num 발화를 흘려 g를 모은다.

        shuffle=False이고 잡음 시드가 utt_id에서 나오므로, 두 번 부르면 **같은 발화**가
        같은 순서로 나온다 — 짝지어 비교가 성립한다.
        """
        captured.clear()
        ds = ManifestEmotionDataset(
            str(ROOT / manifest), cfg, cache_dir=None,
            prosody_stats_path=train_cfg.get("prosody_stats_path"),
            return_waveform=(cfg.audio_backbone == "wav2vec2"),
            noise_snr_db=noise_snr_db,
        )
        loader = torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=make_collate_fn(cfg.text_pretrained),
        )
        seen = 0
        with torch.no_grad():
            for batch in loader:
                model(
                    mel_spec=batch["mel_spec"], prosody_vec=batch["prosody_vec"],
                    frames=batch["frames"], input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    audio_padding_mask=batch.get("audio_padding_mask"),
                    visual_padding_mask=batch.get("visual_padding_mask"),
                    waveform=batch.get("waveform"),
                    wav_attention_mask=batch.get("wav_attention_mask"),
                )
                seen += batch["prosody_vec"].size(0)
                if seen >= args.num:
                    break
        return torch.cat(captured).numpy()

    g = measure(None)
    g_noisy = measure(args.noise_snr) if args.noise_snr is not None else None
    print(f"\n=== 운율 게이트 g — {g.shape[0]}발화 × {g.shape[1]}차원 ===")
    print("  g=1이면 오디오 표현만 쓰고 운율은 버린다. g=0이면 그 반대다.\n")
    print(f"  평균 {g.mean():.4f} · 중앙값 {np.median(g):.4f} · 표준편차 {g.std():.4f}")
    print(f"  범위 {g.min():.4f} ~ {g.max():.4f}")
    for thr in (0.9, 0.8, 0.7):
        print(f"  g > {thr}: {(g > thr).mean() * 100:5.1f}%    g < {1 - thr:.1f}: "
              f"{(g < 1 - thr).mean() * 100:5.1f}%")

    per_dim = g.mean(axis=0)
    print(f"\n  차원별 평균 g: {per_dim.min():.3f} ~ {per_dim.max():.3f}")
    print(f"  운율 쪽으로 기운 차원(평균 g<0.5): {(per_dim < 0.5).sum()} / {len(per_dim)}")

    per_utt = g.mean(axis=1)
    print(f"  발화별 평균 g: {per_utt.min():.3f} ~ {per_utt.max():.3f} "
          f"(표준편차 {per_utt.std():.4f})")

    # ---- 이것이 결정적인 수치다.
    #
    # 설계 취지는 "상황에 따라 비중을 옮긴다"였다. 그런데 g의 전체 분산은 두 가지가
    # 섞인 것이다: (a) 차원마다 비중이 다른 것, (b) 발화마다 비중이 달라지는 것.
    # 설계가 약속한 것은 (b)뿐이다. (a)는 학습된 상수여도 생긴다.
    #
    # 발화별 평균의 표준편차만 보면 차원 간 차이가 상쇄되어 (b)를 과소평가한다.
    # 그래서 **차원을 고정하고** 발화에 걸친 표준편차를 낸 뒤 차원 평균을 취한다.
    within_dim = g.std(axis=0).mean()      # 같은 차원에서 발화가 바뀔 때 흔들리는 정도 = (b)
    across_dim = g.mean(axis=0).std()      # 차원 간 비중 차이 = (a)
    print(f"\n  분산 분해:")
    print(f"    (a) 차원 간 비중 차이       {across_dim:.4f}")
    print(f"    (b) 같은 차원에서 발화별 변동 {within_dim:.4f}   <- 설계가 약속한 것")
    print(f"    비 (b)/(a) = {within_dim / across_dim:.3f}")

    print("\n  해석 기준: 평균이 1에 붙어 있으면 운율이 버려지고 있다."
          "\n  (b)가 (a)에 비해 작으면 게이트는 '상황에 반응하는 문'이 아니라"
          "\n  '학습된 고정 배합비'로 동작하는 것이다 — 그렇다면 게이트 기구"
          "\n  267,776개는 값을 못 하고, 단순 concat으로 같은 것을 얻을 수 있다.")

    if g_noisy is not None:
        # 설계 v3 §5.2의 약속을 그대로 잰다: 음성이 잡음에 묻히면 운율 쪽 비중이
        # **높아져야** 한다 = g가 내려가야 한다. 같은 발화·같은 순서라 짝지어 비교다.
        from scipy import stats

        n = min(len(g), len(g_noisy))
        d = g_noisy[:n] - g[:n]                      # 음수면 운율 쪽으로 옮겨간 것
        per_utt_d = d.mean(axis=1)
        moved = int((per_utt_d < 0).sum())
        print(f"\n=== 소음 SNR {args.noise_snr}dB에서 게이트가 옮겨가는가 "
              f"({n}발화 짝지어) ===")

        # 방향과 크기를 **따로** 판정한다. 방향이 유의해도 크기가 무의미할 수 있고,
        # 이 게이트가 정확히 그 경우다. 하나로 뭉뚱그리면 "작동한다"는 잘못된 결론이 난다.
        binom = stats.binomtest(moved, n, 0.5, alternative="greater")
        print(f"  [방향] 운율 쪽으로 옮겨간 발화 {moved}/{n} = {moved / n * 100:.1f}% "
              f"(우연이면 50%), 이항검정 p={binom.pvalue:.2e}")

        t, p = stats.ttest_1samp(per_utt_d, 0.0)
        sd = per_utt_d.std(ddof=1)
        print(f"  [크기] Δg 평균 {per_utt_d.mean():+.5f} (표준편차 {sd:.5f}), "
              f"짝지어 t={t:.2f} p={p:.2e}, Cohen's d={per_utt_d.mean() / sd:.3f}")

        # 크기를 무엇과 비교할 것인가 — 게이트가 평소에 실제로 쓰는 범위다.
        within = g.std(axis=0).mean()
        print(f"  [맥락] 소음이 옮긴 양 {abs(d.mean()):.4f} vs 평소 발화별 변동 "
              f"{within:.4f}  ->  {abs(d.mean()) / within * 100:.1f}%")
        print(f"         전체 평균 g: 깨끗 {g[:n].mean():.4f} -> 소음 {g_noisy[:n].mean():.4f}")

        print("\n  해석: 방향이 유의한데 크기가 평소 변동의 몇 %에 그치면, 게이트는"
              "\n  소음을 **감지는 하지만 반응하지 않는** 것이다. 그러면 8.30.3절의"
              "\n  '운율 기여 0'은 운율이 쓸모없어서가 아니라 게이트가 필요한 순간에"
              "\n  충분히 움직이지 않아서다 — 재설계 대상이 운율이 아니라 게이트가 된다.")

    if args.save:
        out = ROOT / args.save
        out.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"g": g} if g_noisy is None else {"g": g, "g_noisy": g_noisy}
        np.savez_compressed(out, **arrays)
        print(f"\n  원자료 저장: {args.save}  ({', '.join(f'{k} {v.shape}' for k, v in arrays.items())})"
              f" — 다시 재지 않아도 된다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
