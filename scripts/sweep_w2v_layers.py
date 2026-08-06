"""wav2vec2의 어느 층이 감정 정보를 가장 많이 담고 있는지 프로빙으로 고른다.

배경: config의 `w2v_layer: 12`는 "SER에서는 마지막 층보다 중간 층이 낫다"는 일반론만
보고 정한 값이고, 우리 데이터에서 검증한 적이 없다. 그런데 오디오 브랜치는 남은 여지가
가장 큰 곳이다 — 정답 라벨과 소리 라벨의 일치율이 77.35%인데 우리 오디오 단독 모델은
39.48%다(영상은 41.69% 상한에 35.59%로 이미 근접). 층 선택 하나로 출발점이 달라진다.

**전체 학습을 층마다 반복할 필요가 없다.** 이유가 두 가지다:
  1. wav2vec2는 동결이라 순전파만 하면 되고,
  2. `output_hidden_states=True`면 **한 번의 순전파로 25개 층 표현이 전부 나온다.**
그래서 데이터를 한 번만 훑고, 층별로 선형 프로브만 갈아끼우면 된다.
전체 학습 1회가 약 4시간인데 이 방식은 층 7개를 1~2시간에 비교한다.

프로브는 scripts/probe_embeddings.py의 것을 그대로 쓴다 — 다른 실험과 같은 잣대여야
숫자를 나란히 놓고 볼 수 있다.

**test가 아니라 val 매니페스트를 기본값으로 쓴다.** 층은 하이퍼파라미터이므로
test로 고르면 test가 오염되어 최종 성능 보고가 의미를 잃는다.

실행 예:
    python scripts/sweep_w2v_layers.py --config configs/config_si_w2v.yaml --limit 4000
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import load_config
from src.models.audio_backbone_w2v import Wav2Vec2AudioBackbone
from src.datasets.labels import EMOTION_LABELS, LABEL_TO_IDX, normalize_label
from scripts.probe_embeddings import probe


class AudioOnlyDataset(Dataset):
    """파형과 라벨만 읽는 최소 데이터셋.

    ManifestEmotionDataset을 쓰지 않는 이유: 그건 발화마다 멜·운율·얼굴 프레임 32장을
    함께 만들어낸다. 층 비교에는 파형만 있으면 되는데 나머지를 다 만들면 몇 배 느려진다.
    대신 파형 처리 규약(16kHz 확인, max_audio_seconds 자르기)은 원본과 동일하게 맞춘다.
    """

    def __init__(self, manifest: str, sample_rate: int, max_seconds: float = 8.0):
        import pandas as pd
        self.df = pd.read_csv(manifest)
        self.sample_rate = sample_rate
        self.max_seconds = max_seconds

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        wav, sr = sf.read(row.wav_path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != self.sample_rate:
            raise ValueError(f"{row.wav_path}: {sr}Hz인데 config는 {self.sample_rate}Hz를 기대함")
        wav = wav[: int(self.max_seconds * sr)]
        label = LABEL_TO_IDX[normalize_label(str(row.label).strip().lower())]
        return wav, label


def collate(batch):
    wavs, labels = zip(*batch)
    max_len = max(len(w) for w in wavs)
    arr = np.zeros((len(wavs), max_len), dtype=np.float32)
    mask = np.zeros((len(wavs), max_len), dtype=np.int64)
    for i, w in enumerate(wavs):
        arr[i, : len(w)] = w
        mask[i, : len(w)] = 1
    return (torch.from_numpy(arr), torch.from_numpy(mask),
            torch.tensor(labels, dtype=torch.long))


def masked_mean(h: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    """[B, T, D] -> [B, D]. 패딩 프레임을 평균에서 제외한다.

    빼먹으면 짧은 발화일수록 0쪽으로 끌려가, 배치 구성에 따라 특징이 달라진다
    (8.18절에서 같은 함정을 한 번 겪었다).
    """
    idx = torch.arange(h.size(1), device=h.device).unsqueeze(0)
    valid = (idx < lens.unsqueeze(1)).unsqueeze(-1).to(h.dtype)
    return (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", default=None,
                        help="생략하면 config의 val_manifest — 층은 하이퍼파라미터라 "
                             "test로 고르면 test가 오염된다")
    parser.add_argument("--layers", default="0,4,8,12,16,20,24",
                        help="비교할 층 번호(쉼표 구분). 0=임베딩 출력, 24=마지막")
    parser.add_argument("--limit", type=int, default=4000,
                        help="쓸 발화 수 — 층 비교에는 전체가 필요 없다. 0이면 전부")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5,
                        help="층마다 분할 시드를 바꿔 프로브를 여러 번 돌려 평균낸다 — "
                             "한 번만 돌리면 층 간 미세한 차이가 뽑기 운에 좌우된다. "
                             "특징 추출은 이미 끝난 뒤라 반복 비용이 거의 없다")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--save", default=None, help="추출한 층별 특징을 .npz로 저장(선택)")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    train_cfg = cfg.raw["train"]
    manifest = args.manifest or train_cfg["val_manifest"]
    if args.manifest is None:
        print(f"[sweep] --manifest 생략됨 -> config의 val_manifest 사용: {manifest}")

    layers = sorted({int(x) for x in args.layers.split(",")})
    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))

    ds = AudioOnlyDataset(manifest, cfg.audio_sample_rate)
    if args.limit and args.limit < len(ds):
        # 앞에서 자르면 매니페스트 정렬 때문에 특정 화자·클래스에 쏠린다. 무작위 표집.
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(ds), size=args.limit, replace=False)
        ds.df = ds.df.iloc[sorted(keep)].reset_index(drop=True)
    print(f"[sweep] 발화 {len(ds):,}개 / 비교할 층 {layers}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                        num_workers=train_cfg.get("num_workers", 0),
                        pin_memory=(device.type == "cuda"))

    # 백본 전체(TemporalConvFrontend 포함)를 만들지만 여기서는 .w2v와 정규화만 쓴다.
    # 검증된 _normalize(XLSR이 요구하는 do_normalize, 유효 구간에서만 통계)를 재사용하려는 것 —
    # 이걸 다시 구현하면 8.18절에서 잡은 패딩 오염 버그를 되살릴 위험이 있다.
    backbone = Wav2Vec2AudioBackbone(
        pretrained_model=cfg.audio_pretrained, d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads, ffn_dim=cfg.model.ffn_dim, layer=0, freeze=True,
    ).to(device)
    backbone.eval()

    n_layers = backbone.w2v.config.num_hidden_layers
    bad = [l for l in layers if not (0 <= l <= n_layers)]
    if bad:
        raise ValueError(f"층 {bad}는 범위를 벗어남 — 0~{n_layers}만 가능")

    feats = {l: [] for l in layers}
    all_labels = []
    with torch.no_grad():
        for i, (wav, mask, y) in enumerate(loader, 1):
            wav, mask = wav.to(device), mask.to(device)
            normed = backbone._normalize(wav, mask)
            out = backbone.w2v(normed, attention_mask=mask, output_hidden_states=True)
            lens = backbone.output_lengths(mask.sum(dim=-1))
            for l in layers:
                feats[l].append(masked_mean(out.hidden_states[l], lens).cpu().numpy())
            all_labels.append(y.numpy())
            if i % 20 == 0:
                print(f"    {i}/{len(loader)} 배치", flush=True)

    y = np.concatenate(all_labels)
    feats = {l: np.concatenate(v, axis=0) for l, v in feats.items()}
    print(f"[sweep] 추출 완료 — 층당 {feats[layers[0]].shape}\n")

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save, labels=y, **{f"layer_{l}": v for l, v in feats.items()})
        print(f"[sweep] 특징 저장: {args.save}\n")

    print(f"{'층':>4} {'정확도':>9} {'±표준편차':>10} {'우연':>9} {'상승폭':>10}   현재 설정")
    print("─" * 62)
    means, stds = {}, {}
    for l in layers:
        accs, chances = [], []
        for r in range(args.repeats):
            a, c, _ = probe(feats[l], y, args.seed + r, args.max_iter)
            accs.append(a); chances.append(c)
        acc, sd, chance = float(np.mean(accs)), float(np.std(accs)), float(np.mean(chances))
        means[l], stds[l] = acc - chance, sd
        mark = " <- 지금 쓰는 층" if l == cfg.audio_w2v_layer else ""
        print(f"{l:>4} {100*acc:>8.2f}% {100*sd:>9.2f}%p {100*chance:>8.2f}% "
              f"{100*(acc-chance):>+9.2f}%p{mark}")
    print("─" * 62)
    print(f"({args.repeats}회 반복 평균. 프로브 평가셋 약 {int(0.3*len(y)):,}개)")
    print()

    best = max(means, key=means.get)
    cur = cfg.audio_w2v_layer
    # 두 층의 차이가 의미 있는지는 반복 간 변동폭으로 판단한다. 임의의 고정 임계값
    # 대신 실제 표준편차를 쓰는 이유: 표본 수와 층에 따라 변동폭이 다르기 때문이다.
    noise = max(stds[best], stds.get(cur, 0.0))
    print(f"최고: {best}층 (상승폭 {100*means[best]:+.2f}%p)")
    if cur not in means:
        print(f"현재 설정 {cur}층은 --layers에 없어 비교 불가")
    elif best == cur:
        print(f"현재 설정({cur}층)이 이미 최적 — 바꿀 필요 없음")
    else:
        gain = means[best] - means[cur]
        print(f"현재 {cur}층 대비 {100*gain:+.2f}%p (반복 간 표준편차 {100*noise:.2f}%p)")
        if gain > 2 * noise:
            print(f"  -> 변동폭의 2배를 넘음. config의 w2v_layer를 {best}로 바꿀 근거가 된다")
        else:
            print(f"  -> 변동폭 2배({100*2*noise:.2f}%p) 이내라 뽑기 운과 구분되지 않는다.")
            print(f"     바꾸지 말고 --limit을 늘려 다시 재볼 것")
    print()
    print("주의: 프로브 상승폭이 곧 최종 학습 성능 차이는 아니다. 이 결과는")
    print("      '어느 층에 감정 정보가 많은가'이지 '어느 층이 학습에 최적인가'가 아니다.")


if __name__ == "__main__":
    main()
