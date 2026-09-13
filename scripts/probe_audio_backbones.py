"""오디오 백본 후보를 **동결 특징 + 선형 프로브**로 비교한다 — 재학습 전에 싸게 재기.

## 왜 (통합기록 13.10절)

병목은 오디오 브랜치다: 단독 39.5%인데 라벨 상한이 73~74%(8.23절). wav2vec2-XLSR-53을
동결하고 12층 하나만 쓰는 지금 구조가 그 여지의 절반을 못 쓴다. 백본을 바꾸면 얼마나
오르는지를 트리모달 재학습(3시간 × 시드 3) 전에 **반나절짜리 프로브**로 먼저 잰다 —
v12a(24층) 때 쓴 절차와 같다(8.24절).

## 방법 — 화자 독립으로 잰다

`sweep_w2v_layers.py`는 val 안에서 무작위로 나눠 같은 화자가 양쪽에 들어간다(층 비교엔
충분했다). 백본 비교는 배포 조건과 같아야 하므로 **train 부분집합으로 프로브를 맞추고
val 전체(다른 화자)로 채점**한다. 특징은 유효 프레임 마스크 평균(패딩 제외), 분류기는
표준화 + 다항 로지스틱 하나.

후보와 층:
  xlsr53   facebook/wav2vec2-large-xlsr-53   (기준. 지금 쓰는 12층 + 8·16·24)
  wavlm    microsoft/wavlm-large             (SUPERB 감정 과제 1위. 영어 사전학습)
  hubert   facebook/hubert-large-ll60k       (영어 사전학습)
  whisper  openai/whisper-large-v3 인코더     (다국어·한국어 포함, 30초 패딩 → 유효 구간만 평균)

## 읽는 법

  · 7클래스 val 정확도가 XLSR-12층보다 **3%p 이상** 높은 백본이 있으면 트리모달 재학습 후보다.
    프로브 차이가 트리모달 차이로 그대로 옮겨가진 않는다(융합이 흡수) — 방향과 크기의
    감을 잡는 용도다.
  · 우연 수준(최다 클래스)은 val 기준 약 27%.

실행:
    python scripts/probe_audio_backbones.py --n-train 8000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config                          # noqa: E402
from src.datasets.labels import LABEL_TO_IDX, EMOTION_LABELS, normalize_label  # noqa: E402
from src.eval_report import code_provenance                 # noqa: E402

COARSE = {"happy": "긍정", "surprise": "긍정", "angry": "부정", "disgust": "부정",
          "fear": "부정", "sad": "부정", "neutral": "중립"}
BACKBONES = {
    "xlsr53":  ("facebook/wav2vec2-large-xlsr-53", [8, 12, 16, 18, 20, 22, 24]),
    "wavlm":   ("microsoft/wavlm-large",           [6, 12, 18, 24]),
    "hubert":  ("facebook/hubert-large-ll60k",     [6, 12, 18, 24]),
    "whisper": ("openai/whisper-large-v3",         [16, 24, 32]),
}


class WavOnly(Dataset):
    """ManifestEmotionDataset.return_waveform과 같은 규약(sf.read·평균·8초)."""

    def __init__(self, df: pd.DataFrame, sr: int, max_sec: float = 8.0):
        self.df, self.sr, self.max_sec = df.reset_index(drop=True), sr, max_sec

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        w, sr = sf.read(r.wav_path, dtype="float32", always_2d=False)
        if w.ndim > 1:
            w = w.mean(axis=1)
        assert sr == self.sr, (r.wav_path, sr)
        return w[: int(self.max_sec * sr)], LABEL_TO_IDX[normalize_label(str(r.label).strip().lower())]


def collate(items):
    wavs, ys = zip(*items)
    L = max(len(w) for w in wavs)
    arr = np.zeros((len(wavs), L), np.float32); mask = np.zeros((len(wavs), L), np.int64)
    for i, w in enumerate(wavs):
        arr[i, : len(w)] = w; mask[i, : len(w)] = 1
    return torch.from_numpy(arr), torch.from_numpy(mask), torch.tensor(ys)


def masked_mean(h: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    idx = torch.arange(h.size(1), device=h.device)[None]
    m = (idx < lens[:, None]).to(h.dtype)[..., None]
    return (h * m).sum(1) / m.sum(1).clamp(min=1.0)


# ── 백본별 추출기 ────────────────────────────────────────────────────────────

def make_extractor(name: str, hf_id: str, layers: list[int], dev):
    """(파형, 마스크) -> {층: [B, hidden]} 을 돌려주는 함수. 백본마다 입력 규약이 다르다."""
    from transformers import AutoModel, AutoFeatureExtractor
    if name == "whisper":
        from transformers import WhisperModel, WhisperFeatureExtractor
        fe = WhisperFeatureExtractor.from_pretrained(hf_id)
        # transformers 5.x는 체크포인트 dtype(fp16)으로 올린다 — fp32 입력과 부딪혀
        # "Input type (float) and bias type (Half)"로 죽는다. fp32로 강제한다.
        model = WhisperModel.from_pretrained(hf_id, torch_dtype=torch.float32).encoder.to(dev).eval()
        # Whisper 인코더는 30초 로그멜(3000프레임)을 받고 1500프레임을 낸다(20ms).
        # 유효 프레임 = 샘플 수 / 320. 나머지는 패딩이라 평균에서 뺀다.
        def run(wav, mask):
            wavs = [wav[i, : int(mask[i].sum())].cpu().numpy() for i in range(wav.size(0))]
            inp = fe(wavs, sampling_rate=16000, return_tensors="pt").input_features.to(dev)
            out = model(inp, output_hidden_states=True)
            lens = (mask.sum(1) // 320).clamp(min=1).to(dev)
            return {l: masked_mean(out.hidden_states[l], lens) for l in layers}
        return run
    model = AutoModel.from_pretrained(hf_id, torch_dtype=torch.float32).to(dev).eval()
    fe = AutoFeatureExtractor.from_pretrained(hf_id)
    do_norm = getattr(fe, "do_normalize", False)
    use_mask = getattr(fe, "return_attention_mask", False)

    def normalize(wav, mask):
        # XLSR과 같은 규약: 유효 구간에서만 통계 (8.18.4절 버그의 재발을 막는다)
        if not do_norm:
            return wav
        m = mask.to(wav.dtype); n = m.sum(1, keepdim=True).clamp(min=1.0)
        mean = (wav * m).sum(1, keepdim=True) / n
        var = (((wav - mean) * m) ** 2).sum(1, keepdim=True) / n
        return (wav - mean) / torch.sqrt(var + 1e-7) * m

    def run(wav, mask):
        x = normalize(wav, mask)
        # group-norm 계열(hubert-base 등)은 attention_mask를 안 받는 게 공식 권고다.
        # large 계열(layer-norm)은 받는다. FeatureExtractor의 return_attention_mask가 그 표시다.
        out = model(x, attention_mask=mask if use_mask else None, output_hidden_states=True)
        lens = model._get_feat_extract_output_lengths(mask.sum(1))
        return {l: masked_mean(out.hidden_states[l], lens) for l in layers}
    return run


def extract(run, loader, layers, dev, tag):
    feats = {l: [] for l in layers}; ys = []; t0 = time.time()
    with torch.no_grad():
        for i, (wav, mask, y) in enumerate(loader, 1):
            f = run(wav.to(dev), mask.to(dev))
            for l in layers:
                feats[l].append(f[l].float().cpu().numpy())
            ys.append(y.numpy())
            if i % 25 == 0:
                print(f"    [{tag}] {i}/{len(loader)} 배치 {time.time()-t0:.0f}초", flush=True)
    return {l: np.concatenate(v) for l, v in feats.items()}, np.concatenate(ys)


def probe(Xtr, ytr, Xva, yva, max_iter=2000):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=max_iter, C=1.0).fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xva))
    acc7 = float((pred == yva).mean())
    c = np.array([COARSE[EMOTION_LABELS[i]] for i in range(7)])
    acc3 = float((c[pred] == c[yva]).mean())
    return acc7, acc3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_si_w2v.yaml")
    ap.add_argument("--backbones", nargs="+", default=list(BACKBONES))
    ap.add_argument("--n-train", type=int, default=8000, help="프로브를 맞출 train 발화 수(무작위)")
    ap.add_argument("--n-val", type=int, default=0, help="0이면 val 전체")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/probe_audio_backbones.json")
    ap.add_argument("--combos", default="12+24,16+24,12+18+24",
                    help="같은 백본 안에서 층을 이어붙여 재본다(쉼표 구분, 층은 +로). 해당 층이 다 있을 때만")
    ap.add_argument("--save-feats", default=None, help="추출 특징을 npz로 저장(재프로브용)")
    args = ap.parse_args()

    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    cfg = load_config(ROOT / args.config); tc = cfg.raw["train"]; sr = cfg.audio_sample_rate
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    tr = pd.read_csv(ROOT / tc["train_manifest"]); va = pd.read_csv(ROOT / tc["val_manifest"])
    tr = tr.iloc[sorted(rng.choice(len(tr), args.n_train, replace=False))]
    if args.n_val:
        va = va.iloc[sorted(rng.choice(len(va), args.n_val, replace=False))]
    print(f"[probe] train {len(tr):,} (프로브 학습) / val {len(va):,} (채점, 화자 독립) · {dev}")
    mk = lambda df: DataLoader(WavOnly(df, sr), batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate, pin_memory=True)

    results = {}
    chance = float(va["label"].astype(str).str.strip().str.lower().map(normalize_label).value_counts().iloc[0] / len(va))
    print(f"[probe] val 우연 수준(최다 클래스) {chance*100:.2f}%\n")
    for name in args.backbones:
        hf_id, layers = BACKBONES[name]
        if name == "xlsr53" and (ROOT / "models/wav2vec2-large-xlsr-53").exists():
            hf_id = str(ROOT / "models/wav2vec2-large-xlsr-53")
        t0 = time.time()
        try:
            run = make_extractor(name, hf_id, layers, dev)
        except Exception as e:
            print(f"[{name}] 로드 실패 — 건너뜀: {str(e)[:120]}"); continue
        Ftr, ytr = extract(run, mk(tr), layers, dev, f"{name} train")
        Fva, yva = extract(run, mk(va), layers, dev, f"{name} val")
        t_ext = time.time() - t0
        for l in layers:
            a7, a3 = probe(Ftr[l], ytr, Fva[l], yva)
            results[f"{name}/L{l}"] = {"acc7": a7, "acc3": a3, "hidden": int(Ftr[l].shape[1])}
            print(f"  {name:8s} L{l:<3d}  7클래스 {a7*100:6.2f}%  3클래스 {a3*100:6.2f}%   (우연 {chance*100:.1f})", flush=True)
        for combo in [c for c in args.combos.split(",") if c]:
            ls = [int(x) for x in combo.split("+")]
            if not all(l in layers for l in ls):
                continue
            Xtr = np.concatenate([Ftr[l] for l in ls], 1); Xva = np.concatenate([Fva[l] for l in ls], 1)
            a7, a3 = probe(Xtr, ytr, Xva, yva)
            results[f"{name}/L{combo}"] = {"acc7": a7, "acc3": a3, "hidden": int(Xtr.shape[1])}
            print(f"  {name:8s} L{combo:<6s}  7클래스 {a7*100:6.2f}%  3클래스 {a3*100:6.2f}%   (결합)", flush=True)
        if args.save_feats:
            np.savez_compressed(f"{args.save_feats}_{name}.npz", ytr=ytr, yva=yva,
                                **{f"tr_L{l}": Ftr[l] for l in layers}, **{f"va_L{l}": Fva[l] for l in layers})
        print(f"  [{name}] 추출 {t_ext/60:.1f}분\n")
        del run; torch.cuda.empty_cache()

    base = results.get("xlsr53/L12", {}).get("acc7")
    print(f"{'백본/층':16s} {'7클래스':>8s} {'3클래스':>8s} {'vs XLSR-L12':>12s}")
    for k, v in sorted(results.items(), key=lambda kv: -kv[1]["acc7"]):
        d = f"{(v['acc7']-base)*100:+.2f}%p" if base else ""
        print(f"{k:16s} {v['acc7']*100:8.2f} {v['acc3']*100:8.2f} {d:>12s}")
    out = ROOT / args.out; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_train": len(tr), "n_val": len(va), "chance": chance,
                               "results": results, **code_provenance()}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[probe] 저장 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
