"""동결 wav2vec2의 12층 출력을 발화별로 미리 뽑아둔다 — 학습 한 스텝의 84%를 없앤다.

## 왜 (통합기록 13.6절)

학습 한 스텝의 시간을 재보니(A100, v11 설정) 이렇다.

    데이터 로드     0.00s    0%      wav2vec2 forward   2.05s   84%   <- 동결, no_grad
    나머지 forward  0.30s   12%      backward+step      0.09s    4%
    합계 2.43s/스텝 × 443 = 18분/에폭

84%가 **매 에폭 같은 값을 다시 계산하는** 시간이다. wav2vec2는 동결이라 같은 파형이면
40에폭 내내 같은 출력을 낸다. 멜·운율·얼굴을 캐시한 것(8.1절)과 같은 논리를 한 단계
위에 적용하면 18분 -> 약 3분이다. 8.19절에서 "검토하다가" 캐시 용량 문제를 발견해 그쪽을
고치고 끝났고, 안 한 이유("층을 바꿔가며 실험하려고")는 v12a에서 24층이 실패하고
12층으로 확정되면서 사라졌다.

    증강 학습 40에폭        12시간 -> 2시간
    애블레이션 4종 × 시드 3  144시간 -> 24시간   (시드 2~3개 규칙이 비로소 성립한다)

## 값이 배치 경로와 완전히 같지는 않다 — 그리고 그건 이미 있던 잡음이다

발화 하나를 혼자 넣었을 때와 배치에 섞였을 때 wav2vec2 출력이 다르다: 최대 2.3, 평균
1e-3. 마스킹 버그가 아니라(정규화 파형 차이 0, 배치 자기 재현 100%) **cudnn TF32**다 —
conv 추출기의 10비트 가수 잡음이 12층을 지나며 일부 원소에서 O(1)이 된다. 끄면 100배
줄어든다(최대 0.015). 배치 길이가 바뀌면 값이 바뀌므로 v11 학습은 매 에폭 셔플마다
이 잡음을 겪어왔다. v11 체크포인트로 판정하면 test 1,024발화 중 1건 예측이 바뀐다
(99.90%). fp16 저장은 여기에 아무것도 더하지 않는다(측정으로 확인).

그래서 여기서는 **TF32를 끄고** 뽑는다. 값이 수학적 fp32에 가까워지고, 그 덕에 패딩
배치로 뽑아도 혼자 뽑은 값과 1e-5 안에서 같아 배치 처리가 가능하다.

## 드롭아웃용 0-특징 표

모달리티 드롭아웃은 파형을 0으로 만들고 마스크는 그대로 둔다. 그러면 wav2vec2는
"길이 T의 0 입력"을 받는데 그 출력은 T에 따라 다르다(위치 conv 가장자리, 실측 26.8).
캐시 경로에서 v11과 같은 동작을 재현하려면 프레임 길이 1~399별 출력을 표로 들고 있어야
한다. `zero_table.pt` [400, 399, 1024] fp16, 약 325MB. 체크포인트에는 안 들어간다.

## 산출물

    data/w2v_cache/clean/{utt}.npy       fp16 [T_a, 1024]   train·val·test 전부
    data/w2v_cache/snr{N}/{utt}.npy      fp16 [T_a, 1024]   train만 (증강용)
    data/w2v_cache/zero_table.pt
    data/w2v_cache/meta.json             모델·층·max_sec·TF32·git 커밋

발화별 파일인 이유: feature_cache와 같은 규약이고, 데이터셋이 발화 단위로 읽는다.
np.savez_compressed가 아니라 np.save — fp16 특징은 압축이 안 된다.

실행:
    python scripts/precompute_w2v_cache.py --splits train val test --snr 20 10 5
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config                                       # noqa: E402
from src.datasets.manifest_dataset import add_white_noise, noise_seed    # noqa: E402
from src.eval_report import code_provenance                              # noqa: E402
from src.model import TrimodalEmotionModel                               # noqa: E402


class WavOnly(Dataset):
    """파형과 utt_id만. ManifestEmotionDataset의 return_waveform 경로와 **같은 규약**으로
    읽는다(sf.read, 스테레오 평균, sr 확인, max_sec 자르기, 소음은 noise_seed로). 규약이
    어긋나면 캐시가 학습 때의 파형과 다른 것을 담는데, 그건 에러 없이 틀린다."""

    def __init__(self, df: pd.DataFrame, sr: int, max_sec: float, snr_db: float | None):
        self.df, self.sr, self.max_sec, self.snr_db = df, sr, max_sec, snr_db

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        utt = str(r.utt_id)
        wav, sr = sf.read(r.wav_path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != self.sr:
            raise ValueError(f"{r.wav_path}: {sr}Hz != {self.sr}Hz")
        wav = wav[: int(self.max_sec * sr)]
        if self.snr_db is not None:
            wav = add_white_noise(wav, self.snr_db, seed=noise_seed(utt, self.snr_db))
        return utt, wav


def collate(items):
    utts, wavs = zip(*items)
    L = max(len(w) for w in wavs)
    arr = np.zeros((len(wavs), L), dtype=np.float32)
    mask = np.zeros((len(wavs), L), dtype=np.int64)
    for i, w in enumerate(wavs):
        arr[i, : len(w)] = w
        mask[i, : len(w)] = 1
    return list(utts), torch.from_numpy(arr), torch.from_numpy(mask)


def fill(backbone, df, out: Path, sr, max_sec, snr_db, batch_size, workers, dev) -> int:
    out.mkdir(parents=True, exist_ok=True)
    todo = df[~df["utt_id"].astype(str).map(lambda u: (out / f"{u}.npy").exists())]
    if len(todo) == 0:
        print(f"  [{out.name}] 이미 전부 있음 — 건너뜀 ({len(df):,}건)")
        return 0
    # 길이순 정렬: 배치 안 패딩을 줄여 헛계산을 줄인다 (값에는 영향 없다 — TF32를 껐다).
    lens = todo["wav_path"].map(lambda p: sf.info(p).frames)
    todo = todo.assign(_len=lens).sort_values("_len")
    dl = DataLoader(WavOnly(todo, sr, max_sec, snr_db), batch_size=batch_size,
                    num_workers=workers, collate_fn=collate, pin_memory=True)
    t0, n = time.time(), 0
    for utts, wav, mask in dl:
        wav, mask = wav.to(dev, non_blocking=True), mask.to(dev, non_blocking=True)
        h = backbone.extract_features(wav, mask)                    # [B, T_max, 1024]
        fl = backbone.output_lengths(mask.sum(1)).tolist()
        h16 = h.half().cpu().numpy()
        for i, u in enumerate(utts):
            feat = h16[i, : fl[i]]
            if not np.isfinite(feat).all():
                raise RuntimeError(f"{u}: 비정상 값 — 조용히 저장하면 학습이 에러 없이 망가진다")
            tmp = out / f"{u}.tmp.npy"
            np.save(tmp, feat)
            tmp.replace(out / f"{u}.npy")                            # 원자적
        n += len(utts)
        if n % (batch_size * 50) < batch_size:
            el = time.time() - t0
            print(f"  [{out.name}] {n:,}/{len(todo):,}  {n/el:.0f}건/초  남은 {(len(todo)-n)/(n/el)/60:.0f}분", flush=True)
    print(f"  [{out.name}] 완료 {n:,}건 {(time.time()-t0)/60:.1f}분", flush=True)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_si_w2v.yaml")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--snr", type=float, nargs="*", default=[20.0, 10.0, 5.0],
                    help="train에만 적용. 빈 목록이면 깨끗한 캐시만")
    ap.add_argument("--out-dir", default="data/w2v_cache")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-audio-seconds", type=float, default=8.0,
                    help="ManifestEmotionDataset 기본값과 같아야 한다")
    args = ap.parse_args()

    # TF32를 끈다 — 이유는 모듈 docstring. 이 프로세스 안에서만이다.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    cfg = load_config(ROOT / args.config)
    if cfg.audio_backbone != "wav2vec2":
        raise SystemExit(f"wav2vec2 백본 config가 아니다: {cfg.audio_backbone}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrimodalEmotionModel(cfg).to(dev).eval()
    bb = model.audio_backbone
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sr = cfg.audio_sample_rate

    max_frames = int(bb.output_lengths(torch.tensor([int(args.max_audio_seconds * sr)]))[0])
    print(f"[w2v-cache] {cfg.audio_pretrained} layer={cfg.audio_w2v_layer} · 최대 {max_frames}프레임 · {dev} · TF32 끔")

    # 0-특징 표
    zt = out_dir / "zero_table.pt"
    if zt.exists():
        print(f"  [zero_table] 이미 있음 — 건너뜀")
    else:
        t0 = time.time()
        table = bb.build_zero_table(max_frames, dev).cpu()
        torch.save(table, zt)
        print(f"  [zero_table] {tuple(table.shape)} fp16 {zt.stat().st_size/2**20:.0f}MB {time.time()-t0:.0f}초")

    # 깨끗: 요청한 분할 전부의 합집합
    frames = [pd.read_csv(ROOT / cfg.raw["train"][f"{s}_manifest"]) for s in args.splits]
    df_all = pd.concat(frames).drop_duplicates("utt_id")
    print(f"  깨끗 {len(df_all):,}발화 ({', '.join(args.splits)})")
    fill(bb, df_all, out_dir / "clean", sr, args.max_audio_seconds, None, args.batch_size, args.workers, dev)

    # 소음: train만
    if args.snr:
        df_tr = pd.read_csv(ROOT / cfg.raw["train"]["train_manifest"])
        for s in args.snr:
            print(f"  소음 SNR {s:g} — train {len(df_tr):,}발화")
            fill(bb, df_tr, out_dir / f"snr{s:g}", sr, args.max_audio_seconds, s, args.batch_size, args.workers, dev)

    (out_dir / "meta.json").write_text(json.dumps({
        "pretrained": cfg.audio_pretrained, "layer": cfg.audio_w2v_layer,
        "max_audio_seconds": args.max_audio_seconds, "max_frames": max_frames,
        "dtype": "float16", "tf32": False, "snr": args.snr, "splits": args.splits,
        **code_provenance(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[w2v-cache] 끝. 검증: python tests/test_w2v_cache.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
