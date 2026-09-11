"""wav2vec2 캐시 경로 검증 — **캐시로 학습한 것이 파형으로 학습한 것과 같은 모델인가.**

캐시는 값을 미리 계산해두는 것뿐이라 모델 동작이 바뀌면 안 된다. 그런데 바뀔 수 있는
자리가 넷이다. 넷 다 에러 없이 틀리는 종류라 테스트로 박는다.

  ① 캐시 값이 학습 때 파형에서 뽑을 값과 같은가 (정규화·층·자르기 규약)
  ② 소음 캐시가 학습 때 소음 파형에서 뽑을 값과 같은가 (noise_seed 공유)
  ③ 모델 forward: 파형 경로와 캐시 경로의 로짓이 같은가
  ④ 모달리티 드롭아웃: 캐시 경로에서 "0 파형의 출력"이 정확히 대입되는가

실데이터 필요(data/manifests_si/train.csv). 없으면 건너뛴다. GPU 없으면 CPU로
느리게 돈다(발화 4개, 몇 분).

    python tests/test_w2v_cache.py
"""
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from src.config import load_config
from src.datasets.manifest_dataset import (
    ManifestEmotionDataset, add_white_noise, make_collate_fn, noise_seed,
)
from src.model import TrimodalEmotionModel
from scripts.precompute_w2v_cache import fill

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifests_si" / "train.csv"
SNRS = [20.0, 10.0]
N_UTT = 4
# TF32를 끈 채 비교한다 — 캐시가 그렇게 뽑히고, 켜면 배치 모양마다 1e-3 잡음이 생겨
# "같다"를 판정할 수 없다(13.6절 실측). 파형 경로도 이 테스트 안에서는 끈다.
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def assert_close_feat(live, cached, what):
    """특징 [T, 1024] 둘이 '같은 값'인가 — 최대값 하나가 아니라 분포로 판정한다.

    캐시는 길이순 패딩 배치로 뽑고 여기선 혼자 뽑는다. TF32를 꺼도 conv가 입력 길이에
    따라 다른 커널을 타서 원소 몇 개가 1e-2 수준으로 튄다(CPU는 GPU보다 크다). 원소
    T×1024개 중 최대 하나로 판정하면 그 잡음에 걸린다. 평균 상대차가 1e-3 아래이고
    99.9% 원소가 스케일의 1%(0.03) 안이면 같은 값이다 — 진짜 판정은 로짓(③)이 한다.
    시드가 어긋나면 평균 상대차가 O(1)이라 여기서 확실히 잡힌다.
    """
    d = (live - cached).abs()
    scale = live.abs().mean().item()
    mean_rel = d.mean().item() / scale
    q999 = torch.quantile(d.flatten().float(), 0.999).item()
    assert mean_rel < 1e-3, f"{what}: 평균 상대차 {mean_rel:.2e} — 시드·규약이 어긋났다"
    assert q999 < 0.01 * scale * 3, f"{what}: 99.9% 분위 {q999:.2e} (스케일 {scale:.2f})"
    return mean_rel, d.max().item()


def load_wav(path, sr, max_sec=8.0):
    """ManifestEmotionDataset.return_waveform 경로와 같은 규약."""
    w, r = sf.read(path, dtype="float32", always_2d=False)
    if w.ndim > 1:
        w = w.mean(axis=1)
    assert r == sr
    return w[: int(max_sec * sr)]


def test_cache_equals_live_extract(bb, df, cache, sr, dev):
    """① 캐시 == 파형에서 그 자리에서 뽑은 값 (fp16 반올림 안에서)."""
    worst = worst_mx = 0.0
    for r in df.itertuples():
        w = torch.from_numpy(load_wav(r.wav_path, sr))[None].to(dev)
        m = torch.ones_like(w, dtype=torch.long)
        live = bb.extract_features(w, m)[0].half().float()
        cached = torch.from_numpy(np.load(cache / "clean" / f"{r.utt_id}.npy")).float().to(dev)
        assert live.shape == cached.shape, (r.utt_id, live.shape, cached.shape)
        mr, mx = assert_close_feat(live, cached, r.utt_id)
        worst = max(worst, mr); worst_mx = max(worst_mx, mx)
    print(f"  ✅ 캐시 == 실시간 추출 ({len(df)}발화, 평균 상대차 최악 {worst:.2e}, 원소 최대차 {worst_mx:.2e})")


def test_noisy_cache_equals_live(bb, df, cache, sr, dev):
    """② 소음 캐시 == add_white_noise(파형, noise_seed) 뒤 추출. 시드가 어긋나면 여기서 잡힌다."""
    for s in SNRS:
        for r in df.itertuples():
            w = add_white_noise(load_wav(r.wav_path, sr), s, seed=noise_seed(str(r.utt_id), s))
            w = torch.from_numpy(w)[None].to(dev)
            live = bb.extract_features(w, torch.ones_like(w, dtype=torch.long))[0].half().float()
            cached = torch.from_numpy(np.load(cache / f"snr{s:g}" / f"{r.utt_id}.npy")).float().to(dev)
            assert_close_feat(live, cached, f"{r.utt_id} SNR{s:g} (소음 캐시가 학습 파형과 다른 잡음?)")
        # 깨끗과 실제로 다른가 — 같으면 소음이 안 들어간 것
        c0 = np.load(cache / "clean" / f"{df.iloc[0].utt_id}.npy").astype(np.float32)
        c1 = np.load(cache / f"snr{s:g}" / f"{df.iloc[0].utt_id}.npy").astype(np.float32)
        assert np.abs(c0 - c1).mean() > 1e-2, f"SNR{s:g} 캐시가 깨끗한 것과 같다"
    print(f"  ✅ 소음 캐시 == noise_seed로 오염한 파형의 추출 (SNR {SNRS})")


def test_forward_parity(model, ds_wave, ds_cache, collate, dev):
    """③ 같은 발화, 파형 경로 vs 캐시 경로 -> 로짓이 같은가. eval 모드(드롭아웃 없음)."""
    model.eval()
    bw = collate([ds_wave[i] for i in range(len(ds_wave))])
    bc = collate([ds_cache[i] for i in range(len(ds_cache))])
    assert "waveform" in bw and "audio_feat" not in bw
    assert "audio_feat" in bc and "waveform" not in bc, "캐시를 켜면 파형을 안 읽어야 한다"
    assert bc["audio_feat"].dtype == torch.float32
    common = dict(mel_spec=bw["mel_spec"], prosody_vec=bw["prosody_vec"], frames=bw["frames"],
                  input_ids=bw["input_ids"], attention_mask=bw["attention_mask"],
                  audio_padding_mask=bw["audio_padding_mask"], visual_padding_mask=bw["visual_padding_mask"])
    common = {k: v.to(dev) for k, v in common.items()}
    with torch.no_grad():
        lw = model(**common, waveform=bw["waveform"].to(dev), wav_attention_mask=bw["wav_attention_mask"].to(dev))
        lc = model(**common, audio_feat=bc["audio_feat"].to(dev), audio_feat_padding_mask=bc["audio_feat_padding_mask"].to(dev))
    d = (lw - lc).abs().max().item()
    # fp16 캐시가 12층 뒤 proj·frontend·융합·분류기를 거친 뒤의 차이. 8.18.4의 판정
    # 기준(배치 구성에 따른 로짓 차이 0.0006)보다는 크지만, 이 잡음은 TF32를 켠 v11이
    # 매 에폭 겪는 수준(평균 3e-3)보다 작아야 한다.
    assert d < 5e-2, f"파형 경로와 캐시 경로 로짓이 다르다 (최대차 {d:.3e})"
    assert torch.equal(lw.argmax(1), lc.argmax(1)), "예측이 다르다"
    print(f"  ✅ 파형 경로 == 캐시 경로 (로짓 최대차 {d:.2e}, 예측 일치)")


def test_dropout_substitution(model, ds_cache, collate, dev):
    """④ 드롭아웃이 오디오를 지울 때, 캐시 경로가 '길이 T의 0 파형 출력'을 정확히 넣는가.

    표의 내용(0 파형 출력이 맞는가)과 배선(그 표에서 꺼내 그 자리에 넣는가)을 따로 본다.
    """
    bb = model.audio_backbone
    b = collate([ds_cache[i] for i in range(len(ds_cache))])
    feat = b["audio_feat"].to(dev); mask = b["audio_feat_padding_mask"].to(dev)
    # (a) 표 내용: zero_feature(T) == 0 파형(길이 T프레임) 실시간 추출
    T = int((~mask[0]).sum())
    n = 320 * (T - 1) + 400
    z = torch.zeros(1, n, device=dev); m = torch.ones(1, n, dtype=torch.long, device=dev)
    live = bb.extract_features(z, m)[0].half().float()
    tab = bb.zero_feature(T).float()
    assert live.shape == tab.shape
    # 표도 혼자 뽑았고 여기서도 혼자 뽑으니 fp16 반올림 외엔 차이가 없어야 한다
    d = (live - tab).abs().max().item()
    assert d < 2e-2, f"0-특징 표가 실제 0 파형 출력과 다르다 ({d:.3e})"
    # (b) 배선: 드롭아웃을 오디오로 강제하고, 대입된 값이 표와 같은지.
    # 모델 기본값은 드롭아웃 0이라(train.py가 config로 넣는다) 여기서 켠다 — 안 켜면
    # _maybe_drop_modalities가 즉시 반환해 아무것도 검증하지 않고 통과한다.
    model.train()
    old_p, model.modality_dropout_prob = model.modality_dropout_prob, 1.0
    old_r, old_c = random.random, random.choice
    random.random = lambda: 0.0                      # 항상 드롭
    random.choice = lambda seq: "audio"              # 항상 오디오
    try:
        out = model._maybe_drop_modalities(
            b["mel_spec"].to(dev), b["prosody_vec"].to(dev), b["frames"].to(dev),
            b["input_ids"].to(dev), b["attention_mask"].to(dev), None, feat, mask,
        )
    finally:
        random.random, random.choice = old_r, old_c
        model.modality_dropout_prob = old_p
    dropped = out[-1]
    for i in range(feat.shape[0]):
        Ti = int((~mask[i]).sum())
        assert torch.equal(dropped[i, :Ti], bb.zero_feature(Ti).float()), f"발화 {i}: 대입된 값이 표와 다르다"
        assert torch.equal(dropped[i, Ti:], feat[i, Ti:]), f"발화 {i}: 패딩 자리가 건드려졌다"
    assert not torch.equal(dropped, feat), "드롭아웃이 아무것도 안 바꿨다"
    assert out[1].abs().sum() == 0, "운율이 같이 안 지워졌다"
    model.eval()
    print(f"  ✅ 드롭아웃: 0-특징 표 내용 일치(최대차 {d:.2e}) · 대입 위치·길이 정확")


def test_guards(cfg, mini, cache):
    def fails(exc, **kw):
        try:
            ManifestEmotionDataset(str(mini), cfg, return_waveform=True, **kw)
        except exc as e:
            return str(e)[:70]
        return None
    m = fails(ValueError, w2v_cache_dir=cache, noise_snr_db=10.0)
    assert m, "평가용 고정 SNR + 캐시가 통과했다"
    print(f"  ✅ 고정 SNR + 캐시 차단 — {m}")
    # 운율 캐시 검사가 w2v 검사보다 먼저 돈다. w2v 쪽 가드를 검증하려면 운율 캐시는
    # 있어야 하므로 SNR 5 운율을 가짜로 만들어 두고, w2v 캐시의 snr5/만 없게 한다.
    df = pd.read_csv(mini)
    np.savez_compressed(cache / "snr5.npz", utt_ids=np.array([str(u) for u in df["utt_id"]]),
                        prosody=np.zeros((len(df), 10), np.float32))
    m = fails(FileNotFoundError, w2v_cache_dir=cache, noise_aug_snrs=[5.0], noisy_prosody_dir=cache)
    assert m and "wav2vec2 캐시" in m, f"없는 SNR 조건의 w2v 캐시인데 통과했거나 다른 가드에 걸렸다: {m}"
    print(f"  ✅ 없는 조건 차단 — {m}")
    m = fails(FileNotFoundError, w2v_cache_dir=cache / "nope")
    assert m, "없는 디렉터리인데 통과했다"
    print(f"  ✅ 없는 디렉터리 차단 — {m}")


def main() -> int:
    print("=== wav2vec2 캐시 검증 ===")
    if not MANIFEST.exists():
        print(f"  ⏭  실데이터 없음({MANIFEST}) — 건너뜀")
        return 0
    cfg = load_config(ROOT / "configs" / "config_si_w2v.yaml")
    tc = cfg.raw["train"]
    df = pd.read_csv(MANIFEST).head(N_UTT)
    if not all(Path(p).exists() for p in df["wav_path"]):
        print("  ⏭  wav 없음 — 건너뜀")
        return 0
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  장치 {dev} · 발화 {N_UTT} · SNR {SNRS}")
    torch.manual_seed(0)
    model = TrimodalEmotionModel(cfg).to(dev).eval()
    bb = model.audio_backbone
    sr = cfg.audio_sample_rate

    with tempfile.TemporaryDirectory() as td:
        td = Path(td); mini = td / "mini.csv"; df.to_csv(mini, index=False)
        cache = td / "w2v_cache"
        # 캐시 채우기 — 본 스크립트의 fill()을 그대로 쓴다(테스트가 자기 구현을 검증하지 않게)
        fill(bb, df, cache / "clean", sr, 8.0, None, 2, 0, dev)
        for s in SNRS:
            fill(bb, df, cache / f"snr{s:g}", sr, 8.0, s, 2, 0, dev)
        # 0-표는 이 발화들의 프레임 수만 (CPU에서 399번 안 돌게)
        counts = [np.load(cache / "clean" / f"{u}.npy").shape[0] for u in df["utt_id"]]
        max_frames = int(bb.output_lengths(torch.tensor([8 * sr]))[0])
        table = bb.build_zero_table(max_frames, dev, frame_counts=counts)
        torch.save(table.cpu(), cache / "zero_table.pt")
        bb.set_zero_table(torch.load(cache / "zero_table.pt", map_location=dev))

        test_cache_equals_live_extract(bb, df, cache, sr, dev)
        test_noisy_cache_equals_live(bb, df, cache, sr, dev)

        collate = make_collate_fn(cfg.text_pretrained)
        common = dict(cache_dir=None, prosody_stats_path=tc["prosody_stats_path"], return_waveform=True)
        ds_wave = ManifestEmotionDataset(str(mini), cfg, **common)
        ds_cache = ManifestEmotionDataset(str(mini), cfg, **common, w2v_cache_dir=cache)
        test_forward_parity(model, ds_wave, ds_cache, collate, dev)
        test_dropout_substitution(model, ds_cache, collate, dev)
        test_guards(cfg, mini, cache)

    print("=== 전부 통과 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
