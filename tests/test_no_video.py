"""영상 없는 발화(AI Hub 263) 경로 검증 — 검은 프레임 1장이 드롭아웃의 '영상 지움'과 같은가.

  ① face_frames_dir이 빈 값/NaN이면 frames가 [1,3,H,W] uint8 0, 마스크는 유효 1개
  ② 그 텐서가 모달리티 드롭아웃이 영상을 지운 결과와 torch.equal (첫 프레임 기준)
  ③ 경로가 있는데 디렉터리가 없으면 여전히 에러 (조용히 넘기지 않는다)
  ④ 영상 없는 행과 있는 행이 한 배치에 섞여도 collate·forward가 돈다

실데이터 필요(train.csv). 없으면 건너뛴다.
    python tests/test_no_video.py
"""
import random, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd, torch
from src.config import load_config
from src.datasets.manifest_dataset import ManifestEmotionDataset, make_collate_fn
from src.model import TrimodalEmotionModel

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifests_si" / "train.csv"


def main() -> int:
    print("=== 영상 없음 경로 검증 ===")
    if not MANIFEST.exists():
        print("  ⏭  실데이터 없음"); return 0
    cfg = load_config(ROOT / "configs" / "config_si_w2v.yaml"); tc = cfg.raw["train"]
    df = pd.read_csv(MANIFEST).head(3)
    if not all(Path(p).exists() for p in df.wav_path):
        print("  ⏭  wav 없음"); return 0
    # 행 0·1은 영상 없음(빈 값·NaN), 행 2는 있음
    df = df.copy(); df.loc[df.index[0], "face_frames_dir"] = ""; df.loc[df.index[1], "face_frames_dir"] = np.nan
    with tempfile.TemporaryDirectory() as td:
        m = Path(td) / "m.csv"; df.to_csv(m, index=False)
        ds = ManifestEmotionDataset(str(m), cfg, prosody_stats_path=tc["prosody_stats_path"], return_waveform=True)
        s = cfg.visual_face_size
        for i in (0, 1):
            f = ds[i]["frames"]
            assert f.shape == (1, 3, s, s) and f.dtype == np.float32 and f.max() == 0.0, (i, f.shape, f.dtype, f.max())
        assert ds[2]["frames"].shape[0] > 1 and ds[2]["frames"].max() > 0, "영상 있는 행이 망가졌다"
        print(f"  ✅ 빈 값·NaN -> 검은 프레임 1장 [1,3,{s},{s}] · 있는 행은 그대로")

        collate = make_collate_fn(cfg.text_pretrained)
        b = collate([ds[i] for i in range(3)])
        assert b["frames"].shape[1] == ds[2]["frames"].shape[0], "패딩 길이는 가장 긴 것"
        assert (~b["visual_padding_mask"][0]).sum() == 1 and (~b["visual_padding_mask"][1]).sum() == 1, "영상 없음은 유효 프레임 1개"
        assert b["frames"][0].abs().sum() == 0, "검은 프레임이 collate에서 바뀜"
        print("  ✅ collate: 유효 프레임 1개 · 나머지 패딩 · 배치 섞임 OK")

        # ② 드롭아웃이 영상을 지운 것과 같은가: 행 2(영상 있음)에 드롭을 강제 -> 첫 프레임이 0
        model = TrimodalEmotionModel(cfg); model.train(); model.modality_dropout_prob = 1.0
        old_r, old_c = random.random, random.choice
        random.random, random.choice = (lambda: 0.0), (lambda seq: "visual")
        try:
            out = model._maybe_drop_modalities(b["mel_spec"], b["prosody_vec"], b["frames"], b["input_ids"], b["attention_mask"], b["waveform"])
        finally:
            random.random, random.choice = old_r, old_c
        dropped_frames = out[2]
        assert torch.equal(dropped_frames[2, 0], b["frames"][0, 0]), "드롭아웃이 지운 프레임 != 검은 프레임"
        print("  ✅ 검은 프레임 == 드롭아웃이 지운 프레임 (torch.equal)")

        # ④ forward
        model.eval()
        with torch.no_grad():
            lg = model(mel_spec=b["mel_spec"], prosody_vec=b["prosody_vec"], frames=b["frames"], input_ids=b["input_ids"],
                       attention_mask=b["attention_mask"], audio_padding_mask=b["audio_padding_mask"],
                       visual_padding_mask=b["visual_padding_mask"], waveform=b["waveform"], wav_attention_mask=b["wav_attention_mask"])
        assert lg.shape == (3, 7) and torch.isfinite(lg).all(), "forward 출력 이상"
        print("  ✅ forward: 영상 없음/있음 섞인 배치에서 로짓 유한")

        # ③ 경로가 있는데 없는 디렉터리
        df2 = df.copy(); df2.loc[df2.index[0], "face_frames_dir"] = "/nonexistent/dir"
        m2 = Path(td) / "m2.csv"; df2.to_csv(m2, index=False)
        ds2 = ManifestEmotionDataset(str(m2), cfg, return_waveform=True)
        try:
            ds2[0]; raise AssertionError("없는 디렉터리인데 통과했다")
        except FileNotFoundError as e:
            print(f"  ✅ 잘못된 경로는 여전히 에러 — {str(e)[:50]}")
    print("=== 전부 통과 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
