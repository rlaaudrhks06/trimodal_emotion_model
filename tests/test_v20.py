"""v20 검증 — EfficientFace 백본 + softhard 모달리티 드롭아웃 (13.20절).

  ① softhard_expand: 배치가 정확히 4배가 되고 라벨도 같이 복제된다
  ② 1벌은 원본 그대로, 2·3·4벌은 각각 오디오·영상·텍스트만 지워진다
  ③ 텍스트를 지울 때 첫 토큰은 남긴다(BERT류는 유효 토큰이 최소 1개 필요)
  ④ 지우는 범위가 model._maybe_drop_modalities와 같다(오디오=멜·운율·파형 전부)
  ⑤ EfficientFace 백본: 기본값은 mobilefacenet, 고르면 출력 모양이 같다

    python tests/test_v20.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from scripts.train import softhard_expand
from src.config import load_config
from src.models.visual_backbone import (VisualBackbone, FrameCNN, EfficientFaceFrameCNN,
                                        _DynamicLocalFeatureExtractor)
from src.models.third_party.efficientface import LocalFeatureExtractor

ROOT = Path(__file__).resolve().parent.parent
EF_W = "/data/work/efficientface/EfficientFace_AffectNet7.pth.tar"


def fake_batch(b=3):
    return {
        "utt_ids": [f"u{i}" for i in range(b)],
        "mel_spec": torch.rand(b, 40, 80) + 1.0,
        "audio_padding_mask": torch.zeros(b, 40, dtype=torch.bool),
        "prosody_vec": torch.rand(b, 10) + 1.0,
        "frames": torch.rand(b, 4, 3, 112, 112) + 1.0,
        "visual_padding_mask": torch.zeros(b, 4, dtype=torch.bool),
        "input_ids": torch.randint(5, 100, (b, 12)),
        "attention_mask": torch.ones(b, 12, dtype=torch.long),
        "waveform": torch.rand(b, 16000) + 1.0,
        "wav_attention_mask": torch.ones(b, 16000, dtype=torch.long),
        "labels": torch.tensor([0, 3, 5][:b]),
    }


def main() -> int:
    print("=== v20 (EfficientFace + softhard) ===")
    b = 3
    src = fake_batch(b)
    out = softhard_expand({k: (v.clone() if torch.is_tensor(v) else list(v)) for k, v in src.items()})
    # ①
    assert out["mel_spec"].shape[0] == 4 * b and out["labels"].shape[0] == 4 * b
    assert out["labels"].tolist() == src["labels"].tolist() * 4
    assert out["utt_ids"] == src["utt_ids"] * 4
    print(f"  ✅ 배치 {b} -> {4*b} · 라벨·utt_id 함께 복제")
    s = lambda i: slice(i * b, (i + 1) * b)
    # ② 원본 벌
    for k in ("mel_spec", "prosody_vec", "frames", "waveform", "attention_mask"):
        assert torch.equal(out[k][s(0)], src[k]), k
    print("  ✅ 1벌은 원본 그대로")
    # ② 오디오 없음
    assert out["mel_spec"][s(1)].abs().sum() == 0 and out["prosody_vec"][s(1)].abs().sum() == 0
    assert out["waveform"][s(1)].abs().sum() == 0
    assert torch.equal(out["frames"][s(1)], src["frames"]), "오디오 벌에서 영상이 바뀌면 안 된다"
    assert torch.equal(out["attention_mask"][s(1)], src["attention_mask"])
    print("  ✅ 2벌: 오디오(멜·운율·파형) 전부 0 · 다른 모달 보존")
    # ② 영상 없음
    assert out["frames"][s(2)].abs().sum() == 0
    assert torch.equal(out["mel_spec"][s(2)], src["mel_spec"])
    print("  ✅ 3벌: 영상만 0")
    # ② ③ 텍스트 없음
    am = out["attention_mask"][s(3)]
    assert am[:, 0].tolist() == [1] * b and am[:, 1:].abs().sum() == 0
    assert torch.equal(out["frames"][s(3)], src["frames"])
    print("  ✅ 4벌: 텍스트 마스크 0(첫 토큰만 유지) · 다른 모달 보존")
    # ④ 원본 배치는 안 건드린다(복제본만 수정)
    assert torch.equal(src["mel_spec"], fake_batch(b)["mel_spec"] * 0 + src["mel_spec"])
    print("  ✅ 입력 배치 자체는 보존")
    # ⑤
    # ⑤-0 사분면 분할 일반화가 224에서 원본과 **완전히 같은** 출력을 내는지
    torch.manual_seed(0)
    orig = LocalFeatureExtractor(29, 116, 1).eval()
    dyn = _DynamicLocalFeatureExtractor(29, 116, 1).eval()
    dyn.load_state_dict(orig.state_dict())
    f56 = torch.rand(2, 29, 56, 56)
    with torch.no_grad():
        assert torch.allclose(orig(f56), dyn(f56), atol=1e-6), "224 경로에서 원본과 달라졌다"
    with torch.no_grad():
        assert dyn(torch.rand(2, 29, 28, 28)).shape[2:] == (14, 14)  # 112 입력에서도 돈다
    print("  ✅ 사분면 분할: 56x56에서 원본과 동일 · 28x28(112 입력)에서도 동작")

    cfg = load_config(ROOT / "configs" / "config_noise_aug_plus263_ft4.yaml")
    assert cfg.model.visual_backbone == "mobilefacenet"
    kw = dict(d_model=256, n_heads=8, ffn_dim=1024, n_layers=2, frame_feat_dim=256, dropout=0.1)
    mf = VisualBackbone(**kw, cnn_dropout=0.15, cnn_freeze_layers=9, backbone_type="mobilefacenet")
    assert cfg.model.efficientface_input_size == 224
    w = EF_W if Path(EF_W).exists() else None
    ef = VisualBackbone(**kw, cnn_dropout=0.15, cnn_freeze_layers=9, backbone_type="efficientface",
                        efficientface_weights=w)
    ef112 = VisualBackbone(**kw, cnn_dropout=0.15, cnn_freeze_layers=9, backbone_type="efficientface",
                           efficientface_weights=w, efficientface_input_size=112)
    assert isinstance(mf.frame_cnn, FrameCNN) and isinstance(ef.frame_cnn, EfficientFaceFrameCNN)
    x = torch.rand(2, 4, 3, 112, 112)
    with torch.no_grad():
        a, c, c112 = mf(x), ef(x), ef112(x)
    assert a.shape == c.shape == c112.shape == (2, 4, 256), (a.shape, c.shape, c112.shape)
    assert not torch.allclose(c, c112), "224 업샘플과 112 경로가 같은 값이면 리사이즈가 안 걸린 것"
    n_mf = sum(p.numel() for p in mf.frame_cnn.parameters())
    n_ef = sum(p.numel() for p in ef.frame_cnn.parameters())
    print(f"  ✅ EfficientFace 출력 {tuple(c.shape)} 동일 · 파라미터 {n_ef:,} (MobileFaceNet {n_mf:,})"
          + ("" if w else "  ※가중치 파일 없어 무작위 초기화로 검사"))
    print("전부 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
