"""영상 백본 선택 검증 (13.18절).

  ① 기본값은 mobilefacenet — 설정을 안 바꾸면 v11e와 완전히 같은 모델
  ② scratch를 고르면 ScratchFrameCNN이 들어가고 **전부 학습 대상**이다
  ③ 두 백본의 출력 모양이 같아 융합 이후 구조가 그대로다
  ④ 잘못된 값은 거부한다
  ⑤ scratch는 사전학습 가중치를 안 쓰므로 파라미터가 훨씬 적다

    python tests/test_visual_scratch.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from src.config import load_config
from src.models.visual_backbone import VisualBackbone, FrameCNN, ScratchFrameCNN

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    print("=== 영상 백본 선택 ===")
    cfg = load_config(ROOT / "configs" / "config_noise_aug_plus263_ft4.yaml")
    assert cfg.model.visual_backbone == "mobilefacenet", cfg.model.visual_backbone
    print("  ✅ 기본값 mobilefacenet (기존 설정은 그대로 동작)")

    kw = dict(d_model=256, n_heads=8, ffn_dim=1024, n_layers=2, frame_feat_dim=256, dropout=0.1)
    mf = VisualBackbone(**kw, cnn_dropout=0.15, cnn_freeze_layers=9, backbone_type="mobilefacenet")
    sc = VisualBackbone(**kw, cnn_dropout=0.15, cnn_freeze_layers=9, backbone_type="scratch")
    assert isinstance(mf.frame_cnn, FrameCNN) and isinstance(sc.frame_cnn, ScratchFrameCNN)
    n_mf_train = sum(p.numel() for p in mf.frame_cnn.parameters() if p.requires_grad)
    n_sc_train = sum(p.numel() for p in sc.frame_cnn.parameters() if p.requires_grad)
    assert n_sc_train == sum(p.numel() for p in sc.frame_cnn.parameters()), "scratch는 전부 학습해야 한다"
    print(f"  ✅ scratch는 전부 학습 ({n_sc_train:,}개) · mobilefacenet은 proj만 ({n_mf_train:,}개)")

    x = torch.rand(2, 4, 3, 112, 112)
    with torch.no_grad():
        a, b = mf(x), sc(x)
    assert a.shape == b.shape == (2, 4, 256), (a.shape, b.shape)
    print(f"  ✅ 출력 모양 동일 {tuple(a.shape)} — 융합 이후 구조 불변")

    try:
        VisualBackbone(**kw, backbone_type="resnet")
        raise AssertionError("잘못된 값인데 통과했다")
    except ValueError as e:
        print(f"  ✅ 잘못된 값 거부: {str(e)[:45]}…")

    n_mf = sum(p.numel() for p in mf.frame_cnn.parameters())
    assert n_sc_train < n_mf, (n_sc_train, n_mf)
    print(f"  ✅ 파라미터: scratch {n_sc_train:,} < mobilefacenet {n_mf:,}")
    print("전부 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
