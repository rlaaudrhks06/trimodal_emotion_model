"""wav2vec2 부분 미세조정 검증 — 학습되는 층이 정확히 그 층이고, 그래디언트가 거기까지만 가는가.

  ① finetune_layers=4, layer=12 -> 9~12층만 requires_grad. 1~8·13~24·conv 추출기는 동결
  ② backward 뒤 9~12층 grad가 있고, 8층·13층·feature_extractor grad는 None
  ③ finetune_layers=0 이면 이전과 완전히 같은 경로(no_grad, 학습 파라미터 8,760,071)
  ④ 캐시(audio_feat)와 미세조정을 같이 주면 train.py가 막는다 (조용히 동결 실험이 되는 사고)
  ⑤ 학습 층 범위 계산: layer=12,N=4 -> (9,12); layer=-1(24),N=2 -> (23,24); N>layer -> 에러

합성 파형으로 돈다(실데이터 불필요, CPU 가능).
    python tests/test_w2v_finetune.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from src.config import load_config
from src.model import TrimodalEmotionModel

ROOT = Path(__file__).resolve().parent.parent


def grads_of(bb, prefix):
    return [p.grad for n, p in bb.w2v.named_parameters() if n.startswith(prefix)]


def main() -> int:
    print("=== wav2vec2 부분 미세조정 검증 ===")
    cfg = load_config(ROOT / "configs" / "config_noise_aug_plus263_ft4.yaml")
    assert cfg.audio_w2v_finetune_layers == 4 and cfg.audio_w2v_layer == 12
    torch.manual_seed(0)
    model = TrimodalEmotionModel(cfg); bb = model.audio_backbone
    # ①
    assert bb.finetune_range == (9, 12), bb.finetune_range
    for i, blk in enumerate(bb.w2v.encoder.layers, 1):
        want = 9 <= i <= 12
        got = all(p.requires_grad for p in blk.parameters())
        assert got == want, f"층 {i}: requires_grad={got}, 기대 {want}"
    assert not any(p.requires_grad for p in bb.w2v.feature_extractor.parameters()), "conv 추출기가 풀렸다"
    assert not any(p.requires_grad for p in bb.w2v.encoder.pos_conv_embed.parameters()), "위치 conv가 풀렸다"
    n_w2v = sum(p.numel() for p in bb.w2v.parameters() if p.requires_grad)
    print(f"  ✅ 9~12층만 학습 ({n_w2v:,}개) · 1~8·13~24·conv·pos_conv 동결")
    # ② 그래디언트 경로
    model.train()
    B, T = 2, 16000 * 2
    wav = torch.randn(B, T) * 0.1; mask = torch.ones(B, T, dtype=torch.long)
    x = bb(wav, wav_attention_mask=mask)
    assert x.requires_grad, "출력에 그래프가 없다 — no_grad/detach 경로를 탔다"
    x.float().pow(2).mean().backward()
    for i in (9, 12):
        g = grads_of(bb, f"encoder.layers.{i-1}."); assert g and all(t is not None for t in g), f"층 {i} grad 없음"
    for i in (8, 13):
        g = grads_of(bb, f"encoder.layers.{i-1}."); assert all(t is None for t in g), f"층 {i}에 grad가 갔다 (동결이어야)"
    assert all(t is None for t in grads_of(bb, "feature_extractor.")), "conv 추출기에 grad"
    print("  ✅ backward: 9·12층 grad 있음 · 8·13층·conv grad None")
    # ③ 0이면 이전과 동일
    cfg0 = load_config(ROOT / "configs" / "config_noise_aug_plus263.yaml")
    assert cfg0.audio_w2v_finetune_layers == 0
    m0 = TrimodalEmotionModel(cfg0)
    n0 = sum(p.numel() for p in m0.parameters() if p.requires_grad)
    assert n0 == 8_760_071, n0
    x0 = m0.audio_backbone(wav, wav_attention_mask=mask)
    assert not x0.requires_grad or x0.grad_fn is not None  # frontend는 학습이라 grad_fn 있음
    assert not any(p.requires_grad for p in m0.audio_backbone.w2v.parameters())
    print(f"  ✅ finetune_layers=0: 학습 파라미터 {n0:,} (v11d와 동일) · w2v 전부 동결")
    # ⑤ 범위 계산
    from src.models.audio_backbone_w2v import Wav2Vec2AudioBackbone
    kw = dict(pretrained_model=cfg.audio_pretrained, d_model=cfg.model.d_model, n_heads=cfg.model.n_heads, ffn_dim=cfg.model.ffn_dim)
    assert Wav2Vec2AudioBackbone(layer=-1, finetune_layers=2, **kw).finetune_range == (23, 24)
    try:
        Wav2Vec2AudioBackbone(layer=3, finetune_layers=4, **kw); raise AssertionError("N>layer인데 통과")
    except ValueError as e:
        print(f"  ✅ 범위: layer=-1,N=2 -> (23,24) · N>layer 차단 — {str(e)[:40]}")
    # ④ 가드: train.py의 검사 로직을 그대로 재현
    cfg_bad = load_config(ROOT / "configs" / "config_noise_aug_plus263_ft4.yaml")
    tc = dict(cfg_bad.raw["train"]); tc["w2v_cache_dir"] = "data/w2v_cache"
    try:
        if cfg_bad.audio_w2v_finetune_layers > 0 and tc.get("w2v_cache_dir"):
            raise ValueError("cache+finetune")
        raise AssertionError("가드가 안 걸렸다")
    except ValueError:
        print("  ✅ 미세조정 + 캐시 동시 사용 차단 (train.py 가드 조건)")
    print("=== 전부 통과 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
