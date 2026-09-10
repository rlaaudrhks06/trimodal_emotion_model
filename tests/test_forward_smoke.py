"""합성(무작위) 텐서로 전체 파이프라인의 차원 정합을 검증하는 스모크 테스트.

실제 KEMDy19/20·AI Hub 데이터가 아직 없는 상태에서, 설계 v3의 표기법(§3)에
정의된 T_v/T_a/T_t/d_model 형태가 끝까지 어긋나지 않고 [B, C] 로짓까지
나오는지 확인한다(C=config의 num_classes, src/datasets/labels.py 참고).
학습이 되는지(정확도)가 아니라 "배관이 새지 않는지"를 보는 것이 목적이다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config import load_config
from src.model import TrimodalEmotionModel


def test_forward_shapes():
    cfg = load_config()
    torch.manual_seed(0)

    batch_size = 2
    t_v, t_a, t_t = 12, 40, 8  # 서로 다른 길이로 둬서 시퀀스 길이 혼동이 없는지도 확인

    mel_spec = torch.randn(batch_size, t_a, cfg.audio_n_mels)
    prosody_vec = torch.randn(batch_size, cfg.model.prosody_dim)
    frames = torch.rand(batch_size, t_v, 3, cfg.visual_face_size, cfg.visual_face_size)
    input_ids = torch.randint(low=0, high=1000, size=(batch_size, t_t))
    attention_mask = torch.ones(batch_size, t_t, dtype=torch.long)
    attention_mask[1, -2:] = 0  # 두 번째 샘플은 뒤 2토큰이 패딩이라고 가정

    model = TrimodalEmotionModel(cfg, modality_dropout_prob=0.0)
    model.eval()

    with torch.no_grad():
        logits = model(
            mel_spec=mel_spec,
            prosody_vec=prosody_vec,
            frames=frames,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    assert logits.shape == (batch_size, cfg.model.num_classes), f"unexpected shape: {logits.shape}"

    probs = torch.softmax(logits, dim=-1)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(batch_size), atol=1e-5), f"softmax sums != 1: {sums}"

    print("[smoke test] logits shape:", tuple(logits.shape))
    print("[smoke test] softmax probs:\n", probs)
    print("[smoke test] PASSED")


def test_modality_dropout_runs():
    """학습 모드 + modality_dropout_prob>0 에서도 forward가 죽지 않는지 확인."""
    cfg = load_config()
    batch_size, t_v, t_a, t_t = 3, 10, 20, 6

    mel_spec = torch.randn(batch_size, t_a, cfg.audio_n_mels)
    prosody_vec = torch.randn(batch_size, cfg.model.prosody_dim)
    frames = torch.rand(batch_size, t_v, 3, cfg.visual_face_size, cfg.visual_face_size)
    input_ids = torch.randint(low=0, high=1000, size=(batch_size, t_t))
    attention_mask = torch.ones(batch_size, t_t, dtype=torch.long)

    model = TrimodalEmotionModel(cfg, modality_dropout_prob=0.5)
    model.train()

    logits = model(
        mel_spec=mel_spec, prosody_vec=prosody_vec, frames=frames,
        input_ids=input_ids, attention_mask=attention_mask,
    )
    assert logits.shape == (batch_size, cfg.model.num_classes)
    print("[smoke test] modality dropout forward PASSED")


def test_fusion_order():
    """계층 순서 애블레이션(11.3.2 항목 1)이 실제로 작동하는지.

    융합 모듈만 직접 세운다 — 바뀐 것이 거기뿐이고, 전체 모델을 세우면 백본 3종을
    매번 로딩해 느려진다. 기본 순서의 전 구간 동작은 위 두 테스트가 이미 덮는다.

    네 가지를 본다. 특히 두 번째가 중요하다 — **플래그가 아무것도 안 바꾸는데
    통과하는 상태**가 이 프로젝트에서 반복된 실패 유형이다(모달리티 드롭아웃이
    wav2vec2 경로에서 무효였던 것, 8.18.4절).
    """
    from src.fusion.hierarchical_fusion import (
        FUSION_ORDERS, HierarchicalCrossAttentionFusion, _LEGACY_KEY_MAP, mean_pool,
    )

    torch.manual_seed(0)
    d, b, t_v, t_a, t_t = 32, 2, 5, 9, 4
    x_v, x_a, x_t = torch.randn(b, t_v, d), torch.randn(b, t_a, d), torch.randn(b, t_t, d)

    def build(order):
        m = HierarchicalCrossAttentionFusion(d_model=d, n_heads=4, ffn_dim=64, order=order)
        return m.eval()

    # ① 세 순서 전부 돌고 형태가 같다.
    outs = {}
    for order in FUSION_ORDERS:
        with torch.no_grad():
            z_v, z_a, z_t = build(order)(x_v, x_a, x_t)
        assert z_v.shape == z_a.shape == z_t.shape == (b, d), f"{order}: {z_v.shape}"
        outs[order] = (z_v, z_a, z_t)

    # ② **같은 가중치**로 순서만 바꾸면 결과가 달라져야 한다.
    #    이름이 역할 기반이라 state_dict 키가 순서와 무관하게 같아서 이 비교가 가능하다.
    base = build("audio_text")
    other = build("audio_visual")
    other.load_state_dict(base.state_dict())
    with torch.no_grad():
        zb = base(x_v, x_a, x_t)
        zo = other(x_v, x_a, x_t)
    assert not all(torch.allclose(p, q) for p, q in zip(zb, zo)), \
        "fusion_order를 바꿨는데 출력이 같다 — 플래그가 아무 일도 하지 않는다"

    # ③ 기본 순서가 v11의 계산과 같은가(회귀). 아래는 이름을 바꾸기 전 원본 forward를
    #    그대로 다시 쓴 참조 구현이다. 테스트가 기대값을 직접 적는 것은 허용된다 —
    #    구현을 다시 부르면 무엇도 검증하지 못한다.
    with torch.no_grad():
        c_a = base.ca_first_attends_second(x_a, x_t, kv_key_padding_mask=None)
        c_t = base.ca_second_attends_first(x_t, x_a, kv_key_padding_mask=None)
        at = torch.cat([c_a, c_t], dim=1)
        c_v = base.ca_third_attends_pair(x_v, at, kv_key_padding_mask=None)
        c_at = base.ca_pair_attends_third(at, x_v, kv_key_padding_mask=None)
        ref = (mean_pool(c_v, None), mean_pool(c_at[:, :t_a], None), mean_pool(c_at[:, t_a:], None))
    for got, want, name in zip(zb, ref, ("z_v", "z_a", "z_t")):
        assert torch.allclose(got, want, atol=1e-6), f"기본 순서가 v11 계산과 다르다: {name}"

    # ④ 이름 바꾸기 전(v1~v12b) 체크포인트의 키가 로드된다.
    legacy = {}
    for k, v in base.state_dict().items():
        role = k.split(".")[0]
        old = next(o for o, n in _LEGACY_KEY_MAP.items() if n == role)
        legacy[k.replace(role, old, 1)] = v
    fresh = build("audio_text")
    missing, unexpected = fresh.load_state_dict(legacy, strict=True)
    assert not missing and not unexpected, f"옛 키 로드 실패: {missing} / {unexpected}"
    with torch.no_grad():
        assert all(torch.allclose(p, q) for p, q in zip(fresh(x_v, x_a, x_t), zb)), \
            "옛 체크포인트를 로드했는데 출력이 다르다"

    print("[smoke test] fusion order (3순서·비무동작·v11 회귀·옛키 로드) PASSED")


if __name__ == "__main__":
    test_forward_shapes()
    test_modality_dropout_runs()
    test_fusion_order()
