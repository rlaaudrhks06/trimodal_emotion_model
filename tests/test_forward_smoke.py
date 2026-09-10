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


def test_self_fusion_baseline():
    """late fusion 베이스라인(11.3.2 항목 2)이 정말 모달 간 섞임이 없는지.

    ②가 이 테스트의 핵심이다. "self-attention이라 안 섞인다"는 주장인데, 배선을
    잘못하면 섞이면서도 형태는 맞아 조용히 통과한다. 주장을 그대로 재는 쪽을 택했다 —
    **다른 모달리티 입력을 통째로 바꿔도 내 출력이 한 비트도 안 변해야 한다.**
    """
    from src.fusion.hierarchical_fusion import (
        HierarchicalCrossAttentionFusion, SelfAttentionFusion, build_fusion,
    )

    torch.manual_seed(0)
    d, b, t_v, t_a, t_t = 32, 2, 5, 9, 4
    kw = dict(d_model=d, n_heads=4, ffn_dim=64)
    x_v, x_a, x_t = torch.randn(b, t_v, d), torch.randn(b, t_a, d), torch.randn(b, t_t, d)

    fusion = build_fusion("self", **kw).eval()
    assert isinstance(fusion, SelfAttentionFusion)

    # ① 형태가 계층 융합과 같다 — 분류기·운율 게이트를 손대지 않았다는 뜻.
    with torch.no_grad():
        z_v, z_a, z_t = fusion(x_v, x_a, x_t)
    assert z_v.shape == z_a.shape == z_t.shape == (b, d), f"{z_v.shape}"

    # ② 모달 간 섞임이 없다: 오디오·텍스트를 통째로 갈아도 z_v가 그대로여야 한다.
    with torch.no_grad():
        z_v2, _, _ = fusion(x_v, torch.randn(b, t_a, d), torch.randn(b, t_t, d))
    assert torch.equal(z_v, z_v2), "self 융합인데 다른 모달리티가 z_v를 바꾼다 — 섞이고 있다"

    # ②′ 대조: 계층 융합은 반드시 섞여야 한다(안 섞이면 그쪽이 고장난 것이다).
    hier = build_fusion("hierarchical", **kw).eval()
    with torch.no_grad():
        h1, _, _ = hier(x_v, x_a, x_t)
        h2, _, _ = hier(x_v, torch.randn(b, t_a, d), torch.randn(b, t_t, d))
    assert not torch.allclose(h1, h2), "계층 융합인데 다른 모달리티가 z_v에 영향을 안 준다"

    # ③ 파라미터 차이가 문서에 적은 대로 블록 4개 -> 3개인가.
    n_self = sum(p.numel() for p in fusion.parameters())
    n_hier = sum(p.numel() for p in hier.parameters())
    assert n_hier == n_self * 4 // 3, f"블록 수 가정이 깨졌다: hier {n_hier} vs self {n_self}"

    # ④ self에 fusion_order를 주면 조용히 무시하지 말고 멈춰야 한다.
    try:
        build_fusion("self", order="audio_visual", **kw)
    except ValueError:
        pass
    else:
        raise AssertionError("self + fusion_order 조합이 조용히 통과했다")

    # ⑤ 패딩 위치가 결과에 섞이지 않는가 — 유효 구간만 넣은 것과 같아야 한다.
    short = torch.randn(1, 3, d)
    padded = torch.cat([short, torch.randn(1, 4, d)], dim=1)
    mask = torch.zeros(1, 7, dtype=torch.bool)
    mask[0, 3:] = True  # True=패딩
    with torch.no_grad():
        _, _, z_pad = fusion(x_v[:1], x_a[:1], padded, t_mask=mask)
        _, _, z_ref = fusion(x_v[:1], x_a[:1], short)
    assert torch.allclose(z_pad, z_ref, atol=1e-6), "패딩이 결과에 새어 들어간다"

    print(f"[smoke test] self fusion 베이스라인 PASSED "
          f"(융합 파라미터 hier {n_hier:,} -> self {n_self:,}, {n_self / n_hier - 1:+.1%})")


def test_prosody_fusion_none():
    """운율 게이트 애블레이션(11.3.2 항목 3)이 운율 경로를 정말 끊는지.

    "안 쓴다"를 모듈 유무로만 확인하면 부족하다 — 어딘가에서 여전히 읽고 있을 수 있다.
    **운율 벡터를 통째로 갈아도 로짓이 한 비트도 안 변해야** 경로가 끊긴 것이다.
    """
    cfg = load_config()
    b, t_v, t_a, t_t = 2, 6, 16, 5
    torch.manual_seed(0)
    common = dict(
        mel_spec=torch.randn(b, t_a, cfg.audio_n_mels),
        frames=torch.rand(b, t_v, 3, cfg.visual_face_size, cfg.visual_face_size),
        input_ids=torch.randint(0, 1000, (b, t_t)),
        attention_mask=torch.ones(b, t_t, dtype=torch.long),
    )
    p1 = torch.randn(b, cfg.model.prosody_dim)
    p2 = torch.randn(b, cfg.model.prosody_dim)

    gated = TrimodalEmotionModel(cfg).eval()
    cfg.model.prosody_fusion = "none"
    plain = TrimodalEmotionModel(cfg).eval()
    cfg.model.prosody_fusion = "gate"  # 다른 테스트에 새지 않게 되돌린다

    assert hasattr(gated, "prosody_gate")
    assert not hasattr(plain, "prosody_gate"), "none인데 게이트 모듈이 남아 있다"

    # 파라미터 차이가 게이트 크기와 정확히 같은가 (dims에서 유도 — 상수를 박지 않는다)
    hybrid, pdim = cfg.model.d_model * 2, cfg.model.prosody_dim
    expect = (pdim * hybrid + hybrid) + ((hybrid + pdim) * hybrid + hybrid)
    delta = (sum(p.numel() for p in gated.parameters())
             - sum(p.numel() for p in plain.parameters()))
    assert delta == expect, f"게이트 파라미터 {delta:,} != 기대 {expect:,}"

    with torch.no_grad():
        a = plain(prosody_vec=p1, **common)
        c = plain(prosody_vec=p2, **common)
        g1 = gated(prosody_vec=p1, **common)
        g2 = gated(prosody_vec=p2, **common)
    assert torch.equal(a, c), "prosody_fusion='none'인데 운율이 로짓을 바꾼다 — 경로가 남아 있다"
    assert not torch.allclose(g1, g2), "게이트가 켜져 있는데 운율이 로짓에 영향을 안 준다"

    print(f"[smoke test] prosody_fusion='none' PASSED (게이트 파라미터 {expect:,}개 제거)")


if __name__ == "__main__":
    test_forward_shapes()
    test_modality_dropout_runs()
    test_fusion_order()
    test_self_fusion_baseline()
    test_prosody_fusion_none()
