"""설계 v3 §5.1: 2단계 계층적 교차 어텐션 (총 4개 블록, 6개가 아님).

1단계: 두 모달리티를 먼저 양방향으로 결합.
    CA1: Context_1 = CrossAttn(Q=X_1, KV=X_2)
    CA2: Context_2 = CrossAttn(Q=X_2, KV=X_1)

2단계: 1단계 결과를 시간축으로 이어붙인 pair_seq와 나머지 한 모달리티를 결합.
    pair_seq = concat([Context_1, Context_2], dim=시간축)
    CA3: Context_3    = CrossAttn(Q=X_3,      KV=pair_seq)
    CA4: Context_pair = CrossAttn(Q=pair_seq, KV=X_3)
         -> 다시 1번 구간/2번 구간으로 분리해 각각 풀링

이렇게 하면 최종 분류기(§5.3)에 필요한 z_cross_v / z_cross_a / z_cross_t
세 벡터를 모두 얻으면서, 블록 수는 3쌍×양방향(6개)이 아닌 4개로 줄어든다.

---

## 어느 두 모달리티를 1단계에 넣을 것인가 — `order`

v1~v12b는 전부 **오디오+텍스트가 1단계**였다("무슨 말을 어떤 억양으로" 다음 "표정이
그걸 뒷받침하나"). 이 순서는 설계 v3에서 계산량을 아끼려고 고른 것이지 측정으로
정한 것이 아니다.

그런데 8.30.1절 애블레이션에서 **텍스트를 빼면 가장 크게 떨어졌다**(−6.59%p,
오디오 −3.69 / 영상 −3.32). 텍스트가 정보를 가장 많이 가져서라고 보기 어려운 것이,
8.30.2절에서 글자에 감정이 없는 짧은 발화("왜?", "네.")의 정확도가 오히려 전체보다
2.3%p 높았기 때문이다. 모델은 텍스트의 *의미*에 기대고 있지 않다.

남은 설명이 **구조**다. 텍스트가 1단계에 박혀 있어 앵커 역할을 하므로, 텍스트를
지우면 1단계가 붕괴하고 그 손상이 2단계까지 전파된다 — 즉 애블레이션이 잰 것은
"텍스트가 가진 정보량"이 아니라 "텍스트를 앵커로 박아둔 구조가 무너진 손해"일 수 있다.

`order`를 바꿔 학습하면 이 둘이 갈린다. 텍스트를 1단계에서 빼고도(`audio_visual`)
텍스트 애블레이션 손실이 그대로면 **정보**이고, 줄어들면 **구조**다. 통합기록
11.3.2가 재학습 항목 1번으로 지목한 실험이며, 답에 따라 융합 설계 전체를 다시 본다.
"""
import torch
import torch.nn as nn

from .cross_attention import StackedCrossAttention


def mean_pool(x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
    """[B, T, d_model] -> [B, d_model]. 패딩 위치는 평균에서 제외."""
    if key_padding_mask is None:
        return x.mean(dim=1)
    valid = (~key_padding_mask).unsqueeze(-1).float()  # [B, T, 1], True=패딩이므로 반전
    summed = (x * valid).sum(dim=1)
    count = valid.sum(dim=1).clamp(min=1.0)
    return summed / count


# 값은 (1단계 첫째, 1단계 둘째, 2단계에 붙는 나머지)다.
# "audio_text"가 v1~v12b 전체가 쓴 순서이므로 기본값이다.
FUSION_ORDERS: dict[str, tuple[str, str, str]] = {
    "audio_text": ("a", "t", "v"),    # v11 기본 — 오디오↔텍스트 먼저, 그 다음 영상
    "audio_visual": ("a", "v", "t"),  # 텍스트를 1단계에서 뺀다 (11.3.2 항목 1의 본 실험)
    "visual_text": ("v", "t", "a"),   # 오디오를 1단계에서 뺀다 (대조군)
}

# 역할 기반으로 이름을 바꾸기 전(v1~v12b)의 파라미터 키. 그때는 순서가 항상
# audio_text였으므로 이름이 곧 역할이었다.
_LEGACY_KEY_MAP = {
    "ca_audio_attends_text": "ca_first_attends_second",
    "ca_text_attends_audio": "ca_second_attends_first",
    "ca_visual_attends_at": "ca_third_attends_pair",
    "ca_at_attends_visual": "ca_pair_attends_third",
}


class HierarchicalCrossAttentionFusion(nn.Module):
    """`order`에 따라 1단계에 들어가는 두 모달리티가 달라진다.

    **모듈 이름이 모달리티가 아니라 역할이다.** `ca_first_attends_second`는
    order가 "audio_text"면 오디오가 텍스트를 참조하는 블록이고, "audio_visual"이면
    오디오가 영상을 참조하는 블록이다. 이름에 모달리티를 박아두면 order를 바꾸는
    순간 이름이 거짓말을 하게 되는데, 이 프로젝트는 `face_frames_dir`이라는 이름이
    세 절에 걸쳐 사람을 속인 적이 있다(8.8절). 그래서 역할로 부른다.

    파라미터 형태는 order와 무관하게 같다(전부 d_model). 따라서 v11 체크포인트를
    다른 order 모델에 넣는 것이 **에러 없이** 된다 — 하지만 그 가중치는 audio_text
    순서로 학습된 것이라 의미가 다르다. 아래 로드 훅이 그 경우 경고한다.
    """

    def __init__(
        self, d_model: int, n_heads: int, ffn_dim: int, n_layers: int = 1,
        dropout: float = 0.1, drop_path: float = 0.0, order: str = "audio_text",
    ):
        super().__init__()
        if order not in FUSION_ORDERS:
            raise ValueError(
                f"fusion_order는 {list(FUSION_ORDERS)} 중 하나여야 한다, got {order!r}"
            )
        self.order = order
        self.roles = FUSION_ORDERS[order]

        def block() -> StackedCrossAttention:
            return StackedCrossAttention(d_model, n_heads, ffn_dim, n_layers, dropout, drop_path)

        self.ca_first_attends_second = block()
        self.ca_second_attends_first = block()
        self.ca_third_attends_pair = block()
        self.ca_pair_attends_third = block()

        self.register_load_state_dict_pre_hook(_remap_legacy_keys)

    def forward(
        self,
        x_v: torch.Tensor,
        x_a: torch.Tensor,
        x_t: torch.Tensor,
        v_mask: torch.Tensor | None = None,
        a_mask: torch.Tensor | None = None,
        t_mask: torch.Tensor | None = None,
    ):
        # 호출부는 항상 (v, a, t)로 준다. 여기서 order에 맞춰 역할로 재배치한다 —
        # 호출부가 순서를 알 필요가 없어야 order를 바꿔도 model.py가 그대로다.
        seqs = {"v": x_v, "a": x_a, "t": x_t}
        masks = {"v": v_mask, "a": a_mask, "t": t_mask}
        k1, k2, k3 = self.roles
        x1, x2, x3 = seqs[k1], seqs[k2], seqs[k3]
        m1, m2, m3 = masks[k1], masks[k2], masks[k3]

        t_1 = x1.size(1)

        # --- 1단계: 1번 <-> 2번 ---
        context_1 = self.ca_first_attends_second(x1, x2, kv_key_padding_mask=m2)  # [B, T_1, d]
        context_2 = self.ca_second_attends_first(x2, x1, kv_key_padding_mask=m1)  # [B, T_2, d]

        pair_seq = torch.cat([context_1, context_2], dim=1)  # [B, T_1+T_2, d]
        if m1 is not None or m2 is not None:
            m1_ = m1 if m1 is not None else torch.zeros_like(context_1[..., 0], dtype=torch.bool)
            m2_ = m2 if m2 is not None else torch.zeros_like(context_2[..., 0], dtype=torch.bool)
            pair_mask = torch.cat([m1_, m2_], dim=1)
        else:
            pair_mask = None

        # --- 2단계: 3번 <-> (1번+2번) ---
        context_3 = self.ca_third_attends_pair(x3, pair_seq, kv_key_padding_mask=pair_mask)   # [B, T_3, d]
        context_pair = self.ca_pair_attends_third(pair_seq, x3, kv_key_padding_mask=m3)       # [B, T_1+T_2, d]

        z = {
            k1: mean_pool(context_pair[:, :t_1, :], m1),
            k2: mean_pool(context_pair[:, t_1:, :], m2),
            k3: mean_pool(context_3, m3),
        }
        # 반환은 다시 (v, a, t) 고정 — 분류기 입력 순서가 order에 따라 흔들리면 안 된다.
        return z["v"], z["a"], z["t"]


def _remap_legacy_keys(module, state_dict, prefix, local_metadata, strict,
                       missing_keys, unexpected_keys, error_msgs):
    """모듈 이름을 역할 기반으로 바꾸기 전(v1~v12b) 체크포인트의 키를 흡수한다.

    훅으로 하는 이유: `load_state_dict`를 부르는 곳이 train·evaluate·engine·
    export·benchmark 다섯 군데인데, 각자 고치면 반드시 한 곳이 빠진다. 모듈에
    한 번 걸어두면 부르는 쪽은 아무것도 몰라도 된다.
    """
    renamed = False
    for old, new in _LEGACY_KEY_MAP.items():
        old_prefix = f"{prefix}{old}."
        for key in [k for k in state_dict if k.startswith(old_prefix)]:
            state_dict[f"{prefix}{new}.{key[len(old_prefix):]}"] = state_dict.pop(key)
            renamed = True

    if renamed and module.order != "audio_text":
        # 조용히 틀리는 자리다 — 형태가 같아서 로딩은 성공하지만 그 가중치는
        # audio_text 순서로 학습된 것이라 다른 순서에서는 의미가 다르다.
        print(f"[fusion] ⚠️  옛 체크포인트(audio_text 순서로 학습)를 "
              f"order={module.order} 모델에 로드했다. 처음부터 재학습할 것이 아니라면 "
              f"이 가중치는 의미가 맞지 않는다.")
