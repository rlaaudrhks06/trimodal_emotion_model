"""wav2vec2 기반 오디오 백본 — 멜스펙트로그램+자체학습 프론트엔드의 대안.

배경(11.0.1절): 세 모달리티 중 오디오만 사전학습 없이 처음부터 학습하고 있었고,
단일 모달리티 성능도 가장 낮았다(33.14%, 8.7절). 화자 독립 조건(v10)에서는 화자
개인화 단서에 기댈 수 없으므로 각 모달리티 자체의 표현력이 더 중요해졌다.

**모델 선택 근거 — ASR 파인튜닝본을 쓰면 안 된다**
"Emotion Recognition from Speech Using Wav2vec 2.0 Embeddings"(arXiv:2104.03502)는
ASR로 파인튜닝한 wav2vec2가 자기지도 학습만 한 모델보다 SER 성능이 **나빴다**고
보고한다 — ASR은 "누가 어떤 감정으로 말했든 같은 글자"를 뽑는 것이 목표라 감정·화자
정보를 의도적으로 지우도록 학습되기 때문이다. 따라서 한국어 ASR 파인튜닝 모델
(kresnik/wav2vec2-large-xlsr-korean 등)이 아니라 **SSL만 거친 다국어 모델**
(facebook/wav2vec2-large-xlsr-53, 한국어 포함)을 기본값으로 쓴다.

**층 선택 — 마지막 층이 아니다**
같은 계열 연구들이 공통적으로 "SER에서는 중간 층 표현이 더 낫다"고 보고한다.
`layer` 인자로 어느 층을 뽑을지 지정한다(0=임베딩 출력, 1~24=각 트랜스포머 층).
음수 인덱스도 지원한다(-1=마지막 층). 기본값 12는 24층의 중간이다.

**동결 기본**: v7(BERT 완전 동결)에서 확인했듯 이 데이터 규모에서 대형 사전학습
모델을 파인튜닝하면 과적합이 커진다. 3.17억 파라미터인 wav2vec2는 더욱 그렇다.
freeze=True(기본)면 특징 추출기로만 쓰고, forward도 no_grad로 감싸 메모리·속도를
아낀다.
"""
import torch
import torch.nn as nn
from transformers import AutoConfig, Wav2Vec2Model

from .common import TemporalConvFrontend


class Wav2Vec2AudioBackbone(nn.Module):
    """원본 파형 [B, T_samples] -> X_a [B, T_a, d_model].

    wav2vec2는 16kHz 파형을 받아 20ms(50fps) 단위 표현을 낸다 — 멜스펙트로그램
    경로(10ms hop, 100fps)보다 시퀀스가 절반이라 뒤쪽 교차 어텐션 비용도 줄어든다.
    """

    def __init__(
        self, pretrained_model: str, d_model: int, n_heads: int, ffn_dim: int,
        n_layers: int = 2, layer: int = 12, dropout: float = 0.1, freeze: bool = True,
        finetune_layers: int = 0,
    ):
        super().__init__()
        self.w2v = Wav2Vec2Model.from_pretrained(pretrained_model)
        w2v_cfg = AutoConfig.from_pretrained(pretrained_model)
        n_w2v_layers = w2v_cfg.num_hidden_layers

        # hidden_states는 (임베딩 출력, layer1, ..., layerN) 총 N+1개다.
        if not (-(n_w2v_layers + 1) <= layer <= n_w2v_layers):
            raise ValueError(
                f"layer는 -{n_w2v_layers + 1}~{n_w2v_layers} 범위여야 함 "
                f"({pretrained_model}은 {n_w2v_layers}층), got {layer}"
            )
        self.layer = layer
        self.freeze = freeze
        if freeze:
            self.w2v.eval()
            for p in self.w2v.parameters():
                p.requires_grad = False

        # ── 부분 미세조정(13.13절): 꺼내는 층 바로 아래 N개 트랜스포머 층만 학습한다.
        #
        # "상위 N층"이 아니다. hidden_states[layer]를 쓰므로 layer보다 위의 층은 출력에
        # 영향이 없다 — 21~24층을 풀면 아무것도 안 배운다. layer=12, N=4면 9~12층이다.
        #
        # v1~v11d까지 전부 동결이었고(2.5M~8.7M만 학습), 프로브(13.11절)는 오디오 표현에
        # 여지가 있다고 했는데 층 바꾸기·백본 교체로는 못 꺼냈다. 학습으로 꺼내는 것이
        # 남은 레버다. v7(BERT 전체 미세조정)이 과적합했으므로 N을 작게, lr을 낮게 둔다.
        #
        # w2v는 여전히 eval 모드로 둔다 — LayerDrop(0.1)·드롭아웃이 켜지면 "동결 대비
        # 무엇이 달라졌나"에 변수가 둘이 된다. 가중치만 움직인다.
        self.finetune_layers = int(finetune_layers)
        if self.finetune_layers > 0:
            if not freeze:
                raise ValueError("finetune_layers는 freeze=True(부분 동결)와 함께 쓴다 — freeze=False는 전체 미세조정")
            top = layer if layer > 0 else n_w2v_layers + 1 + layer   # 음수 인덱스 정규화
            lo = top - self.finetune_layers
            if lo < 0:
                raise ValueError(f"finetune_layers={self.finetune_layers}가 꺼내는 층({top})보다 많다")
            for blk in self.w2v.encoder.layers[lo:top]:
                for p in blk.parameters():
                    p.requires_grad = True
            self.finetune_range = (lo + 1, top)   # 사람이 읽는 1-based 층 번호
        else:
            self.finetune_range = None

        # wav2vec2의 hidden(예: 1024)을 d_model로 맞춘 뒤, 기존 경로와 동일하게
        # TemporalConvFrontend를 태운다 — 위치 인코딩·시간 컨텍스트 처리를 재사용.
        self.proj = nn.Linear(w2v_cfg.hidden_size, d_model)
        self.frontend = TemporalConvFrontend(
            in_dim=d_model, d_model=d_model, n_heads=n_heads,
            ffn_dim=ffn_dim, n_layers=n_layers, dropout=dropout,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            # 동결 시엔 상위 모델이 train()이어도 w2v는 eval 유지 — 내부 dropout/
            # LayerNorm 통계가 흔들리지 않게 한다(visual_backbone의 FrameCNN과 같은 처리).
            self.w2v.eval()
        return self

    @staticmethod
    def _normalize(waveform: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        """XLSR-53이 요구하는 zero-mean/unit-variance 정규화(do_normalize=True).

        HuggingFace의 Wav2Vec2FeatureExtractor가 하는 일을 여기서 직접 한다 —
        전처리기를 따로 두면 배치·패딩 처리가 이중이 되기 때문이다.

        **반드시 패딩을 제외한 유효 구간에서만 통계를 낸다.** 0으로 채운 부분까지
        평균/분산에 넣으면 짧은 발화일수록 통계가 0쪽으로 끌려가, 같은 발화라도
        배치에 어떤 길이가 같이 담기느냐에 따라 입력이 달라진다.
        """
        if attention_mask is None:
            mask = torch.ones_like(waveform)
        else:
            mask = attention_mask.to(waveform.dtype)
        n = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mean = (waveform * mask).sum(dim=-1, keepdim=True) / n
        var = (((waveform - mean) * mask) ** 2).sum(dim=-1, keepdim=True) / n
        normed = (waveform - mean) / torch.sqrt(var + 1e-7)
        return normed * mask  # 패딩 자리는 다시 0으로

    def _extract(self, waveform: torch.Tensor, attention_mask: torch.Tensor | None):
        waveform = self._normalize(waveform, attention_mask)
        out = self.w2v(waveform, attention_mask=attention_mask, output_hidden_states=True)
        return out.hidden_states[self.layer]  # [B, T_a, hidden]

    # ── 캐시 경로 (13.6절) ──────────────────────────────────────────────────
    #
    # 학습 한 스텝의 84%가 위 _extract다(실측 2.05s/2.43s). 동결이라 같은 파형이면
    # 40에폭 내내 같은 값을 내는데 매번 다시 계산했다. 멜·운율·얼굴을 캐시한 것과 같은
    # 논리로 이 출력을 미리 뽑아두면 18분/에폭 -> 약 3분이 된다.
    #
    # 경계를 여기(hidden_states[layer])로 잡는 이유: 이 아래 proj·frontend가 학습
    # 파라미터의 전부다. 캐시는 동결 부분만 담고, 학습되는 부분은 매 스텝 돈다.
    #
    # [실측] 캐시(발화 하나씩 추출)와 배치 경로는 값이 완전히 같지 않다 — 최대 2.3,
    # 평균 1e-3. 원인은 cudnn TF32(끄면 100배 줄어 최대 0.015). 배치 길이가 바뀌면
    # 값이 바뀌므로 v11 학습도 매 에폭 셔플마다 이 잡음을 겪어왔다. v11 체크포인트로
    # 판정하면 test 1,024발화 중 1건 예측이 바뀐다(99.90%). 그래서 수용한다.

    def extract_features(self, waveform: torch.Tensor, wav_attention_mask: torch.Tensor | None) -> torch.Tensor:
        """캐시 채우기용 공개 진입점. forward가 쓰는 _extract와 **같은 함수**를 부른다 —
        따로 구현하면 정규화·층 선택이 한쪽만 바뀌는 날이 온다."""
        with torch.no_grad():
            return self._extract(waveform, wav_attention_mask)

    # ── 모달리티 드롭아웃용 0-특징 표 ────────────────────────────────────────
    #
    # 드롭아웃은 파형을 0으로 만들고 마스크는 그대로 둔다(model.py). 그러면 wav2vec2는
    # "길이 T의 0 입력"을 받는데, 그 출력은 **T에 따라 다르다**(위치 conv가 경계
    # 64프레임을 다르게 봄 — 실측 가장자리 차이 26.8). 캐시 경로에서 v11과 같은 동작을
    # 재현하려면 프레임 길이별 출력을 표로 들고 있다가 드롭 시 대입해야 한다. 특징을
    # 그냥 0으로 하면 v11과 다른 입력이 된다.
    #
    # 체크포인트에 넣지 않는다(persistent=False) — 399×399×1024 fp16 = 325MB가 매
    # 체크포인트에 실리면 안 된다. 학습 시작 때 캐시 디렉터리에서 올린다.

    def build_zero_table(self, max_frames: int, device: torch.device,
                         frame_counts=None) -> torch.Tensor:
        """[max_frames+1, max_frames, hidden] fp16. 행 T = 프레임 T개짜리 0 파형의 출력.

        frame_counts를 주면 그 길이들만 채운다 — 테스트가 CPU에서 399번 forward를
        안 돌게 하기 위한 것이고, 실제 캐시는 전부 채운다.
        """
        table = torch.zeros(max_frames + 1, max_frames, self.w2v.config.hidden_size,
                            dtype=torch.float16, device=device)
        for T in (range(1, max_frames + 1) if frame_counts is None else sorted(set(frame_counts))):
            # 프레임 T개를 내는 최소 샘플 수. 전체 conv가 kernel 400 / stride 320이다.
            n = 320 * (T - 1) + 400
            assert int(self.output_lengths(torch.tensor([n]))[0]) == T, (T, n)
            z = torch.zeros(1, n, device=device)
            m = torch.ones(1, n, dtype=torch.long, device=device)
            table[T, :T] = self.extract_features(z, m)[0].half()
        return table

    def set_zero_table(self, table: torch.Tensor | None) -> None:
        self._zero_table = table

    def zero_feature(self, n_frames: int) -> torch.Tensor:
        """프레임 n개짜리 0 파형이 냈을 wav2vec2 출력 [n, hidden]."""
        t = getattr(self, "_zero_table", None)
        if t is None:
            raise RuntimeError(
                "0-특징 표가 없다 — 캐시 경로에서 모달리티 드롭아웃을 쓰려면 "
                "set_zero_table()로 올려야 한다 (scripts/precompute_w2v_cache.py가 만든다)"
            )
        if n_frames > t.size(1):
            raise ValueError(f"프레임 {n_frames}개는 표 범위({t.size(1)})를 넘는다")
        return t[n_frames, :n_frames]

    def forward_cached(self, h: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        """캐시된 wav2vec2 출력 [B, T_a, hidden] -> X_a [B, T_a, d_model].

        forward의 뒷부분(proj -> frontend)과 **한 글자도 다르지 않아야** 한다. 그래서
        forward도 이 함수를 부르게 했다 — 사본이 둘이면 한쪽만 고쳐진다.
        """
        return self.frontend(self.proj(h), key_padding_mask=key_padding_mask)

    def frame_padding_mask(self, wav_attention_mask: torch.Tensor, n_frames: int) -> torch.Tensor:
        """파형 마스크 [B, T_samples] -> 프레임 마스크 [B, T_a] (True=패딩).

        wav2vec2의 conv 스트라이드를 아는 건 이 백본뿐이므로 여기서 만든다.
        호출부가 멜 기준 마스크를 그대로 넘기면 길이가 안 맞아 조용히 틀린다.
        """
        lens = self.output_lengths(wav_attention_mask.sum(dim=-1))
        idx = torch.arange(n_frames, device=wav_attention_mask.device).unsqueeze(0)
        return idx >= lens.to(wav_attention_mask.device).unsqueeze(1)

    def forward(
        self,
        waveform: torch.Tensor,                       # [B, T_samples] 16kHz, -1~1
        wav_attention_mask: torch.Tensor | None = None,  # [B, T_samples] 1=유효
        key_padding_mask: torch.Tensor | None = None,    # [B, T_a] True=패딩. 생략 시 내부 계산
    ) -> torch.Tensor:
        if self.freeze and self.finetune_layers == 0:
            with torch.no_grad():
                h = self._extract(waveform, wav_attention_mask)
            h = h.detach()
        else:
            # 부분 미세조정: 동결된 하위 층은 파라미터·입력 모두 grad가 없어 autograd가
            # 그래프를 안 만든다 — 메모리는 학습 층 분만 든다. detach하면 안 된다.
            h = self._extract(waveform, wav_attention_mask)

        # 프론트엔드의 트랜스포머가 패딩 위치까지 어텐션하면 유효 구간 출력이 오염된다.
        # 마스크를 안 받았으면 여기서 직접 만들어 넘긴다 — 호출부가 잊어버려도 안전하게.
        if key_padding_mask is None and wav_attention_mask is not None:
            key_padding_mask = self.frame_padding_mask(wav_attention_mask, h.size(1))
        return self.forward_cached(h, key_padding_mask)

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """파형 샘플 수 -> wav2vec2 출력 프레임 수."""
        return self.w2v._get_feat_extract_output_lengths(input_lengths)
