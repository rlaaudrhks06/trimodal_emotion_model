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
        if self.freeze:
            with torch.no_grad():
                h = self._extract(waveform, wav_attention_mask)
            h = h.detach()
        else:
            h = self._extract(waveform, wav_attention_mask)

        # 프론트엔드의 트랜스포머가 패딩 위치까지 어텐션하면 유효 구간 출력이 오염된다.
        # 마스크를 안 받았으면 여기서 직접 만들어 넘긴다 — 호출부가 잊어버려도 안전하게.
        if key_padding_mask is None and wav_attention_mask is not None:
            key_padding_mask = self.frame_padding_mask(wav_attention_mask, h.size(1))
        return self.frontend(self.proj(h), key_padding_mask=key_padding_mask)

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """파형 샘플 수 -> wav2vec2 출력 프레임 수."""
        return self.w2v._get_feat_extract_output_lengths(input_lengths)
