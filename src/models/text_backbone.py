"""설계 v3 §4 ③: Text Backbone.

한국어 사전학습 언어모델(기본값 klue/bert-base)로 STT 전사문을 토큰 임베딩
시퀀스 X_t로 변환한다. 이 프로젝트는 텍스트를 시퀀스로 유지해 교차 어텐션에
사용하므로([CLS] 벡터만 쓰지 않음), 각 토큰의 last_hidden_state를 그대로
d_model로 선형 투영한다.
"""
import torch
import torch.nn as nn
from transformers import AutoModel


class TextBackbone(nn.Module):
    def __init__(self, pretrained_model: str, d_model: int, freeze_base: bool = False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model)
        hidden_size = self.bert.config.hidden_size
        self.proj = nn.Linear(hidden_size, d_model)
        if freeze_base:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # input_ids/attention_mask: [B, T_t]
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x_t = self.proj(out.last_hidden_state)  # [B, T_t, d_model]
        return x_t

    @property
    def key_padding_mask_from_attention_mask(self):
        # attention_mask==1이 "실제 토큰", nn.Transformer의 key_padding_mask는
        # True가 "무시할 위치"이므로 반전해서 써야 한다는 점을 호출부에 상기시키는 헬퍼.
        def _fn(attention_mask: torch.Tensor) -> torch.Tensor:
            return attention_mask == 0

        return _fn
