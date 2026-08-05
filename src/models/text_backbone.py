"""설계 v3 §4 ③: Text Backbone.

한국어 사전학습 언어모델(기본값 klue/bert-base)로 STT 전사문을 토큰 임베딩
시퀀스 X_t로 변환한다. 이 프로젝트는 텍스트를 시퀀스로 유지해 교차 어텐션에
사용하므로([CLS] 벡터만 쓰지 않음), 각 토큰의 last_hidden_state를 그대로
d_model로 선형 투영한다.
"""
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class TextBackbone(nn.Module):
    def __init__(
        self, pretrained_model: str, d_model: int,
        dropout: float = 0.1, freeze_layers: int = 0,
    ):
        super().__init__()
        # 과적합 대응: BERT 자체 내부 dropout(기본 0.1)을 키울 수 있게 config로 노출
        bert_cfg = AutoConfig.from_pretrained(pretrained_model)
        bert_cfg.hidden_dropout_prob = dropout
        bert_cfg.attention_probs_dropout_prob = dropout
        self.bert = AutoModel.from_pretrained(pretrained_model, config=bert_cfg)
        hidden_size = self.bert.config.hidden_size
        self.proj = nn.Linear(hidden_size, d_model)

        # 과적합 대응: 1.1억 파라미터 BERT를 통째로 파인튜닝하면 커스텀 소형 모듈들보다
        # 훨씬 큰 용량으로 학습 데이터를 암기하기 쉬움 -> 하위 N개 encoder layer(+embeddings)를
        # 얼려서 일반적인 한국어 표현은 그대로 두고, 상위 layer만 태스크에 맞춰 미세조정한다.
        if freeze_layers > 0:
            for p in self.bert.embeddings.parameters():
                p.requires_grad = False
            for layer in self.bert.encoder.layer[:freeze_layers]:
                for p in layer.parameters():
                    p.requires_grad = False
            # pooler(59만 파라미터)는 forward에서 안 쓴다 — last_hidden_state만 쓰므로
            # pooler_output은 계산되자마자 버려진다. 그래서 grad가 None이라 실제로 학습되진
            # 않지만, requires_grad=True로 남아 있으면 "학습 가능 파라미터" 집계와 옵티마이저
            # param group에 잡혀서 수치가 실제와 어긋난다(예: v7의 895만 중 59만이 허수).
            # 여기서 같이 얼려 집계를 실제와 일치시킨다. state_dict 키는 그대로라 기존
            # 체크포인트 로딩에는 영향이 없다.
            if self.bert.pooler is not None:
                for p in self.bert.pooler.parameters():
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
