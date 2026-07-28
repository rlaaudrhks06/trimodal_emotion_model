"""합성(무작위) 텐서로 전체 파이프라인의 차원 정합을 검증하는 스모크 테스트.

실제 KEMDy19/20·AI Hub 데이터가 아직 없는 상태에서, 설계 v3의 표기법(§3)에
정의된 T_v/T_a/T_t/d_model 형태가 끝까지 어긋나지 않고 [B, C=7] 로짓까지
나오는지 확인한다. 학습이 되는지(정확도)가 아니라 "배관이 새지 않는지"를
보는 것이 목적이다.
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


if __name__ == "__main__":
    test_forward_shapes()
    test_modality_dropout_runs()
