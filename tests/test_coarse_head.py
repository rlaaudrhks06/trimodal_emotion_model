"""v11g 3클래스 머리 검증 (13.14절).

  ① coarse_head=False(v11e 설정) -> classifier.coarse 없음, 파라미터 수 v11e와 동일, state_dict 키 동일
  ② coarse_head=True  -> forward()는 7클래스 로짓만(배포 경로 불변), return_aux=True면 aux["coarse"] [B,3]
  ③ 3클래스 묶음이 배포 엔진(robot/brain/engine.py COARSE)과 같다
  ④ 손실: 7클래스 라벨 -> 3클래스 인덱스 변환 + 중립 가중 CE가 유한하고 coarse 머리에 grad가 간다
  ⑤ v11e 체크포인트를 v11g 모델에 strict=False로 실으면 coarse.* 만 빠진다 (기존 가중치 재사용 가능)

합성 입력으로 돈다(실데이터 불필요, CPU 가능).
    python tests/test_coarse_head.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from src.config import load_config
from src.model import TrimodalEmotionModel
from src.datasets.labels import EMOTION_LABELS, COARSE_LABELS, COARSE_OF, COARSE_IDX_OF_LABEL_IDX

ROOT = Path(__file__).resolve().parent.parent


def fake_batch(cfg, B=2):
    T, F, N = 40, cfg.n_mels if hasattr(cfg, "n_mels") else 80, 4
    return dict(
        mel_spec=torch.randn(B, T, F), prosody_vec=torch.randn(B, cfg.model.prosody_dim),
        frames=torch.rand(B, N, 3, 112, 112), input_ids=torch.randint(5, 100, (B, 12)),
        attention_mask=torch.ones(B, 12, dtype=torch.long),
        audio_padding_mask=torch.zeros(B, T, dtype=torch.bool), visual_padding_mask=torch.zeros(B, N, dtype=torch.bool),
        waveform=torch.randn(B, 16000) * 0.1, wav_attention_mask=torch.ones(B, 16000, dtype=torch.long),
    )


def main() -> int:
    print("=== v11g 3클래스 머리 검증 ===")
    torch.manual_seed(0)
    cfg_e = load_config(ROOT / "configs" / "config_noise_aug_plus263_ft4.yaml")
    cfg_g = load_config(ROOT / "configs" / "config_noise_aug_plus263_ft4_coarse.yaml")
    assert not cfg_e.model.coarse_head and cfg_g.model.coarse_head
    m_e = TrimodalEmotionModel(cfg_e); m_g = TrimodalEmotionModel(cfg_g)
    # ①
    assert m_e.classifier.coarse is None and not m_e.use_coarse
    n_e = sum(p.numel() for p in m_e.parameters()); n_g = sum(p.numel() for p in m_g.parameters())
    hidden = m_g.classifier.mlp[3].in_features
    assert n_g - n_e == hidden * 3 + 3, (n_g - n_e, hidden)
    assert set(m_g.state_dict()) - set(m_e.state_dict()) == {"classifier.coarse.weight", "classifier.coarse.bias"}
    print(f"  ✅ v11e 설정엔 머리 없음 · v11g는 +{n_g - n_e:,}개(Linear {hidden}->3)만 추가")
    # ②
    m_g.eval(); b = fake_batch(cfg_g)
    with torch.no_grad():
        out = m_g(**b); logits, aux = m_g(**b, return_aux=True)
    assert isinstance(out, torch.Tensor) and out.shape == (2, len(EMOTION_LABELS)), out.shape
    assert aux["coarse"].shape == (2, len(COARSE_LABELS)) and torch.allclose(out, logits)
    print("  ✅ forward()는 7클래스 로짓만 · return_aux=True에 aux['coarse'] [B,3]")
    # ③
    sys.path.insert(0, str(ROOT / "robot" / "brain"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("engine", ROOT / "robot" / "brain" / "engine.py")
    eng = importlib.util.module_from_spec(spec); sys.modules["engine"] = eng; spec.loader.exec_module(eng)
    groups_engine = {e: eng.COARSE[e] for e in EMOTION_LABELS}
    for a in EMOTION_LABELS:
        for c in EMOTION_LABELS:
            assert (COARSE_OF[a] == COARSE_OF[c]) == (groups_engine[a] == groups_engine[c]), (a, c)
    assert [COARSE_LABELS[i] for i in COARSE_IDX_OF_LABEL_IDX] == [COARSE_OF[e] for e in EMOTION_LABELS]
    print(f"  ✅ 3클래스 묶음이 엔진과 동일: {COARSE_OF}")
    # ④
    m_g.train()
    labels = torch.tensor([EMOTION_LABELS.index("neutral"), EMOTION_LABELS.index("sad")])
    cmap = torch.tensor(COARSE_IDX_OF_LABEL_IDX)
    assert cmap[labels].tolist() == [COARSE_LABELS.index("neutral"), COARSE_LABELS.index("negative")]
    cw = torch.ones(3); cw[COARSE_LABELS.index("neutral")] = 2.0
    logits, aux = m_g(**b, return_aux=True)
    loss = torch.nn.functional.cross_entropy(logits, labels) + 0.5 * torch.nn.functional.cross_entropy(aux["coarse"], cmap[labels], weight=cw)
    assert torch.isfinite(loss); loss.backward()
    assert m_g.classifier.coarse.weight.grad is not None and m_g.classifier.mlp[0].weight.grad is not None
    print(f"  ✅ 손실 유한({loss.item():.3f}) · coarse 머리·공유 은닉층에 grad")
    # ⑤
    missing, unexpected = m_g.load_state_dict(m_e.state_dict(), strict=False)
    assert set(missing) == {"classifier.coarse.weight", "classifier.coarse.bias"} and not unexpected
    print("  ✅ v11e 가중치를 v11g에 실으면 coarse.* 만 빠짐")
    print("전부 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
