"""v11d_audio263 오디오 갈래 이식 검증 (13.15절).

  ① 오디오 단독 모델(SingleModalityModel audio, ft4 설정)도 9~12층이 학습 가능(finetune_range == (9,12))
  ② 단독 모델 state_dict → 트리모달 load_audio_from_single_modality: audio_backbone 전 파라미터가 값까지 같아진다
  ③ 얼굴·글·융합·분류기는 안 바뀐다
  ④ 키가 빠진 체크포인트(예: 텍스트 단독)는 ValueError — 조용한 부분 적재 없음

합성 입력, CPU 가능.
    python tests/test_audio_init.py
"""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from src.config import load_config
from src.model import TrimodalEmotionModel
from src.model_single_modality import SingleModalityModel

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    print("=== v11d_audio263 오디오 갈래 이식 검증 ===")
    cfg1 = load_config(ROOT / "configs" / "config_v11d_audio263_stage1.yaml")
    cfg2 = load_config(ROOT / "configs" / "config_v11d_audio263.yaml")
    torch.manual_seed(1); single = SingleModalityModel(cfg1, modality="audio")
    # ①
    assert single.backbone.finetune_range == (9, 12), single.backbone.finetune_range
    n_ft = sum(p.numel() for n, p in single.named_parameters() if n.startswith("backbone.w2v.") and p.requires_grad)
    assert n_ft > 0
    print(f"  ✅ 단독 모델 9~12층 학습 가능 ({n_ft:,}개)")
    # 값을 흔들어 "사전학습 값 그대로"와 구분되게 한다
    with torch.no_grad():
        for p in single.backbone.parameters(): p.add_(torch.randn_like(p) * 1e-3)
    tmp = Path(tempfile.mkdtemp()) / "best_model.pt"; torch.save(single.state_dict(), tmp)
    # ② ③
    torch.manual_seed(2); tri = TrimodalEmotionModel(cfg2)
    before = {k: v.clone() for k, v in tri.state_dict().items() if not k.startswith("audio_backbone.")}
    info = tri.load_audio_from_single_modality(tmp)
    src = single.backbone.state_dict()
    for k, v in tri.audio_backbone.state_dict().items():
        assert torch.equal(v, src[k]), f"audio_backbone.{k} 불일치"
    print(f"  ✅ audio_backbone {info['init_audio_keys']}키 {info['init_audio_params']:,}개 전부 이식")
    after = tri.state_dict()
    for k, v in before.items():
        assert torch.equal(v, after[k]), f"{k} 가 바뀌었다"
    print(f"  ✅ 나머지 {len(before)}키(얼굴·글·융합·분류기) 불변")
    # ④
    bad = Path(tempfile.mkdtemp()) / "bad.pt"; torch.save({"backbone.foo": torch.zeros(1)}, bad)
    try:
        tri.load_audio_from_single_modality(bad); raise AssertionError("빠진 키인데 통과했다")
    except ValueError as e:
        print(f"  ✅ 키 빠진 체크포인트는 거부: {str(e)[:40]}…")
    print("전부 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
