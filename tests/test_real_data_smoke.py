"""실제 AI Hub 데이터(data/manifests/test.csv)로 전체 파이프라인 검증.

합성 텐서가 아니라 진짜 오디오 파형/얼굴 프레임/전사문으로 데이터 로더 ->
collate_fn -> 모델 forward까지 실제로 돌려서 차원과 파이프라인이 실데이터에서도
문제없이 동작하는지 확인한다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.model import TrimodalEmotionModel
from src.datasets.manifest_dataset import ManifestEmotionDataset, make_collate_fn
from src.datasets.labels import IDX_TO_LABEL


def main():
    cfg = load_config()
    manifest = Path(__file__).resolve().parent.parent / "data" / "manifests" / "test.csv"

    ds = ManifestEmotionDataset(str(manifest), cfg)
    print(f"[real smoke test] 데이터셋 크기: {len(ds)}")

    collate_fn = make_collate_fn(cfg.text_pretrained)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn)

    model = TrimodalEmotionModel(cfg, modality_dropout_prob=0.0)
    model.eval()

    for batch in loader:
        with torch.no_grad():
            logits = model(
                mel_spec=batch["mel_spec"],
                prosody_vec=batch["prosody_vec"],
                frames=batch["frames"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                audio_padding_mask=batch["audio_padding_mask"],
                visual_padding_mask=batch["visual_padding_mask"],
            )
        preds = logits.argmax(dim=-1).tolist()
        print(f"[real smoke test] batch mel_spec={tuple(batch['mel_spec'].shape)} "
              f"frames={tuple(batch['frames'].shape)} input_ids={tuple(batch['input_ids'].shape)}")
        for utt_id, true_label, pred in zip(batch["utt_ids"], batch["labels"].tolist(), preds):
            print(f"  {utt_id}: 정답={IDX_TO_LABEL[true_label]} / (학습 전) 예측={IDX_TO_LABEL[pred]}")

    print("[real smoke test] PASSED — 실제 오디오/영상/텍스트 데이터로 forward pass 성공")


if __name__ == "__main__":
    main()
