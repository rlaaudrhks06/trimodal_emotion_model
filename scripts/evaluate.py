"""평가 스크립트. 설계 v3 §7.2 — Accuracy, weighted F1, 클래스별 Confusion Matrix.

실행 예:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --manifest data/manifests/test.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from src.config import load_config
from src.model import TrimodalEmotionModel
from src.datasets.manifest_dataset import ManifestEmotionDataset, make_collate_fn
from src.datasets.labels import EMOTION_LABELS


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent.parent / "configs" / "config.yaml"))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    num_workers = cfg.raw["train"].get("num_workers", 0)
    cache_dir = cfg.raw["train"].get("feature_cache_dir")
    collate_fn = make_collate_fn(cfg.text_pretrained)
    test_ds = ManifestEmotionDataset(args.manifest, cfg, cache_dir=cache_dir)
    test_loader = DataLoader(
        test_ds, batch_size=16, shuffle=False, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    model = TrimodalEmotionModel(cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = move_batch_to_device(batch, device)
            logits = model(
                mel_spec=batch["mel_spec"],
                prosody_vec=batch["prosody_vec"],
                frames=batch["frames"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                audio_padding_mask=batch["audio_padding_mask"],
                visual_padding_mask=batch["visual_padding_mask"],
            )
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    w_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(EMOTION_LABELS))))

    print(f"Accuracy      : {acc:.4f}")
    print(f"Weighted F1   : {w_f1:.4f}")
    print("\nConfusion Matrix (행=정답, 열=예측):")
    print("        " + " ".join(f"{l[:4]:>6}" for l in EMOTION_LABELS))
    for label, row in zip(EMOTION_LABELS, cm):
        print(f"{label[:6]:>8}" + " ".join(f"{v:>6}" for v in row))

    print("\n" + classification_report(all_labels, all_preds, target_names=EMOTION_LABELS, zero_division=0))


if __name__ == "__main__":
    main()
