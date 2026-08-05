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

from src.config import load_config
from src.model import TrimodalEmotionModel
from src.model_single_modality import SingleModalityModel
from src.datasets.manifest_dataset import ManifestEmotionDataset, make_collate_fn
from src.datasets.labels import EMOTION_LABELS
from src.eval_report import print_and_collect, save_eval_result


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
    parser.add_argument(
        "--modality", choices=["audio", "visual", "text"], default=None,
        help="지정하면 SingleModalityModel(베이스라인 체크포인트)을 평가. 생략하면 트리모달 본 모델.",
    )
    parser.add_argument(
        "--save-as", type=str, default=None,
        help="결과를 results/eval/{이름}.json으로 저장 (예: v9, v7_swa). 생략하면 화면 출력만.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    train_cfg = cfg.raw["train"]
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    num_workers = train_cfg.get("num_workers", 0)
    cache_dir = train_cfg.get("feature_cache_dir")
    prosody_stats_path = train_cfg.get("prosody_stats_path")
    collate_fn = make_collate_fn(cfg.text_pretrained)
    test_ds = ManifestEmotionDataset(args.manifest, cfg, cache_dir=cache_dir, prosody_stats_path=prosody_stats_path)
    test_loader = DataLoader(
        # 학습 때와 같은 batch_size 사용 — 하드코딩된 16보다 훨씬 빠르고, 배치 크기가
        # 결과(정확도/지표)에 영향을 주지 않으므로 값을 맞출 이유는 없지만 속도상 유리하다.
        test_ds, batch_size=train_cfg["batch_size"], shuffle=False, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    if args.modality is not None:
        model = SingleModalityModel(cfg, modality=args.modality).to(device)
    else:
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

    metrics = print_and_collect(all_labels, all_preds)

    if args.save_as:
        save_eval_result(
            metrics, name=args.save_as, manifest=args.manifest,
            models=[{"config": args.config, "checkpoint": args.checkpoint}],
            extra={"modality": args.modality} if args.modality else None,
        )


if __name__ == "__main__":
    main()
