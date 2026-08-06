"""설계 v3 §7.3 베이스라인 1: 단일 모달리티(텍스트 / 오디오 / 영상 only) 학습.

트리모달 본 모델과 성능을 비교해 "어느 모달리티가 실제로 정보를 주는가"를
확인하는 용도. v1~v4(트리모달 실험 계열, checkpoints_prosody_norm 등)와 절대
안 헷갈리게 체크포인트 폴더명에 "baseline_{modality}"를 쓴다 — v 번호 안 씀.

실행 예:
    python scripts/train_single_modality.py --modality text
    python scripts/train_single_modality.py --modality audio
    python scripts/train_single_modality.py --modality visual
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.model_single_modality import SingleModalityModel
from src.datasets.manifest_dataset import ManifestEmotionDataset, make_collate_fn
from scripts.train import compute_class_weights, run_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent.parent / "configs" / "config.yaml"))
    parser.add_argument("--modality", required=True, choices=["audio", "visual", "text"])
    parser.add_argument("--epochs", type=int, default=None, help="생략 시 config.yaml의 epochs 그대로 사용")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    train_cfg = cfg.raw["train"]

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[train_single_modality:{args.modality}] device = {device}")

    cache_dir = train_cfg.get("feature_cache_dir")
    prosody_stats_path = train_cfg.get("prosody_stats_path")
    collate_fn = make_collate_fn(cfg.text_pretrained)
    # wav2vec2 백본은 멜이 아니라 원본 파형을 받으므로 데이터셋에 파형도 요청한다.
    need_wav = args.modality == "audio" and cfg.audio_backbone == "wav2vec2"
    ds_kwargs = dict(cache_dir=cache_dir, prosody_stats_path=prosody_stats_path, return_waveform=need_wav)
    train_ds = ManifestEmotionDataset(train_cfg["train_manifest"], cfg, **ds_kwargs)
    val_ds = ManifestEmotionDataset(train_cfg["val_manifest"], cfg, **ds_kwargs)
    if need_wav:
        print(f"[train_single_modality:audio] wav2vec2 백본 사용 — {cfg.audio_pretrained} "
              f"layer={cfg.audio_w2v_layer} freeze={cfg.audio_w2v_freeze}", flush=True)

    num_workers = train_cfg.get("num_workers", 0)
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True, collate_fn=collate_fn, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False, collate_fn=collate_fn, **loader_kwargs)

    model = SingleModalityModel(cfg, modality=args.modality).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])

    class_weights = compute_class_weights(train_ds, device)
    label_smoothing = train_cfg.get("label_smoothing", 0.0)
    train_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    eval_loss_fn = torch.nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=train_cfg.get("lr_scheduler_factor", 0.5),
        patience=train_cfg.get("lr_scheduler_patience", 3),
        min_lr=1e-6,
    )

    # v1~v4(트리모달 실험)와 안 겹치게 이름 분리 — archived_runs로 옮길 때도
    # baseline_{modality}_best 식으로, checkpoint_v{N}_* 규칙과 구분되게 유지할 것.
    ckpt_dir = Path(f"checkpoints_baseline_{args.modality}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    epochs = args.epochs or train_cfg["epochs"]
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, train_loss_fn, optimizer, log_label=f"{args.modality} epoch {epoch:03d} train")
        val_metrics = run_epoch(model, val_loader, device, eval_loss_fn, optimizer=None, log_label=f"{args.modality} epoch {epoch:03d} val")

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["loss"])
        cur_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[{args.modality} epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['weighted_f1']:.4f} "
            f"lr={cur_lr:.2e}",
            flush=True,
        )
        if cur_lr < prev_lr:
            print(f"  -> val_loss 정체로 학습률 감소: {prev_lr:.2e} -> {cur_lr:.2e}", flush=True)

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
            print(f"  -> 최고 성능 갱신, 체크포인트 저장 (val_acc={best_val_acc:.4f})", flush=True)

    print(f"[train_single_modality:{args.modality}] 완료. 최고 val_accuracy = {best_val_acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
