"""학습 스크립트. 설계 v3 §7 학습 및 평가 프로토콜.

실행 예:
    python scripts/train.py --config configs/config.yaml
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 개인 노트북에서 다른 작업과 같이 돌릴 때만 CPU를 낮게 제한한다.
# THROTTLE_CPU=1 환경변수를 줘야 활성화됨 — 전용 GPU 서버(예: A100)에서는 기본값(끔)으로
# 모든 코어를 다 써서 데이터 로딩이 GPU를 굶기지 않게 한다.
THROTTLE_CPU = os.environ.get("THROTTLE_CPU") == "1"
if THROTTLE_CPU:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")

import cv2
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

if THROTTLE_CPU:
    cv2.setNumThreads(2)
    torch.set_num_threads(2)

from src.config import load_config
from src.model import TrimodalEmotionModel
from src.datasets.manifest_dataset import ManifestEmotionDataset, make_collate_fn
from src.datasets.labels import EMOTION_LABELS, LABEL_TO_IDX


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


def compute_class_weights(train_ds: ManifestEmotionDataset, device: torch.device) -> torch.Tensor:
    """클래스 불균형 대응: 데이터가 적은 클래스일수록 틀렸을 때 벌점을 크게 준다.

    1차 실험(400클립, 경멸 57개): 가중치 없음 → 다수 클래스로 완전히 쏠림.
    2차 실험(1/count 가중치): 너무 세서 오히려 학습 불안정(train_acc가 22%까지밖에 못 오름).
    지금은 80,122개로 데이터가 커지고 불균형 비율도 완화(16배→6.35배)되어서,
    1/count보다 완만한 1/sqrt(count)로 재시도 — 극단적인 소수 클래스 과대보정을 피한다.
    평균이 1이 되도록 정규화(전체 loss 스케일이 크게 안 변하게).
    """
    counts = train_ds.df["label"].astype(str).str.strip().str.lower().value_counts()
    weights = torch.zeros(len(EMOTION_LABELS))
    for label, idx in LABEL_TO_IDX.items():
        c = counts.get(label, 0)
        weights[idx] = 1.0 / (c ** 0.5) if c > 0 else 0.0
    weights = weights * (len(EMOTION_LABELS) / weights.sum())

    print("[train] 클래스 가중치 (적을수록 큰 값):")
    for label, idx in LABEL_TO_IDX.items():
        print(f"  {label:10s}: count={int(counts.get(label, 0)):4d}  weight={weights[idx]:.3f}")

    return weights.to(device)


def run_epoch(model, loader, device, loss_fn, optimizer=None, log_label: str = "") -> dict:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, all_preds, all_labels = 0.0, [], []

    # 배치 몇 %까지 왔는지 실시간으로 보여준다 — 몇 시간짜리 에폭을 ps/nvidia-smi로만
    # 간접 확인해야 했던 문제 때문에 추가함. nohup으로 리다이렉트해도 바로 보이도록
    # flush=True로 강제 출력.
    n_batches = len(loader)
    log_every = max(1, n_batches // 20)  # 대략 5%마다 한 번씩
    start_time = time.time()

    with torch.set_grad_enabled(is_train):
        for batch_idx, batch in enumerate(loader, 1):
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
            loss = loss_fn(logits, batch["labels"])

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * logits.size(0)
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

            if batch_idx % log_every == 0 or batch_idx == n_batches:
                pct = 100.0 * batch_idx / n_batches
                elapsed_min = (time.time() - start_time) / 60
                running_loss = total_loss / len(all_labels)
                print(
                    f"    [{log_label}] {pct:5.1f}% ({batch_idx}/{n_batches} 배치) "
                    f"누적loss={running_loss:.4f} 경과={elapsed_min:.1f}분",
                    flush=True,
                )

    return {
        "loss": total_loss / len(all_labels),
        "accuracy": accuracy_score(all_labels, all_preds),
        "weighted_f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent.parent / "configs" / "config.yaml"))
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    train_cfg = cfg.raw["train"]

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[train] device = {device}")

    cache_dir = train_cfg.get("feature_cache_dir")
    collate_fn = make_collate_fn(cfg.text_pretrained)
    train_ds = ManifestEmotionDataset(train_cfg["train_manifest"], cfg, cache_dir=cache_dir)
    val_ds = ManifestEmotionDataset(train_cfg["val_manifest"], cfg, cache_dir=cache_dir)

    # num_workers>0이면 오디오/영상 전처리(느린 CPU 작업)를 여러 프로세스가 병렬로 미리
    # 준비해두므로 GPU가 놀지 않는다 — 노트북에서는 0(안전), 전용 서버에서는 CPU 코어 수만큼 올릴 것.
    num_workers = train_cfg.get("num_workers", 0)
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True, collate_fn=collate_fn, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False, collate_fn=collate_fn, **loader_kwargs)

    model = TrimodalEmotionModel(cfg, modality_dropout_prob=train_cfg["modality_dropout_prob"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])

    class_weights = compute_class_weights(train_ds, device)
    # 과적합 대응: label smoothing — 정답 라벨에 100% 확신을 두지 않게 해서 학습 데이터의
    # 잡음/애매한 라벨(감정은 원래 경계가 모호함)에 과하게 확신하는 것을 막는다.
    # 검증 손실은 실제 분포 기준 지표를 그대로 보기 위해 smoothing/가중치 없이 계산.
    label_smoothing = train_cfg.get("label_smoothing", 0.0)
    train_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    eval_loss_fn = torch.nn.CrossEntropyLoss()

    # 과적합 대응: val_loss가 개선되지 않으면 학습률을 낮춰서 이미 찾은 좋은 지점 근처에서
    # 더 조심스럽게(과적합 덜 일으키며) 탐색하게 한다. 최근 학습에서 val_loss가 계속
    # 올라가기만 하는 패턴이 관찰되어 추가.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=train_cfg.get("lr_scheduler_factor", 0.5),
        patience=train_cfg.get("lr_scheduler_patience", 3),
        min_lr=1e-6,
    )

    ckpt_dir = Path(train_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 최고 성능(best_model.pt)과 별개로, N에폭마다 스냅샷을 따로 저장해둔다 —
    # 나중에 특정 에폭 시점으로 되돌아가 비교하고 싶을 때(예: 과적합 시작 지점 확인) 필요.
    periodic_ckpt_dir = Path(train_cfg.get("periodic_checkpoint_dir", "checkpoints_periodic"))
    periodic_ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_interval = train_cfg.get("checkpoint_interval", 20)

    best_val_acc = 0.0

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_metrics = run_epoch(model, train_loader, device, train_loss_fn, optimizer, log_label=f"epoch {epoch:03d} train")
        val_metrics = run_epoch(model, val_loader, device, eval_loss_fn, optimizer=None, log_label=f"epoch {epoch:03d} val")

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["loss"])
        cur_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[epoch {epoch:03d}] "
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

        if epoch % checkpoint_interval == 0:
            snapshot_path = periodic_ckpt_dir / f"checkpoint_epoch{epoch}.pt"
            torch.save(model.state_dict(), snapshot_path)
            print(f"  -> {checkpoint_interval}에폭마다 스냅샷 저장: {snapshot_path}", flush=True)

    print(f"[train] 완료. 최고 val_accuracy = {best_val_acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
