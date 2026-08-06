"""학습 스크립트. 설계 v3 §7 학습 및 평가 프로토콜.

실행 예:
    python scripts/train.py --config configs/config.yaml
"""
import argparse
import os
import random
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
from src.datasets.manifest_dataset import ManifestEmotionDataset, make_collate_fn, IGNORE_INDEX
from src.datasets.labels import EMOTION_LABELS, LABEL_TO_IDX, normalize_label


def set_seed(seed: int) -> None:
    # v6까지 seed 고정이 전혀 없어서, 같은 config를 두 번 돌려도 가중치 초기화·
    # DataLoader 셔플 순서·dropout 마스크가 매번 달라졌다 — v6/v7/v8처럼 0.5~2pp
    # 단위로 버전을 비교할 때, 그 차이가 설정 변경 효과인지 단순 실행 간 노이즈인지
    # 구분할 수가 없었다. cudnn 완전결정(deterministic=True)까지는 강제하지 않음 —
    # 학습 속도 저하가 크고, RNG 시드 고정만으로도 실행 간 노이즈를 충분히 줄일 수 있음.
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    # normalize_label: 매니페스트 CSV엔 "contempt" 문자열이 그대로 남아있으므로
    # (8->7클래스 병합, src/datasets/labels.py 참고) 여기서도 흡수해야 disgust 카운트가
    # 안 빠짐 — 안 그러면 contempt 행들이 어느 클래스 가중치에도 안 잡히는 버그가 생김.
    counts = train_ds.df["label"].astype(str).str.strip().str.lower().map(normalize_label).value_counts()
    weights = torch.zeros(len(EMOTION_LABELS))
    for label, idx in LABEL_TO_IDX.items():
        c = counts.get(label, 0)
        weights[idx] = 1.0 / (c ** 0.5) if c > 0 else 0.0
    weights = weights * (len(EMOTION_LABELS) / weights.sum())

    print("[train] 클래스 가중치 (적을수록 큰 값):")
    for label, idx in LABEL_TO_IDX.items():
        print(f"  {label:10s}: count={int(counts.get(label, 0)):4d}  weight={weights[idx]:.3f}")

    return weights.to(device)


def aux_loss(aux_logits: dict, batch: dict, loss_fn) -> torch.Tensor | None:
    """v12 보조 손실. 유효한 보조 라벨이 하나도 없으면 None을 돌려준다.

    **nan 방어가 이 함수의 존재 이유다.** 보조 라벨이 없는 표본은 IGNORE_INDEX(-100)로
    들어오고 CrossEntropyLoss가 알아서 건너뛰는데, **배치 전체가 무시 대상이면 0으로
    나누게 되어 nan이 나온다**(실측 확인). nan은 역전파를 타고 전 가중치를 파괴하는데
    로그에는 loss=nan만 뜨고 원인이 안 보인다. 그래서 유효 표본 수를 먼저 세고,
    0이면 그 항을 아예 빼버린다.

    커버리지가 99.8%라 실사용에서 배치 전체가 비는 일은 거의 없지만, 보조 라벨이
    희박한 매니페스트를 쓰면 조용히 터지므로 막아둔다.
    """
    terms = []
    for key, logits in aux_logits.items():
        target = batch.get(key)
        if target is None:
            continue
        if (target != IGNORE_INDEX).sum() == 0:
            continue  # 이 배치엔 이 모달리티의 유효 라벨이 하나도 없음 -> 항 제외
        terms.append(loss_fn(logits, target))
    return torch.stack(terms).sum() if terms else None


def run_epoch(model, loader, device, loss_fn, optimizer=None, log_label: str = "",
              aux_loss_fn=None, aux_weight: float = 0.0) -> dict:
    """aux_loss_fn과 aux_weight를 주면 v12 보조 손실을 함께 학습한다.

    둘 중 하나라도 없으면(기본) 보조 경로를 아예 타지 않아 v1~v11과 완전히 동일하다.
    검증(is_train=False) 시에는 보조 손실을 빼고 주 손실만 본다 — 체크포인트 선택과
    학습률 스케줄이 보조 과제 성적에 흔들리면 안 되기 때문이다.
    """
    is_train = optimizer is not None
    use_aux = is_train and aux_loss_fn is not None and aux_weight > 0

    model.train() if is_train else model.eval()

    total_loss, all_preds, all_labels = 0.0, [], []
    aux_sum, aux_batches = 0.0, 0

    # 배치 몇 %까지 왔는지 실시간으로 보여준다 — 몇 시간짜리 에폭을 ps/nvidia-smi로만
    # 간접 확인해야 했던 문제 때문에 추가함. nohup으로 리다이렉트해도 바로 보이도록
    # flush=True로 강제 출력.
    n_batches = len(loader)
    log_every = max(1, n_batches // 20)  # 대략 5%마다 한 번씩
    start_time = time.time()

    with torch.set_grad_enabled(is_train):
        for batch_idx, batch in enumerate(loader, 1):
            batch = move_batch_to_device(batch, device)
            model_inputs = dict(
                mel_spec=batch["mel_spec"],
                prosody_vec=batch["prosody_vec"],
                frames=batch["frames"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                audio_padding_mask=batch["audio_padding_mask"],
                visual_padding_mask=batch["visual_padding_mask"],
            )
            # wav2vec2 오디오 백본을 쓸 때만 배치에 원본 파형이 들어있다
            # (ManifestEmotionDataset(return_waveform=True)). 없으면 그대로 지나가므로
            # 기존 멜 경로·트리모달 모델은 영향받지 않는다.
            if "waveform" in batch:
                model_inputs["waveform"] = batch["waveform"]
                model_inputs["wav_attention_mask"] = batch["wav_attention_mask"]
            if use_aux:
                logits, aux_logits = model(**model_inputs, return_aux=True)
            else:
                logits = model(**model_inputs)
                aux_logits = {}
            main_loss = loss_fn(logits, batch["labels"])
            loss = main_loss

            if aux_logits:
                a = aux_loss(aux_logits, batch, aux_loss_fn)
                if a is not None:
                    loss = main_loss + aux_weight * a
                    aux_sum += a.item()
                    aux_batches += 1

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # 최적화 안정성 대응: gradient clipping이 지금까지 전혀 없었음 — 특히 v8부터
                # 사전학습 BERT 상위 층과 처음부터 학습하는 모듈을 서로 다른 lr로 같이
                # 업데이트하는데, 이런 상황에서 튀는 gradient가 학습을 불안정하게 만들 수 있어
                # 표준 관행(max_norm=1.0)을 추가한다.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # 역전파는 loss(주+보조)로 하되, 기록·로그는 **주 손실만** 쌓는다.
            # 보조 항까지 섞으면 로그의 train_loss가 v1~v11과 비교 불가능해지고,
            # 보조 가중치를 바꿀 때마다 값이 튀어 곡선 해석이 어려워진다.
            total_loss += main_loss.item() * logits.size(0)
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

    out = {
        "loss": total_loss / len(all_labels),
        "accuracy": accuracy_score(all_labels, all_preds),
        "weighted_f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
    }
    if aux_batches:
        # 보조 손실이 실제로 몇 배치에 걸렸는지도 함께 낸다. 라벨 결측이 많으면
        # 이 수가 전체 배치 수보다 훨씬 작게 나와 바로 눈에 띈다.
        out["aux_loss"] = aux_sum / aux_batches
        out["aux_batches"] = aux_batches
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent.parent / "configs" / "config.yaml"))
    parser.add_argument("--seed", type=int, default=42, help="재현성/버전 간 비교 노이즈 축소용 — 같은 config를 다른 seed로 여러 번 돌려 결과 폭을 확인할 때 바꿔서 사용")
    args = parser.parse_args()

    set_seed(args.seed)

    cfg = load_config(Path(args.config))
    train_cfg = cfg.raw["train"]

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[train] device = {device}, seed = {args.seed}")

    cache_dir = train_cfg.get("feature_cache_dir")
    # None(기본)이면 기존과 동일하게 동작 — 데이터 전처리 EDA 문서의 prosody 정규화를
    # 적용한 A/B 비교 run에서만 config.yaml에 이 경로를 지정한다.
    prosody_stats_path = train_cfg.get("prosody_stats_path")
    collate_fn = make_collate_fn(cfg.text_pretrained)
    # wav2vec2 오디오 백본은 멜이 아니라 원본 파형을 받는다(8.18절).
    need_wav = cfg.audio_backbone == "wav2vec2"
    ds_kwargs = dict(cache_dir=cache_dir, prosody_stats_path=prosody_stats_path, return_waveform=need_wav)
    train_ds = ManifestEmotionDataset(train_cfg["train_manifest"], cfg, **ds_kwargs)
    val_ds = ManifestEmotionDataset(train_cfg["val_manifest"], cfg, **ds_kwargs)
    if need_wav:
        print(f"[train] wav2vec2 오디오 백본 — {cfg.audio_pretrained} "
              f"layer={cfg.audio_w2v_layer} freeze={cfg.audio_w2v_freeze}", flush=True)

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

    # 과적합 대응(8.11/8.12절): BERT 상위 층을 나머지 모듈과 동일한 lr로 파인튜닝한 게
    # v1~v6 반복된 과적합의 주요 원인 중 하나였음을 확인. v7은 BERT를 아예 동결해 검증했고,
    # 여기서는 "동결 대신 훨씬 낮은 lr(bert_lr, 기본 2e-5 — 표준 BERT 파인튜닝 lr)로 살짝만
    # 적응시키면 v7보다 나은지"를 테스트하기 위해 param group을 분리한다. bert_lr을 config에
    # 안 주면 기존과 동일하게 단일 lr로 동작(하위 호환).
    bert_lr = train_cfg.get("bert_lr", train_cfg["lr"])
    bert_params = [p for n, p in model.named_parameters() if n.startswith("text_backbone.bert.") and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if not n.startswith("text_backbone.bert.") and p.requires_grad]
    print(
        f"[train] param groups: bert={sum(p.numel() for p in bert_params):,}개(lr={bert_lr:.1e}), "
        f"other={sum(p.numel() for p in other_params):,}개(lr={train_cfg['lr']:.1e})",
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": bert_params, "lr": bert_lr},
            {"params": other_params, "lr": train_cfg["lr"]},
        ],
        weight_decay=train_cfg["weight_decay"],
    )

    class_weights = compute_class_weights(train_ds, device)
    # 과적합 대응: label smoothing — 정답 라벨에 100% 확신을 두지 않게 해서 학습 데이터의
    # 잡음/애매한 라벨(감정은 원래 경계가 모호함)에 과하게 확신하는 것을 막는다.
    # 검증 손실은 실제 분포 기준 지표를 그대로 보기 위해 smoothing/가중치 없이 계산.
    label_smoothing = train_cfg.get("label_smoothing", 0.0)
    train_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    eval_loss_fn = torch.nn.CrossEntropyLoss()

    # v12 보조 손실(11.2절). aux_weight가 0(기본)이면 보조 경로를 아예 타지 않아
    # v1~v11과 완전히 동일하게 동작한다.
    #
    # 보조 손실에는 클래스 가중치를 걸지 않는다: 가중치는 train 세트의 **주 라벨** 분포에서
    # 계산한 것이라 모달리티별 라벨 분포와 다르다(예: 소리 라벨은 혐오 28.3%로 분포가 또 다름).
    # 맞지 않는 가중치를 걸면 보조 과제가 왜곡된다. label_smoothing은 주 손실과 맞춘다.
    aux_weight = float(train_cfg.get("aux_loss_weight", 0.0))
    aux_loss_fn = None
    if aux_weight > 0:
        if not model.use_aux:
            raise ValueError(
                "train.aux_loss_weight > 0인데 model.aux_head_dim이 0이라 보조 헤드가 없다 — "
                "config에서 model.aux_head_dim을 설정할 것"
            )
        if not train_ds.aux_columns:
            raise ValueError(
                f"train.aux_loss_weight > 0인데 {train_cfg['train_manifest']}에 보조 라벨 컬럼이 없다 — "
                "먼저 scripts/add_modality_labels.py를 돌릴 것"
            )
        aux_loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        print(f"[train] 보조 손실 활성 — 가중치 {aux_weight}, 컬럼 {list(train_ds.aux_columns)}", flush=True)

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
        train_metrics = run_epoch(model, train_loader, device, train_loss_fn, optimizer,
                                  log_label=f"epoch {epoch:03d} train",
                                  aux_loss_fn=aux_loss_fn, aux_weight=aux_weight)
        val_metrics = run_epoch(model, val_loader, device, eval_loss_fn, optimizer=None, log_label=f"epoch {epoch:03d} val")

        # param_groups[1]이 기존 모든 버전과 동일한 "메인" lr(크로스어텐션/분류기/프론트엔드) —
        # parse_training_log.py의 lr= 파싱과 버전 간 CSV 호환을 위해 이 값을 lr=로 유지하고,
        # bert 그룹(param_groups[0])은 별도 bert_lr=로 덧붙인다(파서는 뒤 텍스트를 무시하므로 안전).
        prev_lr = optimizer.param_groups[1]["lr"]
        prev_bert_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["loss"])
        cur_lr = optimizer.param_groups[1]["lr"]
        cur_bert_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['weighted_f1']:.4f} "
            f"lr={cur_lr:.2e} bert_lr={cur_bert_lr:.2e}",
            flush=True,
        )
        if "aux_loss" in train_metrics:
            # 보조 손실은 별도 줄로 낸다 — parse_training_log.py의 에폭 요약 정규식이
            # 위 줄의 형식에 맞춰져 있어서, 같은 줄에 끼워 넣으면 v1~v11 CSV와
            # 호환이 깨진다(정규식은 뒤 텍스트를 무시하지만 lr 앞에 끼면 매칭 실패).
            print(f"  -> 보조손실={train_metrics['aux_loss']:.4f} "
                  f"(적용 배치 {train_metrics['aux_batches']}/{len(train_loader)})", flush=True)
        if cur_lr < prev_lr:
            print(f"  -> val_loss 정체로 학습률 감소: {prev_lr:.2e} -> {cur_lr:.2e} (bert: {prev_bert_lr:.2e} -> {cur_bert_lr:.2e})", flush=True)

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
