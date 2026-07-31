"""학습 로그(.log) 파일에서 "에폭 요약"과 "배치 진행률"을 분리해 CSV로 정리한다.

학습 자체는 건드리지 않는다 — 이미 저장된 로그 파일을 읽기만 하는 사후 분석 도구다.

- 에폭 요약 줄("[epoch 003] train_loss=... val_acc=... lr=...")
  -> {name}_epoch_summary.csv (버전 간 비교에 필요한 핵심 지표, 항상 생성)
- 배치 진행률 줄("[epoch 003 train]  33.6% (147/438 배치) 누적loss=... 경과=...")
  -> {name}_batch_progress.csv (에폭 내부 진행 상황 참고용. v1/v2 로그처럼 이
     기능이 추가되기 전 로그에는 이 줄 자체가 없으므로 파일이 생성되지 않는다)

사용 예:
    python scripts/parse_training_log.py \
        --log train_prosody_norm.log --name checkpoint_v4 --out-dir archived_runs/csv_summary
"""
import argparse
import re
from pathlib import Path

import pandas as pd

EPOCH_SUMMARY_RE = re.compile(
    r"^\[epoch (\d+)\] "
    r"train_loss=([\d.]+) train_acc=([\d.]+) \| "
    r"val_loss=([\d.]+) val_acc=([\d.]+) val_f1=([\d.]+)"
    r"(?: lr=([\d.eE+-]+))?"
)

BATCH_PROGRESS_RE = re.compile(
    r"\[epoch (\d+) (train|val)\]\s+([\d.]+)% \((\d+)/(\d+) 배치\) 누적loss=([\d.]+) 경과=([\d.]+)분"
)


def parse_log(log_path: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    epoch_rows, batch_rows = [], []
    with open(log_path, encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()

            m_epoch = EPOCH_SUMMARY_RE.match(line)
            if m_epoch:
                epoch, train_loss, train_acc, val_loss, val_acc, val_f1, lr = m_epoch.groups()
                epoch_rows.append({
                    "epoch": int(epoch),
                    "train_loss": float(train_loss),
                    "train_acc": float(train_acc),
                    "val_loss": float(val_loss),
                    "val_acc": float(val_acc),
                    "val_f1": float(val_f1),
                    "lr": float(lr) if lr is not None else None,
                })
                continue

            m_batch = BATCH_PROGRESS_RE.search(line)
            if m_batch:
                epoch, phase, pct, batch_idx, n_batches, cum_loss, elapsed_min = m_batch.groups()
                batch_rows.append({
                    "epoch": int(epoch),
                    "phase": phase,
                    "pct": float(pct),
                    "batch_idx": int(batch_idx),
                    "n_batches": int(n_batches),
                    "cumulative_loss": float(cum_loss),
                    "elapsed_min": float(elapsed_min),
                })

    if not epoch_rows:
        raise ValueError(f"{log_path}에서 에폭 요약 줄을 하나도 못 찾음 — 로그 형식이 다른지 확인 필요")

    epoch_df = pd.DataFrame(epoch_rows).drop_duplicates(subset="epoch", keep="last").sort_values("epoch")
    batch_df = (
        pd.DataFrame(batch_rows).sort_values(["epoch", "phase", "batch_idx"])
        if batch_rows else None
    )
    return epoch_df, batch_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="정리할 .log 파일 경로")
    parser.add_argument("--name", required=True, help="출력 파일 이름 접두사(예: checkpoint_v4)")
    parser.add_argument("--out-dir", required=True, help="CSV를 저장할 폴더")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epoch_df, batch_df = parse_log(Path(args.log))

    epoch_path = out_dir / f"{args.name}_epoch_summary.csv"
    epoch_df.to_csv(epoch_path, index=False)
    best_row = epoch_df.loc[epoch_df["val_acc"].idxmax()]
    print(f"[parse_training_log] {args.log}: 에폭 {len(epoch_df)}개 -> {epoch_path}")
    print(
        f"  최고 val_acc={best_row['val_acc']:.4f} @ epoch{int(best_row['epoch'])} "
        f"(train_acc={best_row['train_acc']:.4f}, val_loss={best_row['val_loss']:.4f})"
    )

    if batch_df is not None:
        batch_path = out_dir / f"{args.name}_batch_progress.csv"
        batch_df.to_csv(batch_path, index=False)
        print(f"[parse_training_log] 배치 진행률 {len(batch_df)}줄 -> {batch_path}")
    else:
        print("[parse_training_log] 배치 진행률 로그 없음(v1/v2 시절 로그로 추정) -> batch_progress.csv 생성 안 함")


if __name__ == "__main__":
    main()
