"""평가 결과를 화면 출력과 동시에 JSON 파일로도 남기는 공용 헬퍼.

배경: 학습 곡선(에폭별 val_acc 등)은 `parse_training_log.py`로 CSV를 만들어
`results/csv_summary/`에 커밋해왔는데, 정작 **최종 성능 수치(test accuracy,
F1, 혼동 행렬)는 터미널 출력으로만 확인하고 어디에도 파일로 남기지 않았다.**
버전이 9개, 앙상블 조합까지 늘어난 시점에서 이 수치들의 원본이 스크롤백에만
있는 건 위험해서, 평가할 때마다 자동으로 파일에 남기도록 했다.

`results/eval/`에 저장하며(.gitignore 예외로 커밋 대상), 파일 하나가 그
평가 1회를 온전히 재현·검증할 수 있도록 설정(config/checkpoint 경로)까지 함께 기록한다.
"""
import json
from datetime import datetime
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
)

from .datasets.labels import EMOTION_LABELS

DEFAULT_EVAL_DIR = Path("results/eval")
DEFAULT_PRED_DIR = Path("results/predictions")


def save_predictions(utt_ids, all_labels, all_preds, all_probs, name: str,
                     out_dir: str | Path | None = None) -> Path:
    """발화 하나하나의 정답·예측·확률을 CSV로 남긴다.

    왜 필요한가(8.29.1절): v11과 v12b의 test 정확도 차이 0.97%p가 유의한지
    판단하려면 짝지어 검정(McNemar)을 해야 하는데, 두 모델이 **각각 어느 발화를
    맞히고 틀렸는지**를 저장하지 않아 사후에 할 수가 없었다. 집계 지표만으로는
    "차이의 표준오차"를 독립 가정으로밖에 못 구해 실제보다 보수적이 된다.

    확률까지 남기는 이유: 신뢰도 보정(temperature scaling)과 판단 보류(abstention)
    분석이 전부 확률을 필요로 하는데, 이걸 저장해두면 **모델을 다시 돌리지 않고도**
    할 수 있다. 짧은 발화 부분집합·화자별 분산 같은 사후 분석도 마찬가지다.
    """
    import pandas as pd

    out_dir = Path(out_dir) if out_dir else DEFAULT_PRED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(all_labels)
    if not (len(utt_ids) == len(all_preds) == len(all_probs) == n):
        raise ValueError(
            f"길이 불일치: utt_id {len(utt_ids)}, label {n}, "
            f"pred {len(all_preds)}, prob {len(all_probs)}"
        )

    df = pd.DataFrame({
        "utt_id": utt_ids,
        "label": all_labels,
        "pred": all_preds,
    })
    for i, emo in enumerate(EMOTION_LABELS):
        df[f"p_{emo}"] = [round(float(p[i]), 6) for p in all_probs]

    # argmax(확률)과 저장된 pred가 어긋나면 둘 중 하나가 잘못 모인 것이다.
    prob_argmax = df[[f"p_{e}" for e in EMOTION_LABELS]].to_numpy().argmax(axis=1)
    n_mismatch = int((prob_argmax != df["pred"].to_numpy()).sum())
    if n_mismatch:
        # 소수점 6자리 반올림으로 1·2위가 동률이 되는 경우가 있어 경고만 남긴다.
        print(f"[pred] 경고: pred와 확률 argmax가 {n_mismatch}건 불일치 "
              f"(반올림 동률이면 무해)")

    out_path = out_dir / f"{name}.csv"
    df.to_csv(out_path, index=False)
    print(f"[pred] 발화별 예측 저장: {out_path}  ({n:,}건)")
    return out_path


def print_and_collect(all_labels: list, all_preds: list, title: str = "") -> dict:
    """지표를 계산해 화면에 출력하고, 파일로 저장할 dict를 반환한다."""
    acc = accuracy_score(all_labels, all_preds)
    w_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    m_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(EMOTION_LABELS))))
    report = classification_report(
        all_labels, all_preds, target_names=EMOTION_LABELS,
        zero_division=0, output_dict=True,
    )

    if title:
        print(f"\n=== {title} ===")
    print(f"Accuracy      : {acc:.4f}")
    print(f"Weighted F1   : {w_f1:.4f}")
    # macro F1은 클래스를 동등 취급 -> 소수 클래스(공포 등) 성능이 그대로 드러난다.
    # weighted만 보면 다수 클래스 덕에 좋아 보이는 착시가 생겨서 항상 같이 출력한다.
    print(f"Macro F1      : {m_f1:.4f}")
    print("\nConfusion Matrix (행=정답, 열=예측):")
    print("        " + " ".join(f"{l[:4]:>6}" for l in EMOTION_LABELS))
    for label, row in zip(EMOTION_LABELS, cm):
        print(f"{label[:6]:>8}" + " ".join(f"{v:>6}" for v in row))
    print("\n" + classification_report(all_labels, all_preds, target_names=EMOTION_LABELS, zero_division=0))

    return {
        "accuracy": round(float(acc), 6),
        "weighted_f1": round(float(w_f1), 6),
        "macro_f1": round(float(m_f1), 6),
        "n_samples": len(all_labels),
        "labels": EMOTION_LABELS,
        "confusion_matrix": cm.tolist(),
        "per_class": {
            k: v for k, v in report.items() if k in EMOTION_LABELS
        },
    }


def save_eval_result(
    metrics: dict, name: str, manifest: str, models: list[dict],
    out_dir: str | Path | None = None, extra: dict | None = None,
) -> Path:
    """평가 결과 1회를 JSON으로 저장한다.

    name    : 파일 이름에 쓸 식별자 (예: "v9", "ensemble_v6_v7_v8", "v7_swa")
    models  : [{"config": ..., "checkpoint": ...}, ...] — 재현에 필요한 설정
    """
    out_dir = Path(out_dir) if out_dir else DEFAULT_EVAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # hostname은 기록하지 않는다 — 재현에 필요한 것은 config·checkpoint·manifest이고,
    # 장비 이름은 저장소를 공개할 때 내부 호스트명만 노출한다.
    payload = {
        "name": name,
        "evaluated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "manifest": manifest,
        "models": models,
        **metrics,
    }
    if extra:
        payload.update(extra)

    out_path = out_dir / f"{name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[eval] 결과 저장: {out_path}")
    return out_path
