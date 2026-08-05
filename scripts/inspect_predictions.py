"""앙상블 구성 모델들이 개별 발화에 대해 실제로 어떤 확률을 내는지 들여다보는 도구.

evaluate_ensemble.py가 최종 정확도만 보여주는 것과 달리, 이 스크립트는 발화 하나하나에
대해 모델별 7클래스 확률 분포와 앙상블 평균을 나란히 출력한다. 용도:

1. 앙상블이 왜 개별 모델보다 나은지 눈으로 확인 (--mode rescued: 개별 모델은 틀렸는데
   평균내면 맞는 케이스만 골라 보여줌)
2. 모델이 어디서 헷갈리는지 확인 (--mode wrong)
3. 신뢰도 보정(calibration) 점검의 출발점 — 최대 확률값이 실제 정답률과 얼마나 맞는지
   (--summary가 확률 구간별 실제 정확도를 함께 출력한다)

실행 예:
    python scripts/inspect_predictions.py \
        --pairs configs/config_7class.yaml archived_runs/checkpoint_v6_best/best_model.pt \
                configs/config_bert_frozen.yaml archived_runs/checkpoint_v7_best/best_model.pt \
                configs/config_discriminative_lr.yaml archived_runs/checkpoint_v8_best/best_model.pt \
        --manifest data/manifests/test.csv --num 10 --mode rescued
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import load_config
from src.model import TrimodalEmotionModel
from src.datasets.manifest_dataset import ManifestEmotionDataset, make_collate_fn
from src.datasets.labels import EMOTION_LABELS, KOREAN_LABELS


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def fmt_row(label: str, probs, pred_idx: int, width: int = 9) -> str:
    cells = []
    for i, p in enumerate(probs):
        mark = "*" if i == pred_idx else " "
        cells.append(f"{p:>7.3f}{mark}")
    return f"{label:>10} " + "".join(cells)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", required=True,
                        help="'config1 ckpt1 config2 ckpt2 ...' 짝수 개")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--num", type=int, default=10, help="출력할 발화 개수")
    parser.add_argument("--max-batches", type=int, default=30,
                        help="훑어볼 배치 수 상한 (조건에 맞는 샘플을 찾기 위해 스캔하는 범위)")
    parser.add_argument(
        "--mode", choices=["first", "wrong", "disagree", "rescued"], default="first",
        help=("first=앞에서부터 / wrong=앙상블이 틀린 것 / disagree=모델 간 예측이 갈린 것 / "
              "rescued=개별 모델 과반이 틀렸는데 앙상블은 맞힌 것(앙상블 효과 확인용)"),
    )
    parser.add_argument("--summary", action="store_true",
                        help="확률 구간별 실제 정확도(신뢰도 보정 점검)도 함께 출력")
    args = parser.parse_args()

    if len(args.pairs) % 2 != 0:
        raise ValueError("--pairs는 'config checkpoint' 짝으로 짝수 개여야 함")
    combos = list(zip(args.pairs[0::2], args.pairs[1::2]))

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    first_cfg = load_config(Path(combos[0][0]))
    train_cfg = first_cfg.raw["train"]
    collate_fn = make_collate_fn(first_cfg.text_pretrained)
    ds = ManifestEmotionDataset(
        args.manifest, first_cfg,
        cache_dir=train_cfg.get("feature_cache_dir"),
        prosody_stats_path=train_cfg.get("prosody_stats_path"),
    )
    loader = DataLoader(ds, batch_size=train_cfg["batch_size"], shuffle=False, collate_fn=collate_fn,
                        num_workers=train_cfg.get("num_workers", 0), pin_memory=(device.type == "cuda"))

    models, names = [], []
    for cfg_path, ckpt_path in combos:
        cfg = load_config(Path(cfg_path))
        model = TrimodalEmotionModel(cfg).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        models.append(model)
        # checkpoint_v7_best -> v7 처럼 짧은 이름 추출(없으면 파일명 그대로)
        parts = [p for p in Path(ckpt_path).parts if p.startswith("checkpoint_v")]
        names.append(parts[0].replace("checkpoint_", "").replace("_best", "") if parts else Path(ckpt_path).stem)

    # 텍스트는 매니페스트에서 직접 읽어온다(데이터셋은 토큰만 들고 있어서 원문 복원이 어려움)
    utt_text = dict(zip(ds.df["utt_id"].astype(str), ds.df["text"].astype(str)))

    header = "           " + "".join(f"{KOREAN_LABELS[l]:>8}" for l in EMOTION_LABELS)
    shown = 0
    conf_bins = {}  # calibration 집계: (확률구간) -> [맞은 수, 전체 수]

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            if b_idx >= args.max_batches and not args.summary:
                break
            batch_d = move_batch_to_device(batch, device)
            all_probs = []
            for model in models:
                logits = model(
                    mel_spec=batch_d["mel_spec"], prosody_vec=batch_d["prosody_vec"],
                    frames=batch_d["frames"], input_ids=batch_d["input_ids"],
                    attention_mask=batch_d["attention_mask"],
                    audio_padding_mask=batch_d["audio_padding_mask"],
                    visual_padding_mask=batch_d["visual_padding_mask"],
                )
                all_probs.append(F.softmax(logits, dim=-1).cpu())
            stacked = torch.stack(all_probs, dim=0)   # [모델수, B, 7]
            ens = stacked.mean(dim=0)                 # [B, 7]
            labels = batch_d["labels"].cpu()

            for i in range(len(labels)):
                truth = labels[i].item()
                ens_pred = ens[i].argmax().item()
                indiv = [stacked[m, i].argmax().item() for m in range(len(models))]

                if args.summary:
                    conf = ens[i].max().item()
                    lo = int(conf * 10) / 10
                    hit, tot = conf_bins.get(lo, (0, 0))
                    conf_bins[lo] = (hit + int(ens_pred == truth), tot + 1)

                if shown >= args.num:
                    continue
                n_wrong = sum(1 for p in indiv if p != truth)
                if args.mode == "wrong" and ens_pred == truth:
                    continue
                if args.mode == "disagree" and len(set(indiv)) == 1:
                    continue
                if args.mode == "rescued" and not (ens_pred == truth and n_wrong > len(models) / 2):
                    continue

                utt_id = batch["utt_ids"][i]
                text = utt_text.get(utt_id, "?")
                print("=" * 78)
                print(f"[{utt_id}] \"{text}\"")
                print(f"정답: {KOREAN_LABELS[EMOTION_LABELS[truth]]}({EMOTION_LABELS[truth]})"
                      f"   앙상블 예측: {KOREAN_LABELS[EMOTION_LABELS[ens_pred]]}"
                      f"({EMOTION_LABELS[ens_pred]}) {'O' if ens_pred == truth else 'X'}")
                print(header)
                for m, nm in enumerate(names):
                    print(fmt_row(nm, stacked[m, i].tolist(), indiv[m]))
                print("  " + "-" * 76)
                print(fmt_row("평균", ens[i].tolist(), ens_pred))
                shown += 1

            if shown >= args.num and not args.summary:
                break

    if shown == 0:
        print(f"[inspect_predictions] --mode {args.mode} 조건에 맞는 샘플을 "
              f"{args.max_batches}배치 안에서 못 찾음. --max-batches를 늘려보세요.")

    if args.summary:
        print()
        print("=" * 78)
        print("신뢰도 보정(calibration) 점검 — 앙상블 최대 확률 구간별 실제 정답률")
        print("  이상적이라면 '확률 0.7~0.8이라고 한 것들의 실제 정답률'도 70~80%여야 한다.")
        print(f"{'확률 구간':>12} {'실제 정답률':>12} {'샘플 수':>10}")
        for lo in sorted(conf_bins):
            hit, tot = conf_bins[lo]
            print(f"  {lo:.1f} ~ {lo + 0.1:.1f}   {100 * hit / tot:>10.1f}%   {tot:>8}")


if __name__ == "__main__":
    main()
