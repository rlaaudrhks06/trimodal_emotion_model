"""체크포인트 앙상블 평가 스크립트. 통합기록 §11.1 A-2.

서로 다른 실험(예: v6/v7처럼 config가 다른 버전)의 체크포인트 여러 개를 각자의
config로 로드해 test set에 대해 forward한 뒤, softmax 확률을 평균해서 최종
예측을 만든다. 두 실험이 서로 다른 지점에서 틀리는 경향이 있다면 단일 모델보다
나을 수 있다는 가정을 검증한다.

실행 예 (v6 + v7 앙상블):
    python scripts/evaluate_ensemble.py \\
        --pairs configs/config_7class.yaml checkpoints_7class/best_model.pt \\
                configs/config_bert_frozen.yaml checkpoints_bert_frozen/best_model.pt \\
        --manifest data/manifests/test.csv
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
from src.eval_report import print_and_collect, save_eval_result
from scripts.train import move_batch_to_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs", nargs="+", required=True,
        help="'config1 checkpoint1 config2 checkpoint2 ...' 형태로 짝수 개 지정 (모델마다 자기 config로 로드)",
    )
    parser.add_argument("--manifest", type=str, default=None,
                        help="생략하면 config의 train.test_manifest를 쓴다 — 다른 분할로 학습한 모델을 "
                             "옛 test셋으로 잘못 평가하는 사고를 막기 위해 기본값을 config에 맞춘다.")
    parser.add_argument(
        "--save-as", type=str, default=None,
        help="결과를 results/eval/{이름}.json으로 저장 (예: ensemble_v6_v7_v8). 생략하면 화면 출력만.",
    )
    args = parser.parse_args()

    if len(args.pairs) % 2 != 0:
        raise ValueError("--pairs는 'config checkpoint' 짝으로 짝수 개여야 함")
    combos = list(zip(args.pairs[0::2], args.pairs[1::2]))

    # 하나라도 wav2vec2 백본이면 파형을 실어야 한다 — first_cfg만 보면
    # mel 모델이 앞에 올 때 w2v 모델이 파형을 못 받아 크래시한다.
    any_needs_wav = any(load_config(Path(c)).audio_backbone == "wav2vec2" for c, _ in combos)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    # 첫 config의 train 설정(batch_size, manifest 로딩 방식)을 데이터로더 구성에 재사용.
    # 모든 버전이 같은 klue/bert-base 토크나이저·같은 manifest 스키마를 쓰므로 안전.
    first_cfg = load_config(Path(combos[0][0]))
    train_cfg = first_cfg.raw["train"]
    if args.manifest is None:
        args.manifest = train_cfg["test_manifest"]
        print(f"[ensemble] --manifest 생략됨 -> 첫 config의 test_manifest 사용: {args.manifest}")
    collate_fn = make_collate_fn(first_cfg.text_pretrained)
    test_ds = ManifestEmotionDataset(
        args.manifest, first_cfg,
        cache_dir=train_cfg.get("feature_cache_dir"),
        prosody_stats_path=train_cfg.get("prosody_stats_path"),
        return_waveform=any_needs_wav,
    )
    test_loader = DataLoader(
        test_ds, batch_size=train_cfg["batch_size"], shuffle=False, collate_fn=collate_fn,
        num_workers=train_cfg.get("num_workers", 0), pin_memory=(device.type == "cuda"),
    )

    models = []
    for cfg_path, ckpt_path in combos:
        cfg = load_config(Path(cfg_path))
        model = TrimodalEmotionModel(cfg).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        models.append(model)
        print(f"[ensemble] 로드: {cfg_path} + {ckpt_path}")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
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
            # any_needs_wav로 파형을 싣고도 모델엔 안 넘기고 있었다 — w2v 모델이 섞이면
            # ValueError로 죽는다. 멜 전용 앙상블에선 "waveform" 키가 없어 그대로 지나간다.
            # 멜 모델에 파형을 넘겨도 안전하다: src/model.py:114의 use_w2v가 False면 무시한다.
            if "waveform" in batch:
                model_inputs["waveform"] = batch["waveform"]
                model_inputs["wav_attention_mask"] = batch["wav_attention_mask"]

            probs_sum = None
            for model in models:
                logits = model(**model_inputs)
                probs = F.softmax(logits, dim=-1)
                probs_sum = probs if probs_sum is None else probs_sum + probs
            preds = (probs_sum / len(models)).argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

    metrics = print_and_collect(all_labels, all_preds, title=f"앙상블 {len(models)}개 모델")

    if args.save_as:
        save_eval_result(
            metrics, name=args.save_as, manifest=args.manifest,
            models=[{"config": c, "checkpoint": k} for c, k in combos],
            extra={"ensemble_size": len(models), "method": "softmax 확률 산술평균"},
        )


if __name__ == "__main__":
    main()
