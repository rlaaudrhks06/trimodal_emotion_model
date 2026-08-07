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
from src.eval_report import print_and_collect, save_eval_result, save_predictions
from scripts.train import move_batch_to_device

MODALITIES = ("audio", "visual", "text")


def zero_modalities(model_inputs: dict, drop: tuple) -> dict:
    """평가 시 특정 모달리티를 결정적으로 0으로 만든다(모달리티 애블레이션).

    **학습 시 모달리티 드롭아웃(`src/model.py:_maybe_drop_modalities`)과 정확히 같은
    방식으로 지운다.** 다르게 지우면 모델이 한 번도 본 적 없는 입력이 되어, 재는 것이
    "그 브랜치의 기여도"가 아니라 "낯선 입력에 대한 반응"이 되어버린다.

    - audio  : mel_spec + prosody_vec + waveform 셋 다 (설계상 오디오는 이 셋의 합)
    - visual : frames
    - text   : attention_mask를 0으로, 단 첫 토큰만 1 (BERT류는 유효 토큰 최소 1개 필요)
    """
    if not drop:
        return model_inputs
    out = dict(model_inputs)
    if "audio" in drop:
        out["mel_spec"] = torch.zeros_like(out["mel_spec"])
        out["prosody_vec"] = torch.zeros_like(out["prosody_vec"])
        if "waveform" in out:
            out["waveform"] = torch.zeros_like(out["waveform"])
    if "visual" in drop:
        out["frames"] = torch.zeros_like(out["frames"])
    if "text" in drop:
        am = torch.zeros_like(out["attention_mask"])
        am[:, 0] = 1
        out["attention_mask"] = am
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent.parent / "configs" / "config.yaml"))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest", type=str, default=None,
                        help="생략하면 config의 train.test_manifest를 쓴다 — 다른 분할로 학습한 모델을 "
                             "옛 test셋으로 잘못 평가하는 사고를 막기 위해 기본값을 config에 맞춘다.")
    parser.add_argument(
        "--modality", choices=["audio", "visual", "text"], default=None,
        help="지정하면 SingleModalityModel(베이스라인 체크포인트)을 평가. 생략하면 트리모달 본 모델.",
    )
    parser.add_argument(
        "--save-as", type=str, default=None,
        help="결과를 results/eval/{이름}.json으로 저장 (예: v9, v7_swa). 생략하면 화면 출력만.",
    )
    parser.add_argument(
        "--drop-modality", action="append", choices=list(MODALITIES), default=None,
        help="지정한 모달리티를 0으로 만들고 평가(모달리티 애블레이션). 여러 번 줄 수 있다. "
             "학습 시 모달리티 드롭아웃과 같은 방식으로 지운다.",
    )
    parser.add_argument(
        "--save-predictions", action="store_true",
        help="발화별 정답·예측·확률을 results/predictions/{--save-as 이름}.csv로 저장. "
             "짝지어 검정(McNemar)·신뢰도 보정·부분집합 분석에 필요하다.",
    )
    args = parser.parse_args()
    drop = tuple(sorted(set(args.drop_modality or ())))
    if args.save_predictions and not args.save_as:
        parser.error("--save-predictions는 파일 이름이 필요하므로 --save-as와 함께 써야 한다")

    cfg = load_config(Path(args.config))
    train_cfg = cfg.raw["train"]
    if args.manifest is None:
        args.manifest = train_cfg["test_manifest"]
        print(f"[evaluate] --manifest 생략됨 -> config의 test_manifest 사용: {args.manifest}")
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    num_workers = train_cfg.get("num_workers", 0)
    cache_dir = train_cfg.get("feature_cache_dir")
    prosody_stats_path = train_cfg.get("prosody_stats_path")
    collate_fn = make_collate_fn(cfg.text_pretrained)
    test_ds = ManifestEmotionDataset(args.manifest, cfg, cache_dir=cache_dir,
                                     prosody_stats_path=prosody_stats_path,
                                     return_waveform=(cfg.audio_backbone == "wav2vec2"))
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

    if drop:
        print(f"[evaluate] 모달리티 애블레이션 — 0으로 만들 것: {', '.join(drop)}")

    all_preds, all_labels, all_probs = [], [], []
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
            # 위에서 return_waveform=True로 파형을 싣고도 모델엔 안 넘기고 있었다 —
            # wav2vec2 백본은 파형이 없으면 ValueError를 던지므로(src/model.py:115)
            # v11 평가가 아예 불가능한 상태였다. 멜 경로에선 batch에 "waveform" 키
            # 자체가 없어 이 분기를 그냥 지나가므로 v1~v10 결과는 영향받지 않는다.
            # 조건 분기 형태는 scripts/train.py:112-114에서 이미 검증된 것을 그대로 쓴다.
            if "waveform" in batch:
                model_inputs["waveform"] = batch["waveform"]
                model_inputs["wav_attention_mask"] = batch["wav_attention_mask"]
            model_inputs = zero_modalities(model_inputs, drop)
            logits = model(**model_inputs)
            # 예측은 기존 그대로 로짓의 argmax로 뽑는다 — softmax는 단조라 결과가
            # 같지만, 지표 산출 경로를 건드리지 않기 위해 확률은 따로 계산한다.
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_probs.extend(torch.softmax(logits.float(), dim=-1).cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

    metrics = print_and_collect(all_labels, all_preds)

    if args.save_predictions:
        # shuffle=False라 DataLoader가 도는 순서가 매니페스트 행 순서와 같다
        # (ManifestEmotionDataset.__getitem__이 self.df.iloc[idx]를 그대로 쓴다).
        # 그래서 데이터셋을 건드리지 않고 여기서 utt_id를 붙일 수 있다.
        save_predictions(test_ds.df["utt_id"].astype(str).tolist(),
                         all_labels, all_preds, all_probs, name=args.save_as)

    if args.save_as:
        extra = {}
        if args.modality:
            extra["modality"] = args.modality
        if drop:
            extra["dropped_modalities"] = list(drop)
        save_eval_result(
            metrics, name=args.save_as, manifest=args.manifest,
            models=[{"config": args.config, "checkpoint": args.checkpoint}],
            extra=extra or None,
        )


if __name__ == "__main__":
    main()
