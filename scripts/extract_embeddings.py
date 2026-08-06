"""학습된 모델의 내부 표현(임베딩)을 발화별로 뽑아 파일로 저장한다.

배경: v11의 train_acc 60.6% vs test 46.2%(격차 14.4%p)가 "화자를 외운 탓"인지
추측만 하고 있었다. 추측을 숫자로 바꾸려면 모델이 만든 512차원 벡터 자체를
꺼내봐야 하는데, forward()는 이 벡터들을 계산하고 곧바로 분류기에 넣어 버린다.

**src/model.py를 한 줄도 고치지 않는다.** 학습이 끝난 모델에서 값만 꺼내려고
검증된 학습 경로를 건드리는 건 위험 대비 이득이 없다. 대신 PyTorch forward hook을
쓴다 — 모듈의 입출력을 바깥에서 가로채는 공식 기능이라 모델 동작에 영향이 없다.

뽑는 벡터 4종 (전부 512차원):
    z_v  : 시각 브랜치 최종     (교차어텐션 + 평균풀링 결합)
    z_a  : 오디오 브랜치 최종   (운율 게이트까지 통과한 것)
    z_t  : 텍스트 브랜치 최종
    h    : 세 개를 합쳐 MLP 1층 + GELU를 거친 것 = y=Wx+b의 x 그 자체

브랜치별로 따로 뽑는 이유가 핵심이다. "영상 브랜치가 감정이 아니라 얼굴 인식을
하고 있는가?"처럼 모달리티별로 물어봐야 원인이 좁혀진다. h만 뽑으면 셋이 이미
섞여 있어서 어느 브랜치가 문제인지 알 수 없다.

person_id는 utt_id("{clip_id}_{person_id}_{start}_{end}")에서 파싱한다 —
매니페스트에 컬럼을 추가할 필요가 없다(8.14절 split_manifest.py와 같은 방식).

실행 예:
    python scripts/extract_embeddings.py \
        --config configs/config_si_w2v.yaml \
        --checkpoint archived_runs/checkpoint_v11_best/best_model.pt \
        --out results/embeddings/v11_test.npz

TensorFlow Embedding Projector(projector.tensorflow.org)용 TSV도 같이 나온다.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.model import TrimodalEmotionModel
from src.datasets.manifest_dataset import ManifestEmotionDataset, make_collate_fn
from src.datasets.labels import EMOTION_LABELS
from scripts.train import move_batch_to_device
from scripts.split_manifest import parse_utt_id

VECTOR_NAMES = ["z_v", "z_a", "z_t", "h"]


class EmbeddingCollector:
    """forward hook으로 분류기 앞단 벡터들을 배치마다 받아 모아둔다.

    hook 두 개를 건다:
      1) classifier에 forward_pre_hook  -> 입력 (z_v, z_a, z_t)를 가로챈다.
         src/model.py:144가 위치인자로 호출하므로 args 튜플에 셋 다 들어온다.
      2) classifier.mlp[1](GELU)에 forward_hook -> 출력 h를 가로챈다.
         mlp = [Linear(1536,512), GELU, Dropout, Linear(512,7)] 구조이고,
         마지막 Linear의 입력이 곧 h다(eval 모드라 Dropout은 항등).
    """

    def __init__(self, model: TrimodalEmotionModel):
        self.buf = {k: [] for k in VECTOR_NAMES}
        self._pending = {}
        self._handles = [
            model.classifier.register_forward_pre_hook(self._on_classifier_input),
            model.classifier.mlp[1].register_forward_hook(self._on_hidden),
        ]

    def _on_classifier_input(self, module, args):
        z_v, z_a, z_t = args
        self._pending["z_v"] = z_v.detach().cpu()
        self._pending["z_a"] = z_a.detach().cpu()
        self._pending["z_t"] = z_t.detach().cpu()

    def _on_hidden(self, module, args, output):
        self._pending["h"] = output.detach().cpu()

    def flush(self):
        """배치 하나가 끝난 뒤 호출 — 모인 것을 확정하고 다음 배치를 준비한다."""
        missing = [k for k in VECTOR_NAMES if k not in self._pending]
        if missing:
            raise RuntimeError(f"hook이 {missing}를 못 잡았다 — 모델 구조가 바뀌었는지 확인할 것")
        for k in VECTOR_NAMES:
            self.buf[k].append(self._pending[k])
        self._pending = {}

    def stacked(self) -> dict[str, np.ndarray]:
        return {k: torch.cat(v, dim=0).numpy() for k, v in self.buf.items()}

    def close(self):
        for h in self._handles:
            h.remove()


def write_projector_tsv(vectors: np.ndarray, meta: dict, out_prefix: Path) -> None:
    """TensorFlow Embedding Projector가 그대로 읽는 TSV 두 장을 쓴다.

    projector.tensorflow.org에서 'Load' -> vectors.tsv(좌표) + metadata.tsv(색/라벨).
    metadata는 헤더가 필요하고, vectors는 헤더 없이 값만 있어야 한다.
    """
    vec_path = out_prefix.with_name(out_prefix.stem + "_vectors.tsv")
    meta_path = out_prefix.with_name(out_prefix.stem + "_metadata.tsv")

    np.savetxt(vec_path, vectors, delimiter="\t", fmt="%.6f")

    keys = list(meta.keys())
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("\t".join(keys) + "\n")
        for row in zip(*(meta[k] for k in keys)):
            f.write("\t".join(str(v) for v in row) + "\n")

    print(f"[extract] Projector용 TSV: {vec_path}")
    print(f"[extract]                  {meta_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=None,
                        help="생략하면 config의 test_manifest — 다른 분할로 학습한 모델을 "
                             "옛 test셋으로 뽑는 사고를 막기 위한 기본값(evaluate.py와 동일 규칙)")
    parser.add_argument("--out", required=True, help="저장할 .npz 경로")
    parser.add_argument("--tsv-vector", choices=VECTOR_NAMES + ["none"], default="h",
                        help="Projector용 TSV로 내보낼 벡터 (기본 h = 최종 분류 직전 표현)")
    parser.add_argument("--limit", type=int, default=None, help="테스트용: 앞의 N개 발화만")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    train_cfg = cfg.raw["train"]
    manifest = args.manifest or train_cfg["test_manifest"]
    if args.manifest is None:
        print(f"[extract] --manifest 생략됨 -> config의 test_manifest 사용: {manifest}")

    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))

    ds = ManifestEmotionDataset(
        manifest, cfg,
        cache_dir=train_cfg.get("feature_cache_dir"),
        prosody_stats_path=train_cfg.get("prosody_stats_path"),
        return_waveform=(cfg.audio_backbone == "wav2vec2"),
    )
    if args.limit:
        ds.df = ds.df.iloc[: args.limit].reset_index(drop=True)
    loader = DataLoader(
        ds, batch_size=train_cfg["batch_size"], shuffle=False,
        collate_fn=make_collate_fn(cfg.text_pretrained),
        num_workers=train_cfg.get("num_workers", 0), pin_memory=(device.type == "cuda"),
    )

    model = TrimodalEmotionModel(cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    collector = EmbeddingCollector(model)
    utt_ids, labels, preds = [], [], []
    try:
        with torch.no_grad():
            for i, batch in enumerate(loader, 1):
                utt_ids.extend(batch["utt_ids"])
                batch = move_batch_to_device(batch, device)
                model_inputs = dict(
                    mel_spec=batch["mel_spec"], prosody_vec=batch["prosody_vec"],
                    frames=batch["frames"], input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    audio_padding_mask=batch["audio_padding_mask"],
                    visual_padding_mask=batch["visual_padding_mask"],
                )
                if "waveform" in batch:
                    model_inputs["waveform"] = batch["waveform"]
                    model_inputs["wav_attention_mask"] = batch["wav_attention_mask"]
                logits = model(**model_inputs)
                collector.flush()
                preds.extend(logits.argmax(dim=-1).cpu().tolist())
                labels.extend(batch["labels"].cpu().tolist())
                if i % 20 == 0:
                    print(f"    {i}/{len(loader)} 배치", flush=True)
        vecs = collector.stacked()
    finally:
        collector.close()  # hook은 반드시 떼어낸다 — 남겨두면 이후 forward마다 메모리가 쌓인다

    person_ids = [parse_utt_id(u)[1] for u in utt_ids]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, **vecs,
        utt_ids=np.array(utt_ids), person_ids=np.array(person_ids),
        labels=np.array(labels), preds=np.array(preds),
        label_names=np.array(EMOTION_LABELS),
    )

    n = len(utt_ids)
    print(f"\n[extract] {n:,}개 발화, 화자 {len(set(person_ids)):,}명 -> {out}")
    for k in VECTOR_NAMES:
        print(f"    {k}: {vecs[k].shape}")

    if args.tsv_vector != "none":
        write_projector_tsv(
            vecs[args.tsv_vector],
            {
                "utt_id": utt_ids,
                "emotion": [EMOTION_LABELS[i] for i in labels],
                "predicted": [EMOTION_LABELS[i] for i in preds],
                "correct": ["O" if a == b else "X" for a, b in zip(labels, preds)],
                "person_id": person_ids,
            },
            out,
        )


if __name__ == "__main__":
    main()
