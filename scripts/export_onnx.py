"""v11을 ONNX로 내보내고, **같은 스크립트에서 PyTorch와 같은 답을 내는지 검증한다.**

젯슨 배포 경로의 첫 칸이다(11.2절 1번). TensorRT는 ONNX를 받아 빌드하므로 순서가
ONNX -> TensorRT -> FP16/INT8이고, 이 중 ONNX 단계만 GPU 없이 할 수 있다.

**왜 export와 검증을 한 파일에 두는가.** 이 프로젝트에서 가장 비싼 실패는 전부
"조용히 틀리는" 것이었다(얼굴 크롭이 전신, 옛 test 매니페스트, 패딩 포함 정규화).
ONNX 변환은 그 유형이 나오기 딱 좋은 자리다 — 그래프가 잘못 나와도 파일은 만들어지고
추론도 되고 그럴듯한 확률이 나온다. 그래서 내보내기만 하는 명령을 두지 않았다.
**검증을 통과하지 못하면 파일을 남기지 않는다.**

보행 트랙이 같은 것을 먼저 했다 — 학습·실행 동등성을 200스텝 전 구간에서 확인해
"행동 차이 0.0"을 얻었다(제안서 3.1.4). 여기서도 판정하는 것은 로짓의 소수점이
아니라 **로봇이 실제로 쓰는 결정**이다: 7클래스 argmax, 3클래스 판정, 응답/보류.

실행:
    python scripts/export_onnx.py --config configs/config_si_w2v.yaml \
        --checkpoint checkpoints/v11_best.pt --out checkpoints/onnx/v11.onnx
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config                                      # noqa: E402
from src.datasets.labels import EMOTION_LABELS                          # noqa: E402
from src.model import TrimodalEmotionModel                              # noqa: E402

# robot/brain/engine.py의 것과 같아야 한다. 사본을 만들면 한쪽만 고쳐진다는 것을
# 이 프로젝트에서 여러 번 겪었으므로(제안서 3.3.2) 정본에서 가져온다.
sys.path.insert(0, str(ROOT / "robot"))
from robot.brain.engine import COARSE, TEMPERATURE_DEFAULT, THRESHOLD_DEFAULT  # noqa: E402


class ExportWrapper(nn.Module):
    """ONNX 입력을 **배치 1 · 패딩 없음**(=실사용 조건)으로 좁힌 래퍼.

    로봇은 발화 하나를 그때그때 처리한다. 배치도 패딩도 없으므로 마스크 인자들이
    전부 "모두 유효"가 되고, 그러면 다음 셋이 수학적으로 None과 같아진다.

      - `wav_attention_mask` (전부 1) : `_normalize`가 `ones_like`를 쓰는 것과 동일
      - `audio_padding_mask`          : `frame_padding_mask`의 lens가 프레임 수와 같아
                                        전부 False(=유효)가 된다
      - `visual_padding_mask` (전부 False)

    그래서 그래프에서 뺐다. 이득이 두 개다. 입력이 8개에서 5개로 줄고, wav2vec2의
    출력 길이 계산(`_get_feat_extract_output_lengths`, 정수 나눗셈)이 그래프에서
    사라진다 — 동적 길이와 정수 연산이 만나는 자리가 변환이 깨지기 가장 쉬운 곳이다.

    **"같아진다"를 가정으로 두지 않는다.** 아래 검증은 PyTorch 쪽에 마스크를 전부
    주고(=engine.py가 실제로 하는 그대로) 돌린 결과와 비교한다. 두 결과가 맞으면
    이 단순화가 옳다는 것이 측정으로 확인된 것이고, 틀리면 스크립트가 실패한다.

    `mel_spec`은 wav2vec2 경로에서 쓰이지 않지만 forward 시그니처가 요구하므로
    자리표시자를 만들어 넘긴다. `_maybe_drop_modalities`는 eval에서 즉시 반환하므로
    값이 무엇이든 결과에 닿지 않는다.
    """

    def __init__(self, model: TrimodalEmotionModel, n_mels: int):
        super().__init__()
        self.model = model
        self.n_mels = n_mels

    def forward(
        self,
        waveform: torch.Tensor,       # [1, T_samples] 16kHz, -1~1
        frames: torch.Tensor,         # [1, T_v, 3, 112, 112] 0~1
        input_ids: torch.Tensor,      # [1, T_t]
        attention_mask: torch.Tensor,  # [1, T_t] 1=유효
        prosody_vec: torch.Tensor,    # [1, 10] 정규화된 운율
    ) -> torch.Tensor:
        mel = waveform.new_zeros((waveform.size(0), 1, self.n_mels))
        return self.model(
            mel_spec=mel,
            prosody_vec=prosody_vec,
            frames=frames,
            input_ids=input_ids,
            attention_mask=attention_mask,
            waveform=waveform,
        )


def decide(logits: np.ndarray, temperature: float, threshold: float) -> dict:
    """로짓 -> 로봇이 실제로 쓰는 결정. `engine.py`의 판정과 같은 순서다.

    비교 대상을 로짓이 아니라 이것으로 두는 이유: 젯슨에 올라가는 것은 확률이 아니라
    "다가갈지·물러날지·가만히 있을지"다(제안서 3.3.1). 로짓 소수점 여섯째 자리가
    달라도 결정이 같으면 배포에는 지장이 없고, 반대로 결정이 갈리면 소수점이 아무리
    가까워도 실패다.
    """
    z = logits.astype(np.float64) / temperature
    z = z - z.max()
    p = np.exp(z) / np.exp(z).sum()

    i = int(p.argmax())
    coarse_probs: dict[str, float] = {}
    for j, e in enumerate(EMOTION_LABELS):
        coarse_probs[COARSE[e]] = coarse_probs.get(COARSE[e], 0.0) + float(p[j])
    top_coarse = max(coarse_probs, key=coarse_probs.get)
    return {
        "emotion": EMOTION_LABELS[i],
        "coarse": top_coarse,
        # engine.py의 기본값이 decide_on="coarse"이므로 응답 판정도 3클래스 합으로 한다.
        "answered": coarse_probs[top_coarse] >= threshold,
        "probs": p,
    }


def real_samples(cfg, n: int) -> list[dict]:
    """test 매니페스트에서 n건을 실제 파이프라인으로 만든다.

    합성 텐서만으로 검증하면 "그래프가 도는가"는 알 수 있어도 실제 값 분포에서
    같은 답을 내는지는 모른다. 이 프로젝트의 규칙이기도 하다 — 고쳤다고 말하려면
    실제 데이터로 끝까지 돌려본다(이슈기록 13장).

    특징 캐시는 쓰지 않는다(`cache_dir=None`). 로컬에 캐시가 없고, 3건 때문에
    캐시 디렉터리를 만들어 두면 나중에 "캐시가 있다"고 오해할 여지가 생긴다.
    발화당 약 0.8초 걸린다(librosa.pyin).
    """
    from torch.utils.data import DataLoader

    from src.datasets.manifest_dataset import ManifestEmotionDataset, make_collate_fn

    train_cfg = cfg.raw["train"]
    manifest = ROOT / train_cfg["test_manifest"]
    if not manifest.exists():
        return []

    ds = ManifestEmotionDataset(
        str(manifest), cfg, cache_dir=None,
        prosody_stats_path=train_cfg.get("prosody_stats_path"),
        return_waveform=(cfg.audio_backbone == "wav2vec2"),
    )
    # 배치 1로 뽑는다 — 패딩이 생기면 검증 조건(패딩 없음)이 깨진다.
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=make_collate_fn(cfg.text_pretrained))
    out = []
    for batch in loader:
        out.append(batch)
        if len(out) >= n:
            break
    return out


def synthetic_samples(cfg, seed: int = 2026) -> list[dict]:
    """길이 극단값을 합성으로 만든다.

    실제 3건은 우연히 비슷한 길이일 수 있다. 동적 축(발화 길이·얼굴 프레임 수·토큰 수)이
    정말 동적으로 나왔는지는 **범위의 양 끝**을 넣어봐야 갈린다. 데모 실측 범위가
    발화 1.2~8.4초 / 얼굴 17~210프레임이었고(제안서 3.1.4), 프레임은 전처리가
    32로 재샘플링하므로 모델이 보는 것은 최대 32장이다.
    """
    g = torch.Generator().manual_seed(seed)
    sr = cfg.audio_sample_rate
    cases = [
        ("짧은 발화", int(1.2 * sr), 4, 5),
        ("긴 발화", int(8.0 * sr), 32, 64),
    ]
    out = []
    for name, n_wav, n_frames, n_tok in cases:
        # 파형은 정규화(zero-mean/unit-var)를 거치므로 스케일 자체는 결과에 안 남는다.
        wav = torch.randn(1, n_wav, generator=g) * 0.1
        ids = torch.randint(1000, 30000, (1, n_tok), generator=g)
        ids[0, 0], ids[0, -1] = 2, 3  # [CLS]/[SEP] 자리 — 값 범위만 현실적으로
        out.append({
            "_name": name,
            "waveform": wav,
            "wav_attention_mask": torch.ones(1, n_wav, dtype=torch.long),
            "frames": torch.rand(1, n_frames, 3, 112, 112, generator=g),
            "input_ids": ids,
            "attention_mask": torch.ones(1, n_tok, dtype=torch.long),
            "prosody_vec": torch.randn(1, cfg.model.prosody_dim, generator=g),
            "mel_spec": torch.zeros(1, 1, cfg.audio_n_mels),
        })
    return out


def torch_reference(model: TrimodalEmotionModel, s: dict) -> np.ndarray:
    """PyTorch 쪽은 **마스크를 전부 준다** — `engine.py`가 하는 호출 그대로.

    래퍼가 마스크를 빼도 되는지를 여기서 판정하게 되므로, 이쪽을 engine과 다르게
    부르면 검증 자체가 무의미해진다.
    """
    n_frames = s["frames"].size(1)
    with torch.no_grad():
        return model(
            mel_spec=s["mel_spec"],
            prosody_vec=s["prosody_vec"],
            frames=s["frames"],
            input_ids=s["input_ids"],
            attention_mask=s["attention_mask"],
            visual_padding_mask=torch.zeros(1, n_frames, dtype=torch.bool),
            waveform=s["waveform"],
            wav_attention_mask=s["wav_attention_mask"],
        ).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_si_w2v.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/v11_best.pt")
    ap.add_argument("--out", default="checkpoints/onnx/v11.onnx")
    ap.add_argument("--audio-pretrained", default=None,
                    help="config의 백본 경로를 대체한다(맥에서 HF 허브 id를 쓸 때). "
                         "engine.py의 같은 옵션과 목적이 같다.")
    ap.add_argument("--real", type=int, default=3,
                    help="실제 test 발화 몇 건으로 검증할지. 0이면 합성만.")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="확률 벡터의 최대 절대차 허용값. 결정(argmax·3클래스·응답)은 "
                         "허용값과 무관하게 완전 일치를 요구한다.")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    if args.audio_pretrained:
        cfg.audio_pretrained = args.audio_pretrained
        if isinstance(cfg.raw.get("audio"), dict):
            cfg.raw["audio"]["pretrained_model"] = args.audio_pretrained

    print(f"[export] 모델 로딩 — {args.checkpoint}")
    model = TrimodalEmotionModel(cfg).eval()
    model.load_state_dict(torch.load(ROOT / args.checkpoint, map_location="cpu"))

    # ---- 검증 표본을 **먼저** 만든다. export가 수십 초 걸리는데 그 뒤에 데이터가
    #      없어서 실패하면 그 시간을 통째로 버린다(engine.py가 같은 이유로 준비물
    #      검사를 모델 로딩 앞에 둔다).
    samples = synthetic_samples(cfg)
    if args.real > 0:
        got = real_samples(cfg, args.real)
        if not got:
            print(f"[export] test 매니페스트가 없어 실데이터 검증을 건너뛴다 "
                  f"— 합성 {len(samples)}건만으로 판정한다")
        for k, b in enumerate(got):
            b["_name"] = f"실데이터 {k + 1}"
            samples.append(b)

    out_path = ROOT / args.out
    # 임시 디렉터리는 최종 디렉터리의 **형제**여야 한다. 안쪽에 두면 통과 후
    # 최종 디렉터리를 비우는 순간 임시본까지 같이 지워진다(실제로 그렇게 한 번 깨졌다).
    tmp_dir = out_path.parent.parent / f"{out_path.parent.name}__tmp"
    # 검증을 통과하기 전에는 최종 경로에 아무것도 두지 않는다. 반쯤 맞는 파일이
    # 남아 있으면 다음 사람이 그것을 젯슨에 올린다.
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / out_path.name

    wrapper = ExportWrapper(model, cfg.audio_n_mels).eval()
    ex = samples[0]
    example = (ex["waveform"], ex["frames"], ex["input_ids"],
               ex["attention_mask"], ex["prosody_vec"])

    # 동적 축. min이 2 이상인 것은 torch.export가 0·1을 상수로 특수화하기 때문이다.
    dim = torch.export.Dim
    t_wav = dim("t_wav", min=4000, max=int(8.0 * cfg.audio_sample_rate))
    t_v = dim("t_v", min=2, max=32)
    t_t = dim("t_t", min=3, max=64)

    print(f"[export] ONNX 변환 — 동적 축 파형/프레임/토큰, 배치 1 고정")
    torch.onnx.export(
        wrapper, example, str(tmp_path),
        input_names=["waveform", "frames", "input_ids", "attention_mask", "prosody_vec"],
        output_names=["logits"],
        dynamic_shapes=({1: t_wav}, {1: t_v}, {1: t_t}, {1: t_t}, None),
        external_data=True,  # fp32 1.67GB — 단일 protobuf 2GB 한계에 붙는다
    )

    import onnxruntime as ort

    sess = ort.InferenceSession(str(tmp_path), providers=["CPUExecutionProvider"])
    print(f"[export] onnxruntime 세션 생성 완료 — 검증 {len(samples)}건\n")

    worst = 0.0
    failures = []
    for s in samples:
        ref = torch_reference(model, s)
        got = sess.run(["logits"], {
            "waveform": s["waveform"].numpy(),
            "frames": s["frames"].numpy(),
            "input_ids": s["input_ids"].numpy().astype(np.int64),
            "attention_mask": s["attention_mask"].numpy().astype(np.int64),
            "prosody_vec": s["prosody_vec"].numpy(),
        })[0]

        a = decide(ref[0], TEMPERATURE_DEFAULT, THRESHOLD_DEFAULT)
        b = decide(got[0], TEMPERATURE_DEFAULT, THRESHOLD_DEFAULT)
        d = float(np.abs(a["probs"] - b["probs"]).max())
        worst = max(worst, d)

        same = (a["emotion"] == b["emotion"] and a["coarse"] == b["coarse"]
                and a["answered"] == b["answered"])
        mark = "✅" if (same and d <= args.tol) else "❌"
        print(f"  {mark} {s['_name']:<12} 파형 {s['waveform'].size(1):>6} · "
              f"프레임 {s['frames'].size(1):>2} · 토큰 {s['input_ids'].size(1):>2} | "
              f"{a['coarse']}/{a['emotion']} "
              f"{'응답' if a['answered'] else '보류'} | 확률 최대차 {d:.2e}")
        if not same:
            failures.append(f"{s['_name']}: 결정 불일치 "
                            f"torch={a['coarse']}/{a['emotion']}/{a['answered']} "
                            f"onnx={b['coarse']}/{b['emotion']}/{b['answered']}")
        elif d > args.tol:
            failures.append(f"{s['_name']}: 확률 최대차 {d:.2e} > 허용 {args.tol:.0e}")

    print()
    if failures:
        print("[export] ❌ 검증 실패 — 파일을 남기지 않는다")
        for f in failures:
            print(f"  - {f}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return 1

    # 통과했으므로 최종 경로로 옮긴다. 외부 데이터 파일이 함께 있으므로 디렉터리째 옮긴다.
    final_dir = out_path.parent
    shutil.rmtree(final_dir, ignore_errors=True)
    tmp_dir.rename(final_dir)
    total = sum(p.stat().st_size for p in final_dir.rglob("*") if p.is_file())
    print(f"[export] ✅ 검증 통과 — 결정 {len(samples)}/{len(samples)} 일치, "
          f"확률 최대차 {worst:.2e}")
    print(f"[export] 저장: {out_path.relative_to(ROOT)}  ({total / 1e9:.2f}GB)")
    print(f"[export] 다음 칸은 젯슨에서 한다 — trtexec로 FP16 엔진 빌드 후 같은 "
          f"검증을 보드에서 반복할 것(11.2절 1~2번)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
