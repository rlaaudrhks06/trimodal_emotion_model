"""세 갈래 백본 후보를 **동결 특징 + 선형 프로브**로 한 조건에서 비교한다 (13.19절).

## 왜

지금 백본 셋 중 감정 전용은 얼굴(MobileFaceNet)뿐이고, 소리·글은 범용 모델이다.
"시중 최고 감정 모델을 꽂으면 어디까지 가는가"를 재려면 전체 재학습(13시간×3시드) 전에
싸게 걸러야 한다 — 백본 7개를 1시간에 거른 scripts/probe_audio_backbones.py와 같은 절차다.

## 방법 — 배포 조건과 같게

train 부분집합(기본 8,000)으로 로지스틱 회귀를 맞추고 **val 전체(다른 화자)로 채점**한다.
같은 분할·같은 분류기·같은 표준화를 모든 후보에 쓴다. 비교되는 것은 **특징의 질**뿐이다.

## 한계 (반드시 같이 읽을 것)

프로브는 성능을 예측하지만 **수치 안정성은 못 본다** — 24층이 프로브에서 +4.8이었는데
실제 학습은 −2.3에 NaN이었다(13.10.3절, fp16 상한). 최종 판정은 반드시 실제 학습으로 한다.

    python scripts/probe_backbones3.py --modality audio --candidates xlsr12 emotion2vec
    python scripts/probe_backbones3.py --modality text  --candidates kluebert kluebert_emotion
    python scripts/probe_backbones3.py --modality face  --candidates mobilefacenet vit_fer
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import load_config  # noqa: E402
from src.datasets.labels import EMOTION_LABELS, normalize_label  # noqa: E402

# 후보 정의: (설명, 로더 키)
CANDS = {
    "audio": {
        "xlsr12": "facebook/wav2vec2-large-xlsr-53 12층 (현재 쓰는 것)",
        "emotion2vec": "emotion2vec_plus_large (감정 전용 자기지도, FunASR)",
    },
    "text": {
        "kluebert": "klue/bert-base (현재 쓰는 것)",
        "kluebert_emotion": "dlckdfuf141/korean-emotion-kluebert-v2 (한국어 감정 미세조정)",
        "roberta_emotion": "Seonghaa/korean-emotion-classifier-roberta",
        "kcelectra_emotion": "GGARA02/kcelectra-korean-emotion",
    },
    "face": {
        "mobilefacenet": "emotiefflib mbf_va_mtl (현재 쓰는 것)",
        "vit_fer": "trpakov/vit-face-expression (ViT, 표정 미세조정)",
        "beit_fer": "Tanneru/...BEIT-Large (FER+RAF-DB+AffectNet)",
    },
}
HF = {
    "kluebert": "klue/bert-base",
    "kluebert_emotion": "dlckdfuf141/korean-emotion-kluebert-v2",
    "roberta_emotion": "Seonghaa/korean-emotion-classifier-roberta",
    "kcelectra_emotion": "GGARA02/kcelectra-korean-emotion",
    "vit_fer": "trpakov/vit-face-expression",
    "beit_fer": "Tanneru/Facial-Emotion-Detection-FER-RAFDB-AffectNet-BEIT-Large",
}
SR = 16000


def pick(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    if n and n < len(df):
        df = df.sample(n=n, random_state=seed)
    return df.reset_index(drop=True)


def labels_of(df: pd.DataFrame) -> np.ndarray:
    y = df["label"].astype(str).str.strip().str.lower().map(normalize_label)
    return y.map({c: i for i, c in enumerate(EMOTION_LABELS)}).to_numpy()


# ── 갈래별 특징 추출 ────────────────────────────────────────────────────────
def feats_audio(name, df, dev, bs=16, max_sec=8.0):
    import soundfile as sf
    if name == "xlsr12":
        from transformers import AutoModel
        m = AutoModel.from_pretrained("facebook/wav2vec2-large-xlsr-53").to(dev).eval()
        out = []
        for i in range(0, len(df), bs):
            wavs = []
            for p in df["wav_path"].iloc[i:i + bs]:
                y, sr = sf.read(p, dtype="float32")
                if y.ndim > 1: y = y.mean(1)
                wavs.append(torch.from_numpy(y[: int(max_sec * SR)]))
            L = max(len(w) for w in wavs)
            x = torch.zeros(len(wavs), L); mask = torch.zeros(len(wavs), L)
            for j, w in enumerate(wavs): x[j, :len(w)] = w; mask[j, :len(w)] = 1
            x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-7)
            with torch.no_grad():
                h = m(x.to(dev), output_hidden_states=True).hidden_states[12]
            n = torch.ceil(mask.sum(1) / (x.shape[1] / h.shape[1])).long().clamp(1, h.shape[1])
            out += [h[j, :n[j]].mean(0).float().cpu().numpy() for j in range(len(wavs))]
            if i % (bs * 20) == 0: print(f"  [{name}] {i}/{len(df)}", flush=True)
        return np.stack(out)
    if name == "emotion2vec":
        from funasr import AutoModel as FunModel
        m = FunModel(model="iic/emotion2vec_plus_large", disable_update=True)
        out = []
        for i, p in enumerate(df["wav_path"]):
            r = m.generate(p, granularity="utterance", extract_embedding=True)
            out.append(np.asarray(r[0]["feats"], dtype=np.float32))
            if i % 400 == 0: print(f"  [{name}] {i}/{len(df)}", flush=True)
        return np.stack(out)
    raise ValueError(name)


def feats_text(name, df, dev, bs=64):
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    hf = HF[name]
    tok = AutoTokenizer.from_pretrained(hf)
    # 미세조정 모델의 config.json에 적힌 라벨 이름표(id2label)가 transformers 5.x 검증을
    # 통과 못 하는 경우가 있다(dlckdfuf141/korean-emotion-kluebert-v2). 우리는 분류기 머리를
    # 버리고 백본만 쓰므로 그 필드를 비우고 불러온다 — 가중치·구조와는 무관한 형식 문제다.
    cfg = AutoConfig.from_pretrained(hf)
    for f in ("id2label", "label2id"):
        try:
            object.__setattr__(cfg, f, None)
        except Exception:
            try: setattr(cfg, f, None)
            except Exception: pass
    m = AutoModel.from_pretrained(hf, config=cfg).to(dev).eval()
    out = []
    texts = df["text"].astype(str).tolist()
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
        with torch.no_grad():
            h = m(**b).last_hidden_state
        mask = b["attention_mask"].unsqueeze(-1).float()
        out.append(((h * mask).sum(1) / mask.sum(1)).float().cpu().numpy())
        if i % (bs * 20) == 0: print(f"  [{name}] {i}/{len(texts)}", flush=True)
    return np.concatenate(out)


def feats_face(name, df, dev, bs=64, per_utt=8):
    """발화당 프레임 per_utt장을 뽑아 평균. 112x112 크롭을 모델 입력 크기로 맞춘다."""
    from PIL import Image
    import numpy as _np
    paths, owner = [], []
    for k, d in enumerate(df["face_frames_dir"].fillna("").astype(str)):
        p = ROOT / d
        imgs = sorted(q for q in p.glob("*.jpg") if not q.name.startswith("._")) if d and p.is_dir() else []
        if not imgs:
            continue
        step = max(1, len(imgs) // per_utt)
        for q in imgs[::step][:per_utt]:
            paths.append(q); owner.append(k)
    owner = _np.array(owner)

    if name == "mobilefacenet":
        from emotiefflib.facial_analysis import EmotiEffLibRecognizerTorch
        m = EmotiEffLibRecognizerTorch(model_name="mbf_va_mtl", device="cpu").model.to(dev).eval()
        size, mean, std = 112, 0.5, 0.5
        fwd = lambda x: m(x)
    else:
        from transformers import AutoModel, AutoImageProcessor
        hf = HF[name]
        proc = AutoImageProcessor.from_pretrained(hf)
        m = AutoModel.from_pretrained(hf).to(dev).eval()
        size = proc.size.get("height", 224) if isinstance(proc.size, dict) else 224
        mean, std = float(_np.mean(proc.image_mean)), float(_np.mean(proc.image_std))
        fwd = lambda x: m(pixel_values=x).last_hidden_state.mean(1)

    vecs = []
    for i in range(0, len(paths), bs):
        arr = []
        for q in paths[i:i + bs]:
            a = _np.asarray(Image.open(q).convert("RGB").resize((size, size)), dtype=_np.float32) / 255.0
            arr.append(((a - mean) / std).transpose(2, 0, 1))
        with torch.no_grad():
            v = fwd(torch.from_numpy(_np.stack(arr)).to(dev))
        vecs.append(v.float().cpu().numpy())
        if i % (bs * 20) == 0: print(f"  [{name}] {i}/{len(paths)}", flush=True)
    V = _np.concatenate(vecs)
    D = V.shape[1]
    out = _np.zeros((len(df), D), dtype=_np.float32)
    for k in range(len(df)):
        sel = V[owner == k]
        if len(sel): out[k] = sel.mean(0)
    return out


EXTRACT = {"audio": feats_audio, "text": feats_text, "face": feats_face}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True, choices=["audio", "text", "face"])
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--config", default="configs/config_si_w2v.yaml")
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-val", type=int, default=0, help="0이면 val 전체")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    tr = pick(pd.read_csv(ROOT / cfg.raw["train"]["train_manifest"], low_memory=False), args.n_train)
    va = pick(pd.read_csv(ROOT / cfg.raw["train"]["val_manifest"], low_memory=False), args.n_val)
    ytr, yva = labels_of(tr), labels_of(va)
    maj = np.bincount(yva).max() / len(yva) * 100
    print(f"[probe] {args.modality} · train {len(tr):,} → val {len(va):,} · 최다 클래스 {maj:.1f}%", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {}
    for c in args.candidates:
        t0 = time.time()
        print(f"[probe] === {c}: {CANDS[args.modality].get(c, '')}", flush=True)
        try:
            Xtr = EXTRACT[args.modality](c, tr, dev)
            Xva = EXTRACT[args.modality](c, va, dev)
        except Exception as e:
            print(f"[probe] {c} 실패: {type(e).__name__} {str(e)[:160]}", flush=True)
            res[c] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            continue
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr), ytr)
        acc = clf.score(sc.transform(Xva), yva) * 100
        res[c] = {"val_acc": round(acc, 2), "dim": int(Xtr.shape[1]), "minutes": round((time.time() - t0) / 60, 1)}
        print(f"[probe] {c}: val {acc:.2f}% (차원 {Xtr.shape[1]}, {res[c]['minutes']}분)", flush=True)

    print("\n[probe] === 요약 (최다 클래스 %.1f%%)" % maj)
    for c, r in sorted(res.items(), key=lambda kv: -kv[1].get("val_acc", -1)):
        print(f"  {c:18s} {r.get('val_acc', '실패')}" + (f"  ({r['error']})" if "error" in r else ""))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"modality": args.modality, "n_train": len(tr), "n_val": len(va),
                                              "majority": round(maj, 2), "results": res}, ensure_ascii=False, indent=2))
        print(f"[probe] 저장 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
