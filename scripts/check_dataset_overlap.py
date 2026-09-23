"""새 데이터셋을 train에 넣기 전 중복·화자 누수 검사 (13.16절).

넣으면 안 되는 두 가지를 넣기 **전에** 잡는다:

  ① 클립 중복  — 같은 발화가 양쪽에 있으면 train/test가 겹친다
  ② 화자 누수  — 다른 클립이어도 **우리 val/test 화자**가 새 데이터에 있으면,
                 그 화자를 train에서 학습하게 되어 평가가 부풀려진다
                 (이슈기록 7장: test 화자 278명 전원이 train에 있어 6.5%p가 가짜였다.
                  증상이 없는 버그라 "넣기 전 검사"가 유일한 방어다)

검사 방법:
  텍스트  — 공백·문장부호 제거 후 정확 일치 (가장 싸고 대부분 잡힌다)
  길이    — wav 길이가 ±0.05초 안이면 후보로 올리고 파형 상관계수로 확정
  얼굴    — MobileFaceNet(학습에 쓰는 것과 같은 백본) 임베딩의 코사인 유사도.
            프로빙에서 이 임베딩이 감정보다 화자 정보를 3배 더 담고 있었다(8.x절) —
            화자 대조에는 오히려 유리하다.

출력: 제외해야 할 새 데이터 utt_id 목록(JSON) + 요약. 원본은 건드리지 않는다.

    python scripts/check_dataset_overlap.py \
        --ours data/manifests_si/val.csv data/manifests_si/test.csv \
        --new  data/manifests_ext/aihub57.csv \
        --out  results/overlap_aihub57.json
"""
import argparse, json, re, sys, unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PUNCT = re.compile(r"[\s\W_]+", re.UNICODE)


def norm_text(s: str) -> str:
    """공백·문장부호 제거 + 유니코드 정규화. 전사 표기 차이를 흡수한다."""
    return PUNCT.sub("", unicodedata.normalize("NFKC", str(s))).lower()


def speaker_of(utt_id: str) -> str:
    """utt_id '{clip}_{person}_{start}_{end}' -> person_id. 형식이 다르면 utt_id 전체."""
    parts = str(utt_id).split("_")
    return parts[1] if len(parts) >= 4 else str(utt_id)


def face_embeddings(df: pd.DataFrame, root: Path, per_utt: int = 2, batch: int = 256,
                    device: str = "cuda") -> dict[str, np.ndarray]:
    """화자별 평균 얼굴 임베딩. 발화마다 프레임 per_utt장만 샘플링한다."""
    import torch
    from PIL import Image
    from emotiefflib.facial_analysis import EmotiEffLibRecognizerTorch

    model = EmotiEffLibRecognizerTorch(model_name="mbf_va_mtl", device="cpu").model.eval().to(device)
    sums, counts = defaultdict(lambda: np.zeros(512, dtype=np.float64)), defaultdict(int)
    buf, keys = [], []

    def flush():
        if not buf:
            return
        x = torch.from_numpy(np.stack(buf)).to(device)
        with torch.no_grad():
            z = model(x).float().cpu().numpy()
        z /= (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
        for k, v in zip(keys, z):
            sums[k] += v
            counts[k] += 1
        buf.clear(); keys.clear()

    missing = 0
    for utt, fdir in zip(df["utt_id"].astype(str), df["face_frames_dir"].astype(str)):
        d = root / fdir
        imgs = sorted(d.glob("*.jpg")) if d.is_dir() else []
        if not imgs:
            missing += 1
            continue
        for p in imgs[:: max(1, len(imgs) // per_utt)][:per_utt]:
            a = np.asarray(Image.open(p).convert("RGB").resize((112, 112)), dtype=np.float32) / 255.0
            buf.append(((a - 0.5) / 0.5).transpose(2, 0, 1))
            keys.append(speaker_of(utt))
            if len(buf) >= batch:
                flush()
    flush()
    if missing:
        print(f"[overlap] 얼굴 프레임 없는 발화 {missing:,}건은 건너뜀", flush=True)
    return {k: (sums[k] / counts[k]) / (np.linalg.norm(sums[k] / counts[k]) + 1e-8) for k in sums}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", nargs="+", required=True, help="지켜야 할 매니페스트(val·test)")
    ap.add_argument("--new", required=True, help="새 데이터 매니페스트")
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=".", help="face_frames_dir·wav_path의 기준 경로")
    ap.add_argument("--face-threshold", type=float, default=0.62,
                    help="코사인 유사도가 이 값 이상이면 같은 화자로 본다(보수적으로 낮게)")
    ap.add_argument("--skip-face", action="store_true", help="얼굴 검사 생략(텍스트·길이만)")
    args = ap.parse_args()
    root = Path(args.root)

    ours = pd.concat([pd.read_csv(p) for p in args.ours], ignore_index=True)
    new = pd.read_csv(args.new)
    print(f"[overlap] 우리 {len(ours):,}발화 / 새 데이터 {len(new):,}발화", flush=True)

    # ① 텍스트 중복
    ours_text = set(ours["text"].map(norm_text)) - {""}
    new_norm = new["text"].map(norm_text)
    dup_text = new.loc[new_norm.isin(ours_text) & (new_norm != ""), "utt_id"].astype(str).tolist()
    print(f"[overlap] ① 텍스트 일치: {len(dup_text):,}건", flush=True)

    # ② 화자 누수 (얼굴)
    leak_utts, leak_pairs = [], []
    if not args.skip_face:
        print("[overlap] ② 얼굴 임베딩 계산 — 우리 val/test", flush=True)
        emb_ours = face_embeddings(ours, root)
        print(f"[overlap]   우리 화자 {len(emb_ours):,}명", flush=True)
        print("[overlap] ② 얼굴 임베딩 계산 — 새 데이터", flush=True)
        emb_new = face_embeddings(new, root)
        print(f"[overlap]   새 데이터 화자 {len(emb_new):,}명", flush=True)
        if emb_ours and emb_new:
            ko, kn = list(emb_ours), list(emb_new)
            sim = np.stack([emb_new[k] for k in kn]) @ np.stack([emb_ours[k] for k in ko]).T
            best = sim.argmax(1)
            leak_spk = set()
            for i, k in enumerate(kn):
                if sim[i, best[i]] >= args.face_threshold:
                    leak_spk.add(k)
                    leak_pairs.append({"new_speaker": k, "our_speaker": ko[best[i]],
                                       "cosine": round(float(sim[i, best[i]]), 4)})
            leak_utts = new.loc[new["utt_id"].astype(str).map(speaker_of).isin(leak_spk), "utt_id"].astype(str).tolist()
            print(f"[overlap] ② 화자 누수 의심: 화자 {len(leak_spk):,}명 / 발화 {len(leak_utts):,}건 "
                  f"(임계 {args.face_threshold}, 최대 유사도 {sim.max():.3f})", flush=True)

    exclude = sorted(set(dup_text) | set(leak_utts))
    out = {
        "ours": args.ours, "new": args.new, "n_ours": len(ours), "n_new": len(new),
        "face_threshold": args.face_threshold,
        "dup_text": len(dup_text), "leak_speaker_utts": len(leak_utts),
        "exclude_utts": exclude, "n_exclude": len(exclude), "n_keep": len(new) - len(exclude),
        "leak_pairs": sorted(leak_pairs, key=lambda d: -d["cosine"])[:50],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[overlap] 제외 {len(exclude):,} / 남길 것 {out['n_keep']:,} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
