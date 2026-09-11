"""소음 증강 배선 검증 — **운율과 파형이 같은 잡음을 겪는가.**

이게 어긋나면 에러가 안 난다. 운율은 A 잡음, 파형은 B 잡음인 채로 15에폭이 돌고,
"소음 증강 학습" 결과라고 적힌 숫자가 사실은 아무 조건도 아닌 숫자가 된다.
이 프로젝트가 반복해서 당한 유형이라 사후 확인이 아니라 테스트로 박아둔다.

핵심 검증은 사전계산 함수끼리 비교하는 게 아니라 **데이터셋이 실제로 돌려준 것끼리**
비교하는 것이다:

    ds[i]["prosody"]  ==  extract_prosody(ds[i]["waveform"])

왼쪽은 사전계산 npz에서 읽은 값이고, 오른쪽은 학습 때 wav2vec2가 실제로 먹는 파형에서
뽑은 값이다. 이 둘이 같으면 시드·SNR·포맷·librosa/soundfile 경로가 전부 맞은 것이다.

실데이터가 필요하다(data/manifests_si/train.csv). 없으면 건너뛴다.

    python tests/test_noise_aug.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa
import numpy as np
import pandas as pd

from src.config import load_config
from src.datasets.manifest_dataset import (
    ManifestEmotionDataset, add_white_noise, noise_seed,
)
from src.features.prosody import extract_prosody

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifests_si" / "train.csv"
SNRS = [20.0, 10.0, 5.0]
N_UTT = 4


def test_seed_format_invariance():
    """config는 정수 10을, argparse는 10.0을 준다. 둘이 같은 시드여야 한다.

    [사건] 처음 구현은 f"{utt}@{snr_db}"였다. 그러면 "u@10"과 "u@10.0"이 서로 다른
    해시가 되어, config로 돌린 학습과 CLI로 돌린 사전계산이 어긋난다 — 조용히.
    """
    assert noise_seed("u", 10) == noise_seed("u", 10.0)
    assert noise_seed("u", 5) == noise_seed("u", 5.0)
    print("  ✅ 시드 포맷 불변 (10 == 10.0)")


def test_seed_differs_per_snr():
    """SNR마다 독립적인 잡음 실현. 안 섞으면 모양은 같고 진폭만 다른 족이 된다."""
    seeds = {s: noise_seed("u", s) for s in SNRS}
    assert len(set(seeds.values())) == len(SNRS), seeds

    # 왜 SNR을 시드에 섞어야 하는지의 전제: 같은 시드면 numpy가 같은 표준정규를 뽑고
    # scale만 곱하므로 잡음 '모양'이 동일하다.
    #
    # 비율(noise_a / noise_b)로 보지 않고 상관계수로 보는 이유: add_white_noise는
    # (y+noise)를 float32로 되돌리므로 빼서 복원한 잡음에 반올림 오차가 섞인다.
    # 잡음이 0에 가까운 원소에서 비율은 발산하지만 상관계수는 견딘다.
    y = np.random.default_rng(0).normal(0, 1, 4096).astype(np.float32)

    def recovered(snr, seed):
        return (add_white_noise(y, snr, seed=seed) - y).astype(np.float64)

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    same = corr(recovered(20.0, 1234), recovered(10.0, 1234))
    assert same > 0.999, f"같은 시드면 잡음 모양이 같아야 한다(전제 확인): corr={same:.4f}"

    indep = corr(recovered(20.0, noise_seed("u", 20.0)),
                 recovered(10.0, noise_seed("u", 10.0)))
    assert abs(indep) < 0.1, f"SNR별 시드가 독립 실현을 줘야 한다: corr={indep:.4f}"
    print(f"  ✅ SNR별 독립 잡음 실현 (같은 시드 corr={same:.4f} → SNR별 시드 corr={indep:+.4f})")


def test_snr_draw_changes_with_epoch(ds):
    """에폭을 바꾸면 추첨이 바뀌고, 같은 에폭이면 항상 같아야 한다."""
    utts = [str(u) for u in ds.df["utt_id"]]

    ds.set_epoch(1)
    a = [ds._snr_for(u) for u in utts]
    assert a == [ds._snr_for(u) for u in utts], "같은 에폭에서 추첨이 흔들린다"

    # 발화가 4개뿐이라 한 에폭 비교로는 우연히 같을 수 있다. 여러 에폭을 모아서
    # "에폭에 따라 달라진다"를 본다.
    seqs = set()
    for e in range(1, 21):
        ds.set_epoch(e)
        seqs.add(tuple(ds._snr_for(u) for u in utts))
    assert len(seqs) > 1, "에폭을 바꿔도 추첨이 그대로다 — 증강이 아니라 고정 조건이다"

    # 깨끗(None)도 뽑혀야 한다. 안 뽑히면 깨끗한 성능을 버리는 학습이 된다.
    drawn = {s for seq in seqs for s in seq}
    assert None in drawn, f"깨끗한 조건이 한 번도 안 뽑혔다: {drawn}"
    assert drawn == {None, *SNRS}, f"뽑힌 조건: {drawn}"
    print(f"  ✅ 에폭별 추첨 변화 · 20에폭에서 {len(seqs)}종 · 조건 {sorted(drawn, key=str)}")


def test_prosody_matches_returned_waveform(ds):
    """**핵심.** 돌려준 운율 == 돌려준 파형에서 뽑은 운율.

    이게 맞으면 사전계산 npz와 학습 중 파형이 동일한 잡음을 공유한다.
    """
    sr = ds.cfg.audio_sample_rate
    checked = {None: 0, **{s: 0 for s in SNRS}}
    for e in range(1, 13):
        ds.set_epoch(e)
        for i in range(len(ds)):
            utt = str(ds.df.iloc[i]["utt_id"])
            snr = ds._snr_for(utt)
            if checked[snr] >= 2:      # 조건마다 2건이면 충분하다(1건 × 12에폭은 느리다)
                continue
            item = ds[i]
            # dtype이 어긋나면 깨끗·소음이 섞인 배치가 np.stack에서 float64로 올라가고
            # CollateFn은 prosody를 캐스팅하지 않아 그대로 모델까지 간다.
            assert item["prosody"].dtype == np.float32, \
                f"{utt} SNR={snr}: 운율 dtype {item['prosody'].dtype} (float32여야 함)"
            got = extract_prosody(item["waveform"], sr).astype(np.float32)
            d = float(np.abs(item["prosody"] - got).max())
            assert d < 1e-5, (
                f"{utt} SNR={snr}: 운율과 파형이 다른 잡음을 겪고 있다 (최대차 {d:.3e})"
            )
            checked[snr] += 1
    assert all(v > 0 for v in checked.values()), f"검사 못 한 조건이 있다: {checked}"
    print(f"  ✅ 운율 == 파형에서 뽑은 운율 (조건별 건수 {checked})")


def test_clean_draw_is_actually_clean(ds):
    """깨끗하게 뽑힌 발화는 정말 깨끗해야 한다 — 소음 운율이 새어들면 안 된다."""
    sr = ds.cfg.audio_sample_rate
    # 발화가 4개뿐이라 한 에폭에 깨끗이 하나도 안 뽑힐 수 있다(확률 (3/4)^4 ≈ 32%).
    # 에폭을 넘겨가며 2건을 모은다 — 테스트가 운에 따라 흔들리면 안 된다.
    n = 0
    for e in range(1, 13):
        ds.set_epoch(e)
        for i in range(len(ds)):
            row = ds.df.iloc[i]
            if ds._snr_for(str(row["utt_id"])) is not None or n >= 2:
                continue
            y, _ = librosa.load(row["wav_path"], sr=sr, mono=True, duration=8.0)
            want = extract_prosody(y, sr).astype(np.float32)
            assert np.abs(ds[i]["prosody"] - want).max() < 1e-5, \
                f"{row['utt_id']}: 깨끗해야 하는데 오염됐다"
            n += 1
        if n >= 2:
            break
    assert n >= 2, "12에폭 동안 깨끗한 발화가 2건도 안 뽑혔다 — 추첨이 의심스럽다"
    print(f"  ✅ 깨끗한 추첨은 실제로 깨끗 ({n}건)")


def test_guards(cfg, mini_manifest, npz_dir):
    """막아야 하는 조합이 정말 막히는가. 안 막히면 조용히 틀린 실험이 된다."""
    import copy

    def fails(**kw):
        try:
            ManifestEmotionDataset(str(mini_manifest), cfg, **kw)
        except (ValueError, FileNotFoundError) as e:
            return str(e)[:60]
        return None

    m = fails(noise_aug_snrs=SNRS, noisy_prosody_dir=npz_dir, noise_snr_db=10.0)
    assert m, "평가용 고정 SNR과 증강을 같이 줘도 통과했다"
    print(f"  ✅ 고정 SNR + 증강 동시 사용 차단 — {m}")

    m = fails(noise_aug_snrs=SNRS)
    assert m, "noisy_prosody_dir 없이도 통과했다 — 매 에폭 운율을 재계산하게 된다"
    print(f"  ✅ 사전계산 경로 누락 차단 — {m}")

    mel_cfg = copy.deepcopy(cfg)
    # raw가 아니라 필드를 고친다 — audio_backbone은 load_config 시점에 dataclass
    # 필드로 굳으므로 raw["audio"]["backbone"]을 고쳐도 안 바뀐다(src/config.py:84).
    mel_cfg.audio_backbone = "mel"
    try:
        ManifestEmotionDataset(str(mini_manifest), mel_cfg,
                               noise_aug_snrs=SNRS, noisy_prosody_dir=npz_dir)
        raise AssertionError("멜 백본인데 통과했다 — 깨끗한 멜 + 오염된 운율이 된다")
    except ValueError as e:
        print(f"  ✅ 멜 백본 차단 — {str(e)[:60]}")

    m = fails(noise_aug_snrs=[99.0], noisy_prosody_dir=npz_dir)
    assert m, "사전계산이 없는 SNR인데 통과했다"
    print(f"  ✅ 없는 SNR 차단 — {m}")

    # 발화가 빠진 npz는 __init__에서 죽어야 한다 — 에폭 중간 KeyError가 아니라.
    short = Path(npz_dir) / "missing"
    short.mkdir(exist_ok=True)
    for s in SNRS:
        z = np.load(Path(npz_dir) / f"snr{s:g}.npz")
        np.savez_compressed(short / f"snr{s:g}.npz",
                            utt_ids=z["utt_ids"][:1], prosody=z["prosody"][:1])
    m = fails(noise_aug_snrs=SNRS, noisy_prosody_dir=short)
    assert m, "발화가 빠진 npz인데 통과했다 — 에폭 중간에 죽거나 조용히 틀린다"
    print(f"  ✅ 발화 누락 npz 차단 — {m}")


def test_cache_written_and_stays_clean(cfg, mini_manifest, npz_dir, cache_dir: Path):
    """캐시가 **쓰이는가**, 그리고 거기 들어간 운율이 **깨끗한가**.

    [사건] 소음 증강 덮어쓰기 블록을 캐시 저장 코드 앞에 끼워넣으면서 들여쓰기로
    저장 블록을 자기 안에 삼켰다. 결과가 두 겹으로 나빴다.

      · 증강을 안 쓰는 평범한 학습에서 feature_cache가 한 건도 안 쓰인다
        (서버에서 한 시간 들여 만든 39GB가 앞으로 안 늘어난다)
      · 증강을 쓰면 오염된 운율이 깨끗한 캐시에 박힌다 — 바로 위 주석이 막겠다고
        적어둔 그 일이 주석 밑에서 일어난다

    다른 테스트가 전부 cache_dir=None으로 돌아서 못 잡았다. 캐시를 실제로 쓰는
    검증이 없으면 같은 사고가 또 난다.
    """
    from src.features.prosody import extract_prosody

    sr = cfg.audio_sample_rate
    ds = ManifestEmotionDataset(
        str(mini_manifest), cfg, return_waveform=True, cache_dir=cache_dir,
        noise_aug_snrs=SNRS, noisy_prosody_dir=npz_dir,
    )
    # 소음이 뽑힌 발화를 골라 읽는다 — 그 발화의 캐시가 오염되는지가 핵심이다.
    ds.set_epoch(1)
    noisy = [i for i in range(len(ds)) if ds._snr_for(str(ds.df.iloc[i]["utt_id"])) is not None]
    assert noisy, "1에폭에 소음이 뽑힌 발화가 없다"
    for i in noisy:
        ds[i]

    written = sorted(p.name for p in cache_dir.glob("*.npz"))
    assert len(written) == len(noisy), f"캐시가 안 쓰였다: {written}"

    for i in noisy:
        row = ds.df.iloc[i]
        utt = str(row["utt_id"])
        y, _ = librosa.load(row["wav_path"], sr=sr, mono=True, duration=8.0)
        want = extract_prosody(y, sr).astype(np.float32)
        cached = np.load(cache_dir / f"{utt}.npz")["prosody"]
        d = float(np.abs(cached - want).max())
        assert d < 1e-5, f"{utt}: 캐시에 오염된 운율이 저장됐다 (깨끗한 값과 {d:.3e} 차이)"

    # 캐시 히트 경로에서도 소음 덮어쓰기가 걸려야 한다 — 캐시에는 깨끗한 값이 있으니
    # 덮어쓰기가 빠지면 "증강했다"는 실험이 조용히 깨끗한 실험이 된다.
    for i in noisy:
        item = ds[i]
        got = extract_prosody(item["waveform"], sr).astype(np.float32)
        assert np.abs(item["prosody"] - got).max() < 1e-5, "캐시 히트 시 덮어쓰기가 빠졌다"
    # 증강을 **안** 쓰는 평범한 학습에서도 캐시가 쓰여야 한다. 위 버그의 더 심한 절반이
    # 이쪽이었다 — 소음과 무관한 모든 학습이 캐시 없이 돌게 된다.
    plain_dir = cache_dir.parent / "cache_plain"
    plain = ManifestEmotionDataset(str(mini_manifest), cfg, return_waveform=True,
                                   cache_dir=plain_dir)
    assert plain.noise_aug_snrs is None
    plain[0]
    n_plain = len(list(plain_dir.glob("*.npz")))
    assert n_plain == 1, f"소음 없는 경로에서 캐시가 안 쓰였다 ({n_plain}건)"

    print(f"  ✅ 캐시 {len(written)}건 기록 · 내용은 깨끗 · 캐시 히트에도 소음 적용 "
          f"· 소음 없는 경로도 기록")


def test_dataloader_workers(cfg, mini_manifest, npz_dir):
    """워커 프로세스를 거쳐도 되는가. `--with-workers`일 때만 돈다(느리다).

    이게 왜 따로 필요한가: 위 테스트는 전부 메인 프로세스에서 ds[i]를 직접 부른다.
    실제 학습은 DataLoader(num_workers=8)를 거치는데 거기서 두 가지가 새로 걸린다.

      · 데이터셋이 pickle 돼야 한다(공유 numpy 배열·dict를 들고 있다)
      · set_epoch(epoch)가 **워커까지 가야 한다.** 안 가면 15에폭이 1에폭째 조건으로
        돌면서 에러가 하나도 안 난다 — train.py가 증강 시 persistent_workers를 끄는 이유다.
    """
    import torch
    from torch.utils.data import DataLoader

    from src.datasets.manifest_dataset import make_collate_fn

    ds = ManifestEmotionDataset(
        str(mini_manifest), cfg, return_waveform=True,
        prosody_stats_path=ROOT / "data" / "prosody_stats_train_si.json",
        noise_aug_snrs=SNRS, noisy_prosody_dir=npz_dir,
    )
    collate = make_collate_fn(cfg.text_pretrained)

    def run(epoch):
        ds.set_epoch(epoch)
        # persistent_workers=False — train.py가 증강 시 쓰는 것과 같은 조건.
        dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=2,
                        collate_fn=collate, persistent_workers=False)
        out = []
        for b in dl:
            assert b["prosody_vec"].dtype == torch.float32, b["prosody_vec"].dtype
            assert b["waveform"].dtype == torch.float32
            assert torch.isfinite(b["prosody_vec"]).all(), "운율에 NaN/Inf"
            out += [tuple(np.round(x.numpy(), 4)) for x in b["prosody_vec"]]
        return out

    a, b = run(3), run(3)
    assert a == b, "같은 에폭인데 워커를 거치면 결과가 흔들린다"
    assert len(a) == len(ds)
    assert run(4) != a, "에폭을 바꿨는데 워커가 옛 조건으로 돈다 — set_epoch가 안 갔다"
    print(f"  ✅ 워커 2개 통과 · prosody_vec float32 · set_epoch 전달됨 ({len(a)}건)")


def build_npz(cfg, df, out_dir: Path):
    """scripts/precompute_noisy_prosody.py와 **같은 함수**로 만든다.

    여기서 별도 구현을 하면 테스트가 본코드가 아니라 테스트 자신을 검증하게 된다.
    """
    from scripts.precompute_noisy_prosody import one

    for snr in SNRS:
        ids, rows = [], []
        for r in df.itertuples():
            uid, pros = one((str(r.utt_id), str(r.wav_path), snr,
                             cfg.audio_sample_rate, 8.0))
            ids.append(uid)
            rows.append(pros)
        np.savez_compressed(out_dir / f"snr{snr:g}.npz",
                            utt_ids=np.array(ids), prosody=np.stack(rows))


def main() -> int:
    print("=== 소음 증강 배선 검증 ===")
    test_seed_format_invariance()
    test_seed_differs_per_snr()

    if not MANIFEST.exists():
        print(f"  ⏭  실데이터 없음({MANIFEST}) — 나머지 건너뜀")
        return 0

    cfg = load_config(ROOT / "configs" / "config_noise_aug.yaml")
    df = pd.read_csv(MANIFEST).head(N_UTT)
    if not all(Path(p).exists() for p in df["wav_path"]):
        print("  ⏭  wav 파일 없음 — 나머지 건너뜀")
        return 0

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        mini = td / "mini.csv"
        df.to_csv(mini, index=False)
        build_npz(cfg, df, td)

        ds = ManifestEmotionDataset(
            str(mini), cfg, return_waveform=True,
            noise_aug_snrs=SNRS, noisy_prosody_dir=td,
        )
        test_snr_draw_changes_with_epoch(ds)
        test_prosody_matches_returned_waveform(ds)
        test_clean_draw_is_actually_clean(ds)
        test_guards(cfg, mini, td)
        test_cache_written_and_stays_clean(cfg, mini, td, td / "cache")
        if "--with-workers" in sys.argv:
            test_dataloader_workers(cfg, mini, td)
        else:
            print("  ⏭  워커 검증 생략 (--with-workers로 켠다)")

    print("=== 전부 통과 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
