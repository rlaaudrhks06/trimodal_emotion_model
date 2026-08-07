"""저장된 발화별 예측(`results/predictions/*.csv`)을 사후 분석한다.

**모델을 다시 돌리지 않는다.** `scripts/evaluate.py --save-predictions`가 남긴 CSV만
읽어서 계산하므로, 서버에서 한 번 평가해 받아온 파일로 로컬에서 전부 할 수 있다.

배경(통합기록 8.29.1절): v11과 v12b의 test 정확도 차이 0.97%p가 유의한지 판단하려고
보니 **두 모델이 각각 어느 발화를 맞혔는지를 저장하지 않아** 짝지어 검정을 할 수 없었다.
그리고 집계 지표를 비교하면서 *단일 정확도*의 표준오차(0.466%p)로 나눠 "2.08배라
노이즈가 아니다"라고 결론냈는데, 차이를 볼 때는 *차이*의 표준오차(0.658%p)를 써야 해서
실제로는 1.47배였다. 이 스크립트는 그 두 가지를 모두 제자리에 돌려놓는다.

분석 항목(주는 인자에 따라 되는 것만 수행):
  기본            클래스별 재현율, 화자별 정확도 분산, coarse 재매핑, 짧은 발화 부분집합
  --compare       McNemar 짝지어 검정 (+ 독립 가정 대조)
  --ablation      모달리티 애블레이션 표 (기여도 = 전체 - 해당 모달리티 제거)
  --calibrate     온도 스케일링(val에서 적합) + ECE + 위험-커버리지 곡선

실행 예:
    python scripts/analyze_predictions.py --pred results/predictions/v11.csv \\
        --manifest data/manifests_si/test.csv \\
        --ablation results/predictions/v11_no_{audio,visual,text}.csv \\
        --calibrate results/predictions/v11_val.csv
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.datasets.labels import EMOTION_LABELS

PCOLS = [f"p_{e}" for e in EMOTION_LABELS]

# 8.29.3절에서 측정한 매핑. 로봇 행동은 "좋음/나쁨/보통" 해상도면 충분할 수 있다.
COARSE = {"happy": "긍정", "surprise": "긍정",
          "angry": "부정", "disgust": "부정", "fear": "부정", "sad": "부정",
          "neutral": "중립"}


def load_pred(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"utt_id", "label", "pred"} | set(PCOLS)
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path}: 컬럼 없음 {sorted(missing)}")
    return df


def acc_ci(correct: int, n: int) -> tuple:
    """정확도와 95% 신뢰구간 반폭. n=0이면 (nan, nan)."""
    if n == 0:
        return float("nan"), float("nan")
    p = correct / n
    return p, 1.96 * math.sqrt(p * (1 - p) / n)


def speaker_of(utt_ids: pd.Series) -> pd.Series:
    """utt_id = {clip_id}_{person_id}_{start}_{end} (build_manifest_aihub.py:217).

    clip_id에도 밑줄이 들어갈 수 있으므로 **뒤에서** 3칸을 자른다.
    """
    return utt_ids.astype(str).str.rsplit("_", n=3).str[1]


# ---------------------------------------------------------------- 기본 분석
def report_per_class(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("클래스별 재현율 — 전체 정확도가 가리는 것")
    print("=" * 70)
    print(f"  {'감정':10} {'개수':>7} {'재현율':>9} {'예측된 수':>10}")
    for i, emo in enumerate(EMOTION_LABELS):
        sel = df["label"] == i
        n = int(sel.sum())
        if n == 0:
            continue
        rec, half = acc_ci(int((df.loc[sel, "pred"] == i).sum()), n)
        n_pred = int((df["pred"] == i).sum())
        flag = "  <-- 사실상 작동 안 함" if rec < 0.25 else ""
        print(f"  {emo:10} {n:>7,} {100*rec:>8.1f}% {n_pred:>10,}{flag}")


def report_speakers(df: pd.DataFrame, min_utts: int) -> None:
    """화자별 정확도 분산 — '특정 사용자에게만 되는 로봇'인지 확인한다.

    성별·연령 메타데이터가 없어 인구집단별 격차(equal opportunity gap)는 못 구하지만,
    화자 단위 편차는 지금 있는 정보로 계산할 수 있다.
    """
    d = df.assign(spk=speaker_of(df["utt_id"]), ok=(df["label"] == df["pred"]))
    g = d.groupby("spk")["ok"].agg(["mean", "count"])
    g = g[g["count"] >= min_utts].sort_values("mean")
    print("\n" + "=" * 70)
    print(f"화자별 정확도 — 발화 {min_utts}개 이상인 화자 {len(g)}명")
    print("=" * 70)
    if g.empty:
        print("  대상 화자 없음")
        return
    overall = d["ok"].mean()
    print(f"  전체 {100*overall:.2f}%  |  화자 평균 {100*g['mean'].mean():.2f}%  "
          f"표준편차 {100*g['mean'].std():.2f}%p")
    print(f"  최저 {100*g['mean'].iloc[0]:.1f}% (화자 {g.index[0]}, {int(g['count'].iloc[0])}발화)"
          f"  ~  최고 {100*g['mean'].iloc[-1]:.1f}% (화자 {g.index[-1]}, {int(g['count'].iloc[-1])}발화)")
    print(f"  최고-최저 격차 {100*(g['mean'].iloc[-1]-g['mean'].iloc[0]):.1f}%p")
    lo = g[g["mean"] < overall - 0.10]
    if len(lo):
        print(f"  전체보다 10%p 이상 낮은 화자 {len(lo)}명: "
              f"{', '.join(lo.index[:10])}{' ...' if len(lo) > 10 else ''}")


def report_coarse(df: pd.DataFrame) -> None:
    """7클래스 예측을 묶기만 해서 얻는 정확도(8.29.3절). 재학습이 아니다."""
    idx2 = {i: COARSE[e] for i, e in enumerate(EMOTION_LABELS)}
    t = df["label"].map(idx2)
    p = df["pred"].map(idx2)
    acc, half = acc_ci(int((t == p).sum()), len(df))
    base = t.value_counts().iloc[0] / len(df)
    print("\n" + "=" * 70)
    print("클래스 체계 재매핑 — 출력만 묶었을 때 (재학습 없음)")
    print("=" * 70)
    print(f"  긍정=행복·놀람 / 부정=분노·혐오·공포·슬픔 / 중립=중립")
    print(f"  정확도 {100*acc:.2f}% (±{100*half:.2f}%p)  최다클래스 기준선 {100*base:.2f}%  "
          f"정규화이득 {100*(acc-base)/(1-base):.1f}%")
    for grp in sorted(set(COARSE.values())):
        sel = t == grp
        rec, _ = acc_ci(int((p[sel] == grp).sum()), int(sel.sum()))
        print(f"    {grp:6} n={int(sel.sum()):>6,}  재현율 {100*rec:.1f}%")


def report_short(df: pd.DataFrame, manifest: Path, max_chars: int) -> None:
    """짧은 발화 부분집합 — 텍스트로는 감정을 알 수 없는 구간.

    여기서 정확도가 전체보다 크게 낮으면 **모델이 텍스트 지름길을 타고 있다**는 뜻이고,
    그러면 융합 설계 자체를 다시 봐야 한다(리뷰 3번 항목).
    """
    man = pd.read_csv(manifest)
    if "text" not in man.columns:
        print("\n[짧은 발화] 매니페스트에 text 컬럼이 없어 건너뜀")
        return
    m = man[["utt_id", "text"]].copy()
    m["utt_id"] = m["utt_id"].astype(str)
    d = df.assign(utt_id=df["utt_id"].astype(str)).merge(m, on="utt_id", how="left")
    if d["text"].isna().any():
        print(f"\n[짧은 발화] 경고: 매니페스트에서 못 찾은 발화 {int(d['text'].isna().sum())}건")
    d["nchars"] = d["text"].fillna("").astype(str).str.replace(r"\s+", "", regex=True).str.len()
    d["ok"] = d["label"] == d["pred"]

    print("\n" + "=" * 70)
    print("발화 길이별 정확도 — '텍스트 지름길'을 타고 있는가")
    print("=" * 70)
    bins = [(0, max_chars), (max_chars + 1, 20), (21, 40), (41, 10**9)]
    names = [f"≤{max_chars}자 (짧음)", f"{max_chars+1}~20자", "21~40자", "41자~"]
    overall = d["ok"].mean()
    for (lo, hi), nm in zip(bins, names):
        sel = d["nchars"].between(lo, hi)
        n = int(sel.sum())
        if n == 0:
            continue
        acc, half = acc_ci(int(d.loc[sel, "ok"].sum()), n)
        gap = acc - overall
        mark = "  <-- 전체보다 크게 낮음" if gap < -0.05 else ""
        print(f"  {nm:14} n={n:>6,}  정확도 {100*acc:>5.1f}% (±{100*half:.1f}%p)  "
              f"전체 대비 {100*gap:+.1f}%p{mark}")
    print(f"  {'전체':14} n={len(d):>6,}  정확도 {100*overall:>5.1f}%")

    short = d[d["nchars"] <= max_chars]
    if len(short):
        ex = short[~short["ok"]].head(5)
        if len(ex):
            print(f"\n  짧은 발화 오답 예시:")
            for _, r in ex.iterrows():
                print(f"    \"{r['text']}\"  정답 {EMOTION_LABELS[r['label']]} "
                      f"-> 예측 {EMOTION_LABELS[r['pred']]}")


# ---------------------------------------------------------------- 모델 비교
def mcnemar(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str) -> None:
    """같은 test셋 두 모델의 짝지어 비교.

    독립 가정으로 구한 차이의 표준오차는 짝지어진 데이터에서 **보수적**이다.
    실제로 필요한 것은 서로 다르게 답한 발화(불일치쌍)만 보는 McNemar 검정이다.
    """
    a = a.set_index("utt_id"); b = b.set_index("utt_id")
    common = a.index.intersection(b.index)
    if len(common) != len(a) or len(common) != len(b):
        print(f"[비교] 경고: 공통 발화 {len(common):,}건 "
              f"({name_a} {len(a):,}, {name_b} {len(b):,})")
    a, b = a.loc[common], b.loc[common]
    if not (a["label"] == b["label"]).all():
        raise ValueError("두 파일의 정답 라벨이 다르다 — 같은 test셋이 아니다")

    oa = (a["label"] == a["pred"]).to_numpy()
    ob = (b["label"] == b["pred"]).to_numpy()
    n = len(common)
    n01 = int((oa & ~ob).sum())   # a만 맞힘
    n10 = int((~oa & ob).sum())   # b만 맞힘
    acc_a, acc_b = oa.mean(), ob.mean()
    diff = acc_a - acc_b

    se_ind = math.sqrt(acc_a*(1-acc_a)/n + acc_b*(1-acc_b)/n)

    print("\n" + "=" * 70)
    print(f"짝지어 비교 — {name_a} vs {name_b}  (n={n:,})")
    print("=" * 70)
    print(f"  {name_a:20} {100*acc_a:.2f}%")
    print(f"  {name_b:20} {100*acc_b:.2f}%")
    print(f"  차이                 {100*diff:+.2f}%p")
    print(f"\n  불일치쌍: {name_a}만 맞힘 {n01:,} / {name_b}만 맞힘 {n10:,} (합 {n01+n10:,})")

    nd = n01 + n10
    if nd == 0:
        print("  두 모델의 정오 패턴이 완전히 같다 — 검정 불가")
        return
    if nd < 25:
        # 표본이 작으면 정규근사가 부정확해 이항 정확검정을 쓴다.
        from scipy.stats import binomtest
        p = binomtest(n01, nd, 0.5).pvalue
        how = "이항 정확검정"
    else:
        chi2 = (abs(n01 - n10) - 1) ** 2 / nd   # 연속성 보정
        from scipy.stats import chi2 as chi2_dist
        p = 1 - chi2_dist.cdf(chi2, df=1)
        how = "McNemar χ²(연속성 보정)"
    se_paired = math.sqrt(nd) / n
    print(f"\n  차이의 표준오차 — 독립 가정 {100*se_ind:.3f}%p / 짝지어 {100*se_paired:.3f}%p")
    print(f"  {how}: p = {p:.4f}  ->  "
          f"{'유의함 (p<0.05)' if p < 0.05 else '구분되지 않음 (p>=0.05)'}")


def report_ablation(base: pd.DataFrame, paths: list) -> None:
    """모달리티를 하나씩 지웠을 때의 정확도. 기여도 = 전체 - 지운 뒤."""
    print("\n" + "=" * 70)
    print("모달리티 애블레이션 — 어느 브랜치가 실제로 일하는가")
    print("=" * 70)
    b = base.set_index("utt_id")
    acc0 = (b["label"] == b["pred"]).mean()
    n = len(b)
    print(f"  {'조건':22} {'정확도':>9} {'기여도':>9}  판정")
    print(f"  {'전체':22} {100*acc0:>8.2f}% {'—':>9}")
    rows = []
    for p in paths:
        d = load_pred(Path(p)).set_index("utt_id")
        common = b.index.intersection(d.index)
        d = d.loc[common]
        acc = (d["label"] == d["pred"]).mean()
        drop = acc0 - acc
        # 같은 발화를 두 조건으로 평가한 것이므로 짝지어 표준오차를 쓴다:
        # 정오가 엇갈린 발화 수 nd에 대해 SE(차이) = √nd / n.
        ok_b = (b.loc[common, "label"] == b.loc[common, "pred"]).to_numpy()
        ok_d = (d["label"] == d["pred"]).to_numpy()
        nd = int((ok_b != ok_d).sum())
        se = math.sqrt(nd) / len(common) if nd else 0.0
        sig = "유의" if nd and abs(drop) > 1.96 * se else "구분 안 됨"
        rows.append((Path(p).stem, acc, drop, sig))
        print(f"  {Path(p).stem:22} {100*acc:>8.2f}% {100*drop:>+8.2f}%p  {sig}")
    if rows:
        top = max(rows, key=lambda r: r[2])
        print(f"\n  기여도 최대: [{top[0]}] 조건에서 {100*top[2]:+.2f}%p 떨어짐 "
              f"-> 여기서 지운 브랜치가 가장 크게 기여한다")
        for r in (x for x in rows if x[2] < 0):
            print(f"  주의: [{r[0]}] 조건이 오히려 더 높다({100*r[2]:+.2f}%p) "
                  f"-> 지운 브랜치가 방해하고 있을 수 있다")


# ---------------------------------------------------------------- 보정·보류
def ece(probs: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error — 확신도와 실제 정답률의 평균 괴리."""
    conf = probs.max(axis=1)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if not sel.any():
            continue
        total += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return total


def fit_temperature(probs: np.ndarray, labels: np.ndarray) -> float:
    """val셋에서 NLL을 최소화하는 온도 T를 찾는다 (Guo et al. 2017).

    저장된 것이 로짓이 아니라 확률이므로 log(p)를 로짓 대용으로 쓴다.
    softmax(log(p)/T)는 T=1에서 원래 확률과 같고, 온도 스케일링의 성질
    (순위 불변 = 정확도 불변)도 그대로 유지된다.
    """
    z = np.log(np.clip(probs, 1e-12, None))
    best_t, best_nll = 1.0, float("inf")
    grid = np.arange(0.05, 10.001, 0.01)
    for t in grid:
        s = z / t
        s = s - s.max(axis=1, keepdims=True)
        logp = s - np.log(np.exp(s).sum(axis=1, keepdims=True))
        nll = -logp[np.arange(len(labels)), labels].mean()
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    if best_t <= grid[0] + 1e-9 or best_t >= grid[-1] - 1e-9:
        print(f"  경고: 최적 온도 T={best_t:.2f}가 탐색 범위 끝에 걸렸다 — "
              f"실제 최적값은 범위 밖일 수 있다")
    return best_t


def apply_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    z = np.log(np.clip(probs, 1e-12, None)) / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def report_calibration(test: pd.DataFrame, val_path: Path | None) -> None:
    P = test[PCOLS].to_numpy()
    y = test["label"].to_numpy()
    ok = (test["label"] == test["pred"]).to_numpy()

    print("\n" + "=" * 70)
    print("신뢰도 보정 — '행복 51%'가 정말 51%인가")
    print("=" * 70)
    print(f"  보정 전 ECE {ece(P, ok):.4f}   평균 확신도 {P.max(1).mean():.4f} "
          f"vs 실제 정확도 {ok.mean():.4f}")

    Pc = P
    if val_path is not None:
        v = load_pred(val_path)
        t = fit_temperature(v[PCOLS].to_numpy(), v["label"].to_numpy())
        Pc = apply_temperature(P, t)
        moved = int((Pc.argmax(1) != P.argmax(1)).sum())
        print(f"  val에서 적합한 온도 T = {t:.2f} "
              f"({'과신 -> 확률을 눌러야 함' if t > 1 else '과소신 -> 확률을 키워야 함'})")
        print(f"  보정 후 ECE {ece(Pc, ok):.4f}   평균 확신도 {Pc.max(1).mean():.4f}")
        print(f"  예측이 바뀐 발화 {moved}건 (온도 스케일링은 순위를 안 바꾸므로 0이어야 정상)")
    else:
        print("  (--calibrate로 val 예측 파일을 주면 온도 스케일링까지 수행)")

    print("\n" + "-" * 70)
    print("판단 보류(abstention) — 확신이 낮으면 중립 행동으로 넘긴다")
    print("-" * 70)
    conf = Pc.max(1)
    print(f"  {'임계값':>8} {'응답률':>9} {'응답한 것의 정확도':>18} {'보류':>9}")
    for th in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        sel = conf >= th
        if sel.sum() == 0:
            continue
        cov = sel.mean()
        acc, half = acc_ci(int(ok[sel].sum()), int(sel.sum()))
        print(f"  {th:>8.2f} {100*cov:>8.1f}% {100*acc:>17.2f}% "
              f"{100*(1-cov):>8.1f}%")
    print("\n  틀리게 확신하고 반응하는 로봇이 반응을 보류하는 로봇보다 신뢰를 빨리 잃는다.")
    print("  응답률을 얼마나 포기하고 정확도를 살지는 로봇 행동 설계에서 정할 문제다.")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="분석할 예측 CSV (test)")
    ap.add_argument("--manifest", default=None,
                    help="짧은 발화 분석용 — text 컬럼이 필요하다")
    ap.add_argument("--compare", default=None, help="McNemar로 비교할 다른 예측 CSV")
    ap.add_argument("--ablation", nargs="+", default=None,
                    help="모달리티를 지우고 평가한 예측 CSV들")
    ap.add_argument("--calibrate", default=None,
                    help="온도 적합에 쓸 val 예측 CSV")
    ap.add_argument("--min-utts", type=int, default=20,
                    help="화자별 분석에 포함할 최소 발화 수 (기본 20)")
    ap.add_argument("--short-chars", type=int, default=10,
                    help="'짧은 발화' 기준 글자 수, 공백 제외 (기본 10)")
    args = ap.parse_args()

    pred_path = Path(args.pred)
    df = load_pred(pred_path)
    acc, half = acc_ci(int((df["label"] == df["pred"]).sum()), len(df))
    print("=" * 70)
    print(f"{pred_path.name}  —  발화 {len(df):,}건")
    print("=" * 70)
    print(f"  정확도 {100*acc:.2f}% (±{100*half:.2f}%p)")

    report_per_class(df)
    report_speakers(df, args.min_utts)
    report_coarse(df)
    if args.manifest:
        report_short(df, Path(args.manifest), args.short_chars)
    else:
        print("\n[짧은 발화] --manifest를 주지 않아 건너뜀")

    if args.ablation:
        report_ablation(df, args.ablation)
    if args.compare:
        other = load_pred(Path(args.compare))
        mcnemar(df, other, pred_path.stem, Path(args.compare).stem)
    report_calibration(df, Path(args.calibrate) if args.calibrate else None)
    print()


if __name__ == "__main__":
    main()
