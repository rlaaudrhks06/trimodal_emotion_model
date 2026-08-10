"""학습 곡선 차트 PNG를 만든다 — 버전별 val_acc 비교, train/val 격차.

왜 스크립트로 두는가: `docs/assets/training_*.png`는 일회성으로 만들어져 있어서
새 버전이 나올 때마다 다시 만들 방법이 없었다. 실제로 v11·v12가 나온 뒤에도
차트는 v10까지만 담고 있었다. 문서 PDF·대시보드·설계도와 같은 이유로 생성기를 둔다.

**조건이 다른 두 그룹을 한 그림에 그린다는 점을 주의해야 한다.** v1~v9는 화자 누수가
있던 분할이고 v10 이후는 화자 독립 분할이라 val셋 자체가 다르다. 실선/점선과 서로 다른
기준선으로 갈라 두지 않으면, 6.51%p의 하락이 "성능 저하"로 잘못 읽힌다.

실행:
    python scripts/build_training_charts.py
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = ROOT / "results" / "csv_summary"
ASSETS = ROOT / "docs" / "assets"

# 화자 누수 조건(v1~v9)과 화자 독립 조건(v10~)은 val셋이 다르다 — 직접 비교 불가.
LEAKY = [
    ("v1", "v1 · 400클립", "#8a8271"),
    ("v2", "v2 · 80k 무규제", "#4a7ec9"),
    ("v3", "v3 · +규제", "#d97706"),
    ("v4", "v4 · +prosody 정규화", "#2f9468"),
    ("v5", "v5 · 얼굴 크롭 버그 수정", "#9b6bd6"),
    ("v6", "v6 · 8→7클래스 병합", "#e0578f"),
    ("v7", "v7 · BERT 완전 동결", "#26b2a8"),
    ("v8", "v8 · 차등 학습률", "#c9a227"),
    ("v9", "v9 · +DropPath", "#16355e"),
]
CLEAN = [
    ("v10", "v10 · 화자 독립 분할", "#111111"),
    ("v11", "v11 · +wav2vec2 오디오", "#0f8b7e"),
    ("v12a", "v12a · wav2vec2 24층", "#c2185b"),
    ("v12b", "v12b · 모달리티별 보조학습", "#d97706"),
]
BASE_LEAKY, BASE_CLEAN = 23.86, 27.00


def pick_font() -> str:
    names = {f.name for f in fm.fontManager.ttflist}
    # 볼드 웨이트가 있는 것을 먼저 고른다 — AppleGothic은 regular만 있어 경고가 난다.
    for c in ("Apple SD Gothic Neo", "NanumGothic", "AppleGothic", "Malgun Gothic",
              "Noto Sans KR", "Arial Unicode MS"):
        if c in names:
            return c
    raise SystemExit("한글 폰트를 못 찾았다 — 축 라벨이 깨지므로 중단한다")


def load(v: str) -> pd.DataFrame:
    f = CSV_DIR / f"checkpoint_{v}_epoch_summary.csv"
    if not f.exists():
        raise SystemExit(f"{f.name} 없음")
    d = pd.read_csv(f)
    if d["val_acc"].max() > 1.5:
        raise SystemExit(f"{f.name}: val_acc가 소수 스케일이 아니다")
    return d


def val_acc_chart(out: Path) -> None:
    """좌우 두 패널로 나눈다.

    13개 선을 한 축에 그리면 최고점 라벨이 서로 겹쳐 읽을 수 없고, 무엇보다
    **조건이 다른 두 그룹이 같은 잣대처럼 보인다.** 패널을 갈라야 "v9 48.69%가
    v11 46.80%보다 높다"는 잘못된 읽기를 막을 수 있다.
    """
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(16.5, 7.6), dpi=150,
        gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.16})

    # ---- 왼쪽: 화자 누수 조건 (v1~v9) ----
    for v, label, c in LEAKY:
        d = load(v)
        axL.plot(d["epoch"], 100 * d["val_acc"], color=c, lw=1.7, label=label, zorder=2)
        b = d.loc[d["val_acc"].idxmax()]
        axL.plot(b["epoch"], 100 * b["val_acc"], "o", color=c, ms=7,
                 mec="white", mew=1.4, zorder=4)
    axL.axhline(BASE_LEAKY, color="#c76a72", ls="--", lw=1.2, zorder=1)
    axL.text(1.5, BASE_LEAKY + 0.5, f"다수 클래스 기준선 {BASE_LEAKY:.2f}%",
             fontsize=9.5, color="#c76a72")
    axL.set_title("화자 누수 조건 — v1 ~ v9", fontsize=13, pad=10, color="#a8760a")
    axL.set_ylim(15, 52)
    axL.set_ylabel("검증 정확도 (val_acc, %)", fontsize=11.5)
    axL.legend(loc="lower right", fontsize=9.5, framealpha=0.96, edgecolor="#d5d9de")

    # ---- 오른쪽: 화자 독립 조건 (v10~) · 축 확대 ----
    # v10만 74에폭까지 가는데 나머지는 12~16에폭이라, 전체를 그리면 정작 볼 구간이
    # 왼쪽 20%에 눌린다. 최고점은 전부 12에폭 이내이므로 x를 잘라 확대한다.
    XMAX = 22
    # 라벨이 서로 겹치지 않게 위/아래로 엇갈린다.
    offs = {"v10": (10, -13), "v11": (8, 8), "v12a": (7, -20), "v12b": (-10, 9)}
    for v, label, c in CLEAN:
        d = load(v)
        axR.plot(d["epoch"], 100 * d["val_acc"], color=c, lw=2.3, label=label, zorder=3)
        b = d.loc[d["val_acc"].idxmax()]
        axR.plot(b["epoch"], 100 * b["val_acc"], "o", color=c, ms=8,
                 mec="white", mew=1.6, zorder=5)
        axR.annotate(f"{100*b['val_acc']:.2f}%", (b["epoch"], 100 * b["val_acc"]),
                     textcoords="offset points", xytext=offs[v],
                     fontsize=10, color=c, fontweight="bold", zorder=6)
    axR.axhline(BASE_CLEAN, color="#333", ls=":", lw=1.3, zorder=1)
    axR.text(XMAX - 0.6, BASE_CLEAN + 0.35, f"다수 클래스 기준선 {BASE_CLEAN:.2f}%",
             fontsize=9.5, color="#333", ha="right")
    axR.set_title("화자 독립 조건 — v10 ~ v12b  (가로·세로축 확대)",
                  fontsize=13, pad=10, color="#0b6b60")
    axR.set_ylim(26, 49)
    axR.set_xlim(0, XMAX)
    axR.text(XMAX - 0.6, 29.0, "v10은 74에폭까지 이어지나 이후 평탄",
             fontsize=9, color="#9aa0a8", ha="right", style="italic")
    axR.legend(loc="lower left", fontsize=9.5, framealpha=0.96, edgecolor="#d5d9de")

    for ax in (axL, axR):
        ax.set_xlabel("에폭", fontsize=11.5)
        ax.grid(alpha=0.25, lw=0.7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.suptitle("버전별 val_acc 추이 (v1 → v12b) — 점은 각 실험의 최고 지점",
                 fontsize=15, y=0.975)
    fig.text(0.5, 0.015,
             "왼쪽과 오른쪽은 val셋이 다르므로 직접 비교하면 안 된다. "
             "같은 레시피로 화자를 분리하자 실질 6.51%p가 사라졌다(8.14절) — "
             "왼쪽의 높은 점수 중 그만큼은 '사람을 외운 몫'이었다. "
             "오른쪽은 범위가 좁아 세로축을 확대했다.",
             ha="center", fontsize=9.5, color="#6b6f76")

    fig.subplots_adjust(left=0.055, right=0.985, top=0.885, bottom=0.115, wspace=0.16)
    fig.savefig(out, facecolor="white")
    plt.close(fig)


# 화자 독립 실험의 test 정확도(8.21·8.26·8.27절). v12a는 test 미측정.
TEST_ACC = {"v10": 44.15, "v11": 46.19, "v12a": None, "v12b": 45.22}
VERDICT = {"v10": "화자 독립 첫 기준선",
           "v11": "현재 최고",
           "v12a": "개선 없음 (val 기준)",
           "v12b": "val은 최고, test에서 뒤집힘"}


def clean_only_chart(out: Path) -> None:
    """화자 독립 조건 4개만 크게. 보고서·발표용.

    val 곡선 옆에 test 결과를 함께 적는다 — v12b가 **val에서는 v11을 넘고도
    test에서 뒤집힌** 것이 이 프로젝트의 핵심 교훈 중 하나인데, val 곡선만
    보여주면 정반대로 읽힌다.
    """
    fig, ax = plt.subplots(figsize=(13.5, 7.2), dpi=150)
    XMAX = 17
    offs = {"v10": (11, 6), "v11": (9, 10), "v12a": (6, -22), "v12b": (-14, 11)}

    for v, label, c in CLEAN:
        d = load(v)
        d = d[d["epoch"] <= XMAX]
        ax.plot(d["epoch"], 100 * d["val_acc"], color=c, lw=2.6,
                marker="o", ms=4.5, label=label, zorder=3)
        b = d.loc[d["val_acc"].idxmax()]
        ax.plot(b["epoch"], 100 * b["val_acc"], "o", color=c, ms=11,
                mec="white", mew=2, zorder=5)
        ax.annotate(f"{100*b['val_acc']:.2f}%", (b["epoch"], 100 * b["val_acc"]),
                    textcoords="offset points", xytext=offs[v],
                    fontsize=11, color=c, fontweight="bold", zorder=6)

    ax.axhline(BASE_CLEAN, color="#333", ls=":", lw=1.4)
    ax.text(XMAX / 2, BASE_CLEAN + 0.3, f"다수 클래스 기준선 {BASE_CLEAN:.2f}%",
            fontsize=10, color="#333", ha="center")

    ax.set_title("화자 독립 조건의 최근 실험 — v10 ~ v12b", fontsize=15, pad=12)
    ax.set_xlabel("에폭", fontsize=12)
    ax.set_ylabel("검증 정확도 (val_acc, %)", fontsize=12)
    ax.set_xlim(0.5, XMAX + 0.5)
    ax.set_ylim(26, 49)
    ax.grid(alpha=0.25, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", fontsize=10.5, framealpha=0.96, edgecolor="#d5d9de")

    # val 최고점만 보면 v12b가 1등이지만 test는 v11이 1등이다 — 나란히 적는다.
    # 한글 폰트는 monospace가 아니고 DejaVu Sans Mono에는 한글이 없다.
    # 그래서 표는 숫자만 monospace로 두고, 판정 문구는 아래 캡션에 맡긴다.
    lines = ["        best val    test"]
    for v, _, _ in CLEAN:
        d = load(v)
        t = TEST_ACC[v]
        lines.append(f"{v:5} {100*d['val_acc'].max():6.2f}%  "
                     + (f"{t:6.2f}%" if t else "     --"))
    ax.text(1.0, 32.4, "\n".join(lines), fontsize=11, family="monospace",
            va="top", ha="left", color="#3f4650",
            bbox=dict(boxstyle="round,pad=0.6", fc="#f6f7f8", ec="#d5d9de", lw=1))


    fig.text(0.5, 0.015,
             "val 최고점만 보면 v12b가 가장 높지만 test에서는 v11이 앞선다 — "
             "체크포인트는 val 최고 시점에서 고르되, 판정은 test로 한다. "
             "v11과 v12b의 test 차이 0.97%p는 단일 실행으로는 구분되지 않는다(8.29.1절).",
             ha="center", fontsize=9.5, color="#6b6f76")

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def gap_chart(out: Path) -> None:
    """화자 독립 실험들의 train/val 격차 — 과적합이 언제 시작되는가."""
    fig, ax = plt.subplots(figsize=(13, 6.4), dpi=150)
    for v, label, c in CLEAN:
        d = load(v)
        ax.plot(d["epoch"], 100 * (d["train_acc"] - d["val_acc"]),
                color=c, lw=2.2, label=label, marker="o", ms=4)
    ax.axhline(0, color="#999", lw=1)
    ax.set_title("화자 독립 실험의 과적합 격차 (train_acc - val_acc)", fontsize=14, pad=12)
    ax.set_xlabel("에폭", fontsize=11.5)
    ax.set_ylabel("격차 (%p)", fontsize=11.5)
    ax.grid(alpha=0.25, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.96, edgecolor="#d5d9de")
    fig.text(0.5, 0.015,
             "격차가 벌어지기 시작하는 지점부터가 과적합 구간이다. "
             "v12a는 격차가 가장 작지만 val 자체가 낮았다 — 규제가 과했다는 뜻이다.",
             ha="center", fontsize=9.5, color="#6b6f76")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["val", "clean", "gap"], default=None)
    args = ap.parse_args()

    font = pick_font()
    plt.rcParams["font.family"] = font
    # 한글 폰트에는 유니코드 마이너스(U+2212)가 없어 축 눈금이 네모로 깨진다.
    plt.rcParams["axes.unicode_minus"] = False
    print(f"[charts] 한글 폰트: {font}")

    ASSETS.mkdir(parents=True, exist_ok=True)
    jobs = []
    if args.only in (None, "val"):
        jobs.append(("training_val_acc_comparison.png", val_acc_chart))
    if args.only in (None, "clean"):
        jobs.append(("training_val_acc_clean_only.png", clean_only_chart))
    if args.only in (None, "gap"):
        jobs.append(("training_train_val_gap.png", gap_chart))
    for name, fn in jobs:
        out = ASSETS / name
        fn(out)
        print(f"[charts] {out.relative_to(ROOT)}  {out.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
