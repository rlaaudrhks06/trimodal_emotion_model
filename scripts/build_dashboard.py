"""results/ 의 CSV·JSON에서 학습 대시보드 HTML을 만든다.

왜 스크립트로 두는가: 수치를 손으로 옮겨 적으면 반드시 틀린다. 실제로 모델 카드의
train_loss 6줄이 그렇게 어긋난 적이 있다(8.19절). 여기서는 `results/csv_summary/`와
`results/eval/`을 직접 읽어 HTML에 박아 넣으므로, 새 실험이 끝나면 다시 돌리기만 하면 된다.

주의: 학습 곡선 CSV의 정확도는 **소수(0~1)** 스케일이다. 표시할 때만 100을 곱한다.

실행:
    python scripts/build_dashboard.py                 # results/dashboard.html
    python scripts/build_dashboard.py --out /tmp/a.html
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = ROOT / "results" / "csv_summary"
EVAL_DIR = ROOT / "results" / "eval"

# 화자 독립(v10~) 과 화자 누수(v1~v9)는 **비교 불가**다. 조건을 색이 아니라
# 구획으로 갈라 놓아야 눈으로 섞이지 않는다.
CLEAN = ["v10", "v11", "v12a", "v12b"]
LEAKY = ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9"]

HUE = {
    "v1": "#8a8271", "v2": "#3f7bb0", "v3": "#c97a2e", "v4": "#2f9468",
    "v5": "#4a7ec9", "v6": "#d1487a", "v7": "#1c8c8c", "v8": "#a8760a",
    "v9": "#16355e", "v10": "#6b7280", "v11": "#0f8b7e",
    "v12a": "#c2185b", "v12b": "#d97706",
}
NOTE = {
    "v10": "멜 오디오 · 화자 독립 첫 기준선",
    "v11": "wav2vec2 오디오 백본 — 현재 최고",
    "v12a": "wav2vec2 24층 — 개선 없음",
    "v12b": "모달리티별 보조 학습 — 개선 없음",
}

# 8.30절(A단계) 결과. 서버에서 나온 값이며 통합기록·모델카드와 같은 출처다.
# 해당 eval JSON이 로컬에 있으면 그쪽을 우선 읽는다(아래 load_eval 참고).
ABLATION = [("전체", 46.19), ("텍스트 제거", 39.60), ("오디오 제거", 42.50), ("영상 제거", 42.87)]
NOISE = [("깨끗", None, 46.19), ("SNR 20dB", 20, 43.92), ("SNR 10dB", 10, 37.18)]
ABSTAIN = [(0.0, 100.0, 46.19), (0.3, 89.1, 48.41), (0.4, 64.2, 54.06),
           (0.5, 40.4, 60.66), (0.6, 22.6, 68.91), (0.7, 12.1, 75.29), (0.8, 4.4, 82.40)]


def load_runs() -> dict:
    out = {}
    for f in CSV_DIR.glob("checkpoint_v*_epoch_summary.csv"):
        v = f.name.split("_")[1]
        d = pd.read_csv(f)
        need = {"epoch", "train_acc", "val_acc"}
        if not need <= set(d.columns):
            raise ValueError(f"{f.name}: 컬럼 부족 {sorted(need - set(d.columns))}")
        if d["val_acc"].max() > 1.5:
            raise ValueError(f"{f.name}: val_acc가 소수 스케일이 아니다(최대 {d['val_acc'].max()})")
        out[v] = [{"e": int(r.epoch), "t": round(100 * r.train_acc, 2),
                   "v": round(100 * r.val_acc, 2)} for r in d.itertuples()]
    return out


def load_eval() -> dict:
    """results/eval/*.json에서 test 정확도를 읽는다. 없으면 그냥 비운다."""
    out = {}
    for f in EVAL_DIR.glob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "accuracy" in d:
            out[f.stem] = round(100 * d["accuracy"], 2)
    return out


def polyline(rows, key, x0, y0, w, h, xmax, ylo, yhi):
    pts = []
    for r in rows:
        x = x0 + (r["e"] - 1) / max(xmax - 1, 1) * w
        y = y0 + h - (r[key] - ylo) / (yhi - ylo) * h
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts)


def chart(runs, names, title, ylo, yhi, w=860, h=290):
    """val_acc 꺾은선 + 최고점 표시. SVG를 직접 만든다(외부 라이브러리 금지)."""
    pad_l, pad_b, pad_t, pad_r = 44, 30, 14, 96
    iw, ih = w - pad_l - pad_r, h - pad_t - pad_b
    xmax = max(len(runs[n]) for n in names)
    s = [f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px" role="img">']
    # 가로 눈금
    step = 2 if (yhi - ylo) <= 12 else 5
    y = ylo
    while y <= yhi + 0.01:
        yy = pad_t + ih - (y - ylo) / (yhi - ylo) * ih
        s.append(f'<line class="gl" x1="{pad_l}" y1="{yy:.1f}" x2="{pad_l+iw}" y2="{yy:.1f}"/>')
        s.append(f'<text class="tk" x="{pad_l-8}" y="{yy+3.5:.1f}" text-anchor="end">{y:.0f}%</text>')
        y += step
    # 세로 눈금
    for e in range(1, xmax + 1):
        if xmax > 20 and e % 10 or xmax <= 20 and e % 5:
            continue
        xx = pad_l + (e - 1) / max(xmax - 1, 1) * iw
        s.append(f'<text class="tk" x="{xx:.1f}" y="{pad_t+ih+16}" text-anchor="middle">{e}</text>')
    for n in names:
        rows = runs[n]
        c = HUE[n]
        s.append(f'<polyline fill="none" stroke="{c}" stroke-width="2" stroke-linejoin="round" '
                 f'points="{polyline(rows, "v", pad_l, pad_t, iw, ih, xmax, ylo, yhi)}"/>')
        b = max(rows, key=lambda r: r["v"])
        bx = pad_l + (b["e"] - 1) / max(xmax - 1, 1) * iw
        by = pad_t + ih - (b["v"] - ylo) / (yhi - ylo) * ih
        s.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="4" fill="{c}" class="bd"/>')
        s.append(f'<text class="bl" x="{pad_l+iw+8}" y="{by+3.5:.1f}" fill="{c}">'
                 f'{n} {b["v"]:.2f}%</text>')
    s.append(f'<text class="ax" x="{pad_l+iw/2:.0f}" y="{h-2}" text-anchor="middle">에폭</text>')
    s.append("</svg>")
    return "".join(s)


def gap_cards(runs, names):
    out = []
    for n in names:
        rows = runs[n]
        b = max(rows, key=lambda r: r["v"])
        gap = rows[-1]["t"] - rows[-1]["v"]
        w, h, pl, pt = 240, 92, 4, 6
        lo = min(min(r["v"] for r in rows), min(r["t"] for r in rows)) - 2
        hi = max(max(r["t"] for r in rows), max(r["v"] for r in rows)) + 2
        xmax = len(rows)
        pv = polyline(rows, "v", pl, pt, w - 8, h - 12, xmax, lo, hi)
        pt_ = polyline(rows, "t", pl, pt, w - 8, h - 12, xmax, lo, hi)
        c = HUE[n]
        out.append(
            f'<div class="mc"><div class="mh"><span class="mt" style="color:{c}">{n}</span>'
            f'<span class="mg">격차 {gap:.1f}pt</span></div>'
            f'<svg viewBox="0 0 {w} {h}" width="100%"><polyline fill="none" stroke="{c}" '
            f'stroke-width="1.6" points="{pv}"/><polyline fill="none" stroke="{c}" '
            f'stroke-width="1.2" stroke-dasharray="3 3" opacity=".55" points="{pt_}"/></svg>'
            f'<div class="ms">최고 val {b["v"]:.2f}% · {b["e"]}에폭 · {len(rows)}에폭 학습</div></div>')
    return "".join(out)


def bars(items, vmax, unit="%"):
    out = []
    for name, val in items:
        pct = val / vmax * 100
        hl = " hl" if name == "전체" else ""
        out.append(f'<div class="bar{hl}"><span class="bn">{name}</span>'
                   f'<span class="bt"><i style="width:{pct:.1f}%"></i></span>'
                   f'<span class="bv">{val:.2f}{unit}</span></div>')
    return "".join(out)


def build(runs, evals) -> str:
    tpl = (ROOT / "scripts" / "dashboard_template.html").read_text(encoding="utf-8")
    best = max(CLEAN, key=lambda v: max(r["v"] for r in runs[v]))
    rows = []
    for v in CLEAN + LEAKY:
        b = max(runs[v], key=lambda x: x["v"])
        is_clean = v in CLEAN
        test = evals.get(v)
        test_cell = f"{test:.2f}%" if test is not None else "—"
        rows.append(
            f'<tr class="{"clean" if is_clean else "leak"}">'
            f'<td><span class="vc"><i style="background:{HUE[v]}"></i>{v}</span></td>'
            f'<td class="cond">{"화자 독립" if is_clean else "화자 누수"}</td>'
            f'<td class="n">{b["v"]:.2f}%</td>'
            f'<td class="n">{b["e"]}</td>'
            f'<td class="n">{test_cell}</td>'
            f'<td class="vd">{NOTE.get(v, "")}</td></tr>'
        )

    return (tpl
            .replace("{{CHART_CLEAN}}", chart(runs, CLEAN, "화자 독립", 38, 48))
            .replace("{{CHART_LEAKY}}", chart(runs, LEAKY, "화자 누수", 24, 50))
            .replace("{{GAPS}}", gap_cards(runs, CLEAN))
            .replace("{{TABLE}}", "".join(rows))
            .replace("{{ABLATION}}", bars(ABLATION, 50))
            .replace("{{NOISE}}", bars([(n, a) for n, _, a in NOISE], 50))
            .replace("{{ABSTAIN}}", "".join(
                f'<tr><td class="n">{t:.1f}</td><td class="n">{c:.1f}%</td>'
                f'<td class="n hl">{a:.2f}%</td></tr>' for t, c, a in ABSTAIN))
            .replace("{{BEST}}", best))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "dashboard.html"))
    args = ap.parse_args()
    runs = load_runs()
    missing = [v for v in CLEAN + LEAKY if v not in runs]
    if missing:
        raise SystemExit(f"학습 곡선 CSV가 없다: {missing}")
    evals = load_eval()
    html = build(runs, evals)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"[dashboard] {out}  {out.stat().st_size:,} bytes")
    print(f"[dashboard] 실행 {len(runs)}개, test 결과 {len(evals)}개 반영")


if __name__ == "__main__":
    main()
