"""v11 계열 test 결과를 시드별 박스 표 + 3시드 평균 표로 출력한다 (results/eval/*.json 기준).

    .venv/bin/python scripts/results_table.py            # 화면 출력
    .venv/bin/python scripts/results_table.py --md       # 문서 붙여넣기용 (코드 블록 포함)

없는 결과는 '—'. 새 run이 생기면 RUNS에 한 줄 추가한다.
"""
import json, os, sys, unicodedata
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
RUNS = [("v11 (기준)",              ["v11", "v11_s43", "v11_s44"]),
        ("v11a (소음증강)",          ["v11a", "v11a_s43", "v11a_s44"]),
        ("v11b (깨끗50%)",           ["v11b"]),
        ("v11c (L24)",               ["v11c_s42", "v11c_s43", "v11c_s44"]),
        ("v11d (+263)",              ["v11d_s42", "v11d_s43", "v11d_s44"]),
        ("v11e (9~12층 미세조정)",    ["v11e_s42", "v11e_s43", "v11e_s44"]),
        ("v11e_no263 (263 뺌)",      ["v11e_no263_s42", "v11e_no263_s43", "v11e_no263_s44"]),
        ("v11f (5~12층)",            ["v11f_s42", "v11f_s43", "v11f_s44"]),
        ("v11g (3클래스 머리)",       ["v11g_s42", "v11g_s43", "v11g_s44"])]
COND = [("깨끗", ""), ("SNR20", "_noise20"), ("SNR10", "_noise10")]


def acc(n):
    p = ROOT / "results" / "eval" / f"{n}.json"
    return json.load(open(p))["accuracy"] * 100 if p.exists() else None


def w(s): return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)
def pad(s, n): return s + " " * (n - w(s))


def box(hdr, rows):
    W = [max(w(r[i]) for r in [hdr] + rows) + 2 for i in range(len(hdr))]
    line = lambda l, m, r: l + m.join("─" * x for x in W) + r
    row = lambda r: "│" + "│".join(" " + pad(c, W[i] - 1) for i, c in enumerate(r)) + "│"
    out = [line("┌", "┬", "┐"), row(hdr), line("├", "┼", "┤")]
    for r in rows: out += [row(r), line("├", "┼", "┤")]
    out[-1] = line("└", "┴", "┘")
    return "\n".join(out)


def main():
    md = "--md" in sys.argv
    rows = []
    for name, rs in RUNS:
        label = name + ("  (s42/43/44)" if len(rs) > 1 else "  (s42)")
        rows.append([label] + [" / ".join("—" if (v := acc(r + suf)) is None else f"{v:.2f}" for r in rs) for _, suf in COND])
    t1 = box([""] + [c for c, _ in COND], rows)
    rows2 = []
    for name, rs in RUNS:
        vals = [[acc(r + suf) for r in rs] for _, suf in COND]
        vals = [[x for x in v if x is not None] for v in vals]
        if len(vals[0]) < 2: continue
        rows2.append([name.split(" (")[0] + (f" ({len(vals[0])}시드)" if len(vals[0]) != 3 else "")]
                     + [f"{np.mean(v):.2f} ± {np.std(v, ddof=1):.2f}" for v in vals])
    t2 = box([""] + [c for c, _ in COND], rows2)
    if md:
        print("```\n" + t1 + "\n```\n\n3시드 평균 (± 표본표준편차):\n\n```\n" + t2 + "\n```")
    else:
        print(t1); print(); print(t2)


if __name__ == "__main__":
    main()
