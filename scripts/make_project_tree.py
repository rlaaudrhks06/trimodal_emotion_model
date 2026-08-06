"""프로젝트 폴더 구조와 용량 현황을 project_tree.txt로 남긴다.

왜 스크립트로 두는가: 이 파일은 **서버 상태**를 기록한 것이라 로컬에서 만들 수 없다
(체크포인트·캐시가 서버에만 있음). 그동안 수작업으로 만들어서 갱신이 밀렸고, 어떤
기준으로 무엇을 제외했는지도 파일 첫 줄 주석에만 남아 재현이 안 됐다.

용량이 큰 디렉터리(캐시·원본·체크포인트 내용물)는 트리에서 접고 용량만 따로 집계한다 —
파일이 수십만 개라 다 펼치면 읽을 수 없기 때문이다.

실행:
    python scripts/make_project_tree.py            # project_tree.txt 갱신
    python scripts/make_project_tree.py --stdout   # 화면에만 출력
"""
import argparse
import subprocess
from pathlib import Path

# 트리에서 내용을 펼치지 않을 디렉터리(이름 기준). 존재 여부는 그대로 표시한다.
COLLAPSE = {
    ".git", ".venv", "__pycache__", "node_modules",
    "feature_cache", "processed_full", "processed", "raw", "models",
    "embeddings",
}
# 용량을 따로 집계할 최상위 디렉터리
SIZE_TARGETS = ["archived_runs", "data", "results", "models"]


def walk(root: Path, depth: int = 0, lines: list | None = None) -> list:
    lines = [] if lines is None else lines
    try:
        entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except PermissionError:
        return lines
    for p in entries:
        pad = "  " * (depth + 1)
        if p.is_dir():
            if p.name in COLLAPSE:
                # 내용은 접되 몇 개 들어있는지는 알려준다 — 비어 있는지 구분하려는 것
                try:
                    n = sum(1 for _ in p.iterdir())
                except (PermissionError, OSError):
                    n = -1
                note = "(비어 있음)" if n == 0 else (f"({n}개 항목, 접힘)" if n > 0 else "(접근 불가)")
                lines.append(f"{pad}{p.name}/ {note}")
            else:
                lines.append(f"{pad}{p.name}")
                walk(p, depth + 1, lines)
        else:
            lines.append(f"{pad}{p.name}")
    return lines


def human(kb: int) -> str:
    v = float(kb)
    for unit in ("K", "M", "G", "T"):
        if v < 1024 or unit == "T":
            return f"{v:.1f}{unit}".replace(".0", "")
        v /= 1024
    return f"{v:.1f}T"


def du(path: Path, root: Path) -> list[str]:
    """디렉터리별 용량을 **작은 것부터** 정렬해 돌려준다.

    `du -sh | sort` 는 문자열 정렬이라 231M이 3.8M보다 앞에 온다. 그래서 -sk(KB)로
    받아 숫자로 정렬한 뒤 사람이 읽을 단위로 다시 만든다. 경로도 root 기준 상대경로로
    줄여서 서버/로컬 어디서 만들어도 같은 모양이 되게 한다.
    """
    if not path.exists():
        return [f"  (없음: {path.relative_to(root) if path.is_relative_to(root) else path})"]
    children = sorted(path.iterdir())
    if not children:
        return ["  (비어 있음)"]
    try:
        out = subprocess.run(["du", "-sk", "--", *[str(c) for c in children]],
                             capture_output=True, text=True, timeout=600)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ["  (du 실행 불가)"]
    rows = []
    for line in out.stdout.rstrip().splitlines():
        size, _, p = line.partition("\t")
        try:
            kb = int(size)
        except ValueError:
            continue
        rel = Path(p)
        rows.append((kb, f"  {human(kb):>8}  {rel.relative_to(root) if rel.is_relative_to(root) else rel}"))
    return [r for _, r in sorted(rows)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()

    root = args.root
    out_path = args.out or (root / "project_tree.txt")

    parts = [
        f"=== 전체 구조 (아래 디렉터리는 내용을 접음: {', '.join(sorted(COLLAPSE))}) ===",
        ".",
        *walk(root),
        "",
    ]

    for name in SIZE_TARGETS:
        p = root / name
        parts.append(f"=== 용량 ({name}) ===")
        parts.extend(du(p, root))
        parts.append("")

    parts.append("=== 디스크 여유 ===")
    try:
        df = subprocess.run(["df", "-h", str(root)], capture_output=True, text=True, timeout=30)
        parts.extend(df.stdout.rstrip().splitlines())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        parts.append("(df 실행 불가)")

    text = "\n".join(parts) + "\n"
    if args.stdout:
        print(text)
    else:
        out_path.write_text(text, encoding="utf-8")
        print(f"[tree] 갱신: {out_path} ({len(parts)}줄)")


if __name__ == "__main__":
    main()
