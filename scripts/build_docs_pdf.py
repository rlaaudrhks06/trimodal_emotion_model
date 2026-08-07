"""docs/*.md -> PDF. 헤드리스 Chrome으로 인쇄한다.

왜 스크립트로 두는가: 그동안 임시 폴더의 일회성 스크립트로 만들어와서 재현이 안 됐다.
실제로 한 번은 "출력 파일의 앞부분을 잘라 head로 재사용"하는 방식이었는데, 그 파일을
같은 스크립트가 덮어쓰는 바람에 스타일이 늘어나자 head가 반토막 나 989바이트짜리 빈
PDF가 만들어졌다. 스타일을 코드 안에 두어 그 종류의 사고를 없앤다.

pandoc·weasyprint 같은 추가 의존성을 쓰지 않는 이유: 한글 폰트 설정이 까다롭고,
맥에 이미 있는 Chrome이 시스템 폰트로 잘 렌더링한다. 필요한 파이썬 패키지는
`markdown` 하나뿐이다.

실행:
    python scripts/build_docs_pdf.py            # docs/*.md 전부
    python scripts/build_docs_pdf.py 통합기록    # 이름에 그 문자열이 든 것만
"""
import argparse
import subprocess
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
  body { font-family: -apple-system, "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
         line-height: 1.6; color: #1a1a1a; max-width: 900px; margin: 0 auto;
         padding: 40px; font-size: 13.5px; }
  h1 { font-size: 23px; border-bottom: 3px solid #333; padding-bottom: 10px; }
  h2 { font-size: 18px; margin-top: 30px; border-bottom: 1px solid #ccc; padding-bottom: 6px; }
  h3 { font-size: 15px; margin-top: 22px; color: #222; }
  h4 { font-size: 13.8px; margin-top: 18px; color: #333; }
  table { border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 11.5px; }
  th, td { border: 1px solid #999; padding: 5px 7px; text-align: left; vertical-align: top; }
  th { background: #eee; }
  code { background: #f2f2f2; padding: 1px 5px; border-radius: 3px; font-size: 11.5px; }
  pre { background: #f2f2f2; padding: 10px; border-radius: 5px; overflow-x: auto;
        font-size: 9.5px; line-height: 1.35; white-space: pre; }
  pre code { background: none; padding: 0; }
  hr { border: none; border-top: 1px solid #ccc; margin: 24px 0; }
  strong { color: #000; }
  img { max-width: 100%; display: block; margin: 12px 0;
        border: 1px solid #ddd; border-radius: 6px; }
  details { margin: 14px 0; }
  summary { cursor: pointer; font-weight: 600; color: #555; }
  .toc { background: #fafaf7; border: 1px solid #ddd; border-radius: 6px;
         padding: 14px 20px; margin: 14px 0 28px; font-size: 12.5px; }
  .toc > ul { margin: 0; padding-left: 18px; }
  .toc ul { list-style: none; padding-left: 14px; margin: 3px 0; }
  .toc > ul > li > a { font-weight: 700; }
  .toc li { margin: 2.5px 0; line-height: 1.45; }
  .toc a { color: #1a1a1a; text-decoration: none; }
"""


def build(md_path: Path, keep_html: bool) -> bool:
    import markdown

    title = md_path.stem
    body = markdown.markdown(
        md_path.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "toc"],
        # 2~3단계까지만 — 4단계(####)까지 넣으면 목차가 본문만큼 길어진다
        extension_configs={"toc": {"toc_depth": "2-3"}},
    )
    # base href: 문서가 assets/*.png를 상대경로로 참조하므로 docs 디렉터리를 기준으로 삼는다
    html = (f'<!DOCTYPE html>\n<html lang="ko">\n<head>\n<meta charset="UTF-8">\n'
            f'<base href="file://{DOCS}/">\n<title>{title}</title>\n'
            f'<style>{CSS}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n')

    html_path = md_path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")
    pdf_path = md_path.with_suffix(".pdf")

    if not Path(CHROME).exists():
        print(f"  [실패] Chrome을 찾을 수 없음: {CHROME}")
        return False
    r = subprocess.run(
        [CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={pdf_path}", str(html_path)],
        capture_output=True, text=True, timeout=300,
    )
    if not keep_html:
        html_path.unlink(missing_ok=True)

    if not pdf_path.exists():
        print(f"  [실패] {md_path.name}\n{(r.stdout + r.stderr)[-400:]}")
        return False
    size = pdf_path.stat().st_size
    # 빈 PDF(과거에 989바이트짜리가 나온 적 있음)를 조용히 넘기지 않는다
    if size < 50_000:
        print(f"  [경고] {pdf_path.name} 이 {size:,}바이트뿐이다 — 렌더링 실패 가능성")
        return False
    print(f"  {pdf_path.name}  {size:,} bytes")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("filter", nargs="?", default="",
                    help="파일명에 이 문자열이 든 것만 빌드 (생략 시 전부)")
    ap.add_argument("--keep-html", action="store_true", help="중간 HTML을 지우지 않는다")
    args = ap.parse_args()

    try:
        import markdown  # noqa: F401
    except ImportError:
        sys.exit("markdown 패키지가 필요하다:  pip install markdown")

    targets = sorted(p for p in DOCS.glob("*.md") if args.filter in p.name)
    if not targets:
        sys.exit(f"대상 없음: {DOCS}/*{args.filter}*.md")

    print(f"[build_docs] {len(targets)}개 문서")
    ok = sum(build(p, args.keep_html) for p in targets)
    print(f"[build_docs] {ok}/{len(targets)} 성공")
    if ok != len(targets):
        sys.exit(1)


if __name__ == "__main__":
    main()
