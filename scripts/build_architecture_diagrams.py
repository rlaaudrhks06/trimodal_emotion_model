"""설계도 PNG를 만든다 — 처음 설계(이중모달 v2)와 현재 설계(v11).

왜 스크립트로 두는가: `docs/assets/architecture_current.png`는 일회성으로 만들어져
있어서, 설계가 바뀌면 다시 만들 방법이 남아 있지 않았다. 문서 PDF·대시보드와 같은
이유로 생성기를 코드에 둔다.

방법: HTML/CSS로 그리고 헤드리스 Chrome으로 캡처한다. matplotlib보다 한글 조판과
박스 정렬이 훨씬 정확하고, `build_docs_pdf.py`가 이미 쓰는 도구라 의존성이 늘지 않는다.

실행:
    python scripts/build_architecture_diagrams.py            # 둘 다
    python scripts/build_architecture_diagrams.py --only v11
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{background:#fff;font-family:-apple-system,"Apple SD Gothic Neo","Noto Sans KR",sans-serif;
  color:#1b1d22;-webkit-font-smoothing:antialiased}
.sheet{width:1180px;padding:38px 44px 34px}
.title{font-size:27px;font-weight:750;letter-spacing:-.02em;margin-bottom:6px}
.sub{font-size:14px;color:#6b6f76;margin-bottom:4px}
.meta{font-size:12.5px;color:#9aa0a8;margin-bottom:26px}
.meta b{color:#4a4f57;font-weight:600}

.stage{display:flex;align-items:flex-start;gap:14px;margin-bottom:6px}
.num{width:26px;height:26px;border-radius:50%;background:#1b1d22;color:#fff;font-size:12.5px;
  font-weight:700;display:flex;align-items:center;justify-content:center;flex:none;margin-top:5px}
.stage-body{flex:1}
.stage-name{font-size:14px;font-weight:700;margin-bottom:2px}
.stage-why{font-size:12px;color:#8a9098;margin-bottom:9px;line-height:1.45}

.row{display:flex;gap:11px;align-items:stretch}
.box{border:1.6px solid;border-radius:9px;padding:10px 13px;background:#fff;flex:1;min-width:0}
.box .bt{font-size:13px;font-weight:650;line-height:1.35}
.box .bs{font-size:11.5px;color:#7b818a;margin-top:3px;line-height:1.4}
.box .dim{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:#9aa0a8;margin-top:4px}
.box.wide{flex:none;width:100%}

.aud{border-color:#0f8b7e}.aud .bt{color:#0b6b60}
.vis{border-color:#c2185b}.vis .bt{color:#9c1449}
.txt{border-color:#6d28d9}.txt .bt{color:#5620ac}
.fus{border-color:#d97706;background:#fffaf2}.fus .bt{color:#a85c05}
.cls{border-color:#3f4650}.cls .bt{color:#242930}
.pro{border-color:#0f8b7e;background:#f2fbf9;border-style:dashed}.pro .bt{color:#0b6b60}
.out{border-color:#3f4650;background:#f6f7f8}

.arrow{text-align:center;color:#c3c8ce;font-size:15px;line-height:1;margin:7px 0}
.frozen{display:inline-block;font-size:9.5px;font-weight:700;letter-spacing:.03em;
  padding:1px 6px;border-radius:9px;background:#eef0f2;color:#6b7280;margin-left:5px;vertical-align:1.5px}
.train{background:#e7f5ee;color:#15803d}

.note{margin-top:8px;font-size:11.5px;color:#8a9098;line-height:1.55;
  border-left:2.5px solid #e3e5e8;padding-left:11px}
.note b{color:#4a4f57}
.excl{border:1.6px dashed #c8ccd2;border-radius:9px;padding:9px 13px;background:#fafbfc;flex:1}
.excl .bt{font-size:13px;font-weight:650;color:#9aa0a8}
.excl .bs{font-size:11.5px;color:#a8aeb6;margin-top:3px;line-height:1.4}

.foot{margin-top:24px;padding-top:14px;border-top:1px solid #e8eaec;
  font-size:11px;color:#9aa0a8;font-family:ui-monospace,Menlo,monospace}
.legend{display:flex;gap:16px;margin-top:9px;font-size:11.5px;color:#7b818a}
.legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px}
"""

ARROW = '<div class="arrow">▼</div>'


def stage(n, name, why, body):
    return (f'<div class="stage"><div class="num">{n}</div><div class="stage-body">'
            f'<div class="stage-name">{name}</div>'
            f'<div class="stage-why">{why}</div>{body}</div></div>')


# ---------------------------------------------------------------- 처음 설계
V2 = f"""
<div class="sheet">
  <div class="title">처음 설계 — 이중모달 (multimodal_pipeline_v2)</div>
  <div class="sub">표정과 음성을 양방향 교차 어텐션으로 정합·융합해 발화 단위 감정을 분류한다</div>
  <div class="meta">클래스 <b>C = 7</b> (Ekman 6 + 중립) · 모달리티 <b>2종</b> ·
    텍스트는 <b>의도적으로 제외</b> · 구현 전 설계 단계</div>

  {stage(1, "입력", "표정과 음성이 즉시 가용한 대면 상황을 전제했다",
    '<div class="row">'
    '<div class="box vis"><div class="bt">시각 — 얼굴 프레임 시퀀스</div>'
    '<div class="bs">표정</div></div>'
    '<div class="box aud"><div class="bt">청각 — 멜 스펙트로그램</div>'
    '<div class="bs">음성</div></div>'
    '<div class="excl"><div class="bt">텍스트 — 제외</div>'
    '<div class="bs">① ASR 지연·오류 배제 ② 배포 단순성 ③ 대면 시나리오</div></div>'
    '</div>')}
  {ARROW}
  {stage(2, "전처리 (모달리티별)", "국소 시간 구조를 보존하면서 순서 정보를 주입한다",
    '<div class="box wide cls"><div class="bt">Temporal Conv + Positional Encoding</div>'
    '<div class="bs">Conv로 인접 프레임의 국소 패턴을, 위치 인코딩으로 순서를 넣는다</div></div>')}
  {ARROW}
  {stage(3, "백본", "두 모달리티를 같은 차원으로 맞춰야 서로 참조할 수 있다",
    '<div class="row">'
    '<div class="box vis"><div class="bt">Visual Backbone</div>'
    '<div class="dim">X_v : [T_v × d_model]</div></div>'
    '<div class="box aud"><div class="bt">Audio Backbone</div>'
    '<div class="dim">X_a : [T_a × d_model]</div></div>'
    '</div>')}
  {ARROW}
  {stage(4, "양방향 교차 어텐션 (N회 반복)",
    "감정 표현에는 감각 간 시차가 있다 — 입꼬리가 먼저 변하고 1초 뒤 목소리가 떨릴 수 있다. "
    "후기 융합은 이걸 학습하지 못한다",
    '<div class="row">'
    '<div class="box fus"><div class="bt">방향 1 &nbsp;V ← A</div>'
    '<div class="bs">Q = 시각, K/V = 청각</div><div class="dim">Context_v : [T_v × d]</div></div>'
    '<div class="box fus"><div class="bt">방향 2 &nbsp;A ← V</div>'
    '<div class="bs">Q = 청각, K/V = 시각</div><div class="dim">Context_a : [T_a × d]</div></div>'
    '</div>'
    '<div class="note">각 방향 = Multi-Head Cross-Attention + Residual + LayerNorm + FFN. '
    '<b>양방향인 이유</b>는 시차가 양쪽으로 발생하기 때문이다 — '
    'v1의 단방향(Q=시각) 설계는 "표정→음성 참조"만 잡던 결함이었고 v2에서 고쳤다.</div>')}
  {ARROW}
  {stage(5, "풀링", "가변 길이 시퀀스를 고정 길이 벡터로 축약한다",
    '<div class="row">'
    '<div class="box cls"><div class="bt">z_cross_v</div><div class="bs">Pool over T_v</div></div>'
    '<div class="box cls"><div class="bt">z_cross_a</div><div class="bs">Pool over T_a</div></div>'
    '<div class="box vis"><div class="bt">z_v</div><div class="bs">Mean over X_v</div></div>'
    '<div class="box aud"><div class="bt">z_a</div><div class="bs">Mean over X_a</div></div>'
    '</div>')}
  {ARROW}
  {stage(6, "하이브리드 분류기",
    "교차 어텐션만 쓰면 각 모달리티 고유의 거시 맥락이 희석된다 — 미세 타이밍과 전반적 분위기를 함께 본다",
    '<div class="box wide cls">'
    '<div class="bt">concat[ z_cross_v ; z_cross_a ; z_v ; z_a ] → MLP → Softmax</div>'
    '<div class="dim">[4 · d_model] → C = 7</div></div>')}
  {ARROW}
  <div class="box wide out"><div class="bt">7클래스 확률 분포</div>
    <div class="bs">분노 · 혐오 · 공포 · 행복 · 슬픔 · 놀람 · 중립</div></div>

  <div class="note" style="margin-top:18px">
    <b>강건성 설계</b> — 학습 시 모달리티 드롭아웃: 한 모달리티를 통째로 0으로 만들어,
    카메라가 가려지거나 마이크가 죽어도 나머지로 판단하게 한다. <b>이 설계는 v11까지 그대로 남아 있다.</b>
  </div>

  <div class="foot">multimodal_pipeline_v2 · 통합기록 2장 · scripts/build_architecture_diagrams.py로 생성</div>
</div>
"""

# ---------------------------------------------------------------- v11
V11 = f"""
<div class="sheet">
  <div class="title">현재 설계 — v11 트리모달</div>
  <div class="sub">사전학습 백본 3종 위에 2단계 계층적 교차 어텐션을 얹고, 운율로 오디오 비중을 조절한다</div>
  <div class="meta">클래스 <b>C = 7</b> · 전체 <b>436,875,655</b> 파라미터(fp32 1,667MB) ·
    학습 <b>8,760,071</b>개(2.0%) · test <b>46.19%</b>(화자 독립)</div>

  {stage(1, "입력 — 같은 발화 구간에서 동시 수집",
    "처음 설계에서 텍스트를 뺐던 이유(ASR 지연·오류)가 사라졌다 — AI Hub 데이터에 전사문이 이미 있다",
    '<div class="row">'
    '<div class="box aud"><div class="bt">음성 파형</div><div class="bs">16kHz mono</div></div>'
    '<div class="box vis"><div class="bt">얼굴 프레임</div>'
    '<div class="bs">mediapipe 재검출+정렬</div><div class="dim">112 × 112</div></div>'
    '<div class="box txt"><div class="bt">전사문</div><div class="bs">STT / 대본</div></div>'
    '<div class="box pro"><div class="bt">운율 10차원</div>'
    '<div class="bs">f0 · jitter · shimmer · HNR · energy</div></div>'
    '</div>')}
  {ARROW}
  {stage(2, "사전학습 백본 — 전체 파라미터의 98.0%가 여기에 있고 전부 동결",
    "8만 발화 규모에서 대형 사전학습 모델을 파인튜닝하면 과적합이 심해진다는 것을 v1~v7에서 여섯 번 확인했다",
    '<div class="row">'
    '<div class="box aud"><div class="bt">wav2vec2-large-XLSR-53<span class="frozen">동결</span></div>'
    '<div class="bs">12번째 층 사용 · <b>ASR 파인튜닝본이 아닌 SSL 모델</b> — '
    'ASR은 감정·화자 정보를 지우도록 학습된다</div>'
    '<div class="dim">315.4M · proj 1024→256</div></div>'
    '<div class="box vis"><div class="bt">MobileFaceNet<span class="frozen">동결</span></div>'
    '<div class="bs">얼굴 인식 전용 사전학습 (emotiefflib)</div>'
    '<div class="dim">2.1M · proj 512→256</div></div>'
    '<div class="box txt"><div class="bt">KLUE-BERT base<span class="frozen">동결</span></div>'
    '<div class="bs">12층 전부 동결 — 파인튜닝이 과적합의 주원인이었다</div>'
    '<div class="dim">110.6M · proj 768→256</div></div>'
    '</div>')}
  {ARROW}
  {stage(3, "공통 프론트엔드", "세 모달리티가 모두 256차원으로 통일되는 지점 — 여기서부터 비교·결합이 가능해진다",
    '<div class="row">'
    '<div class="box aud"><div class="bt">TemporalConv + 트랜스포머 2층<span class="frozen train">학습</span></div>'
    '<div class="dim">X_a</div></div>'
    '<div class="box vis"><div class="bt">TemporalConv + 트랜스포머 2층<span class="frozen train">학습</span></div>'
    '<div class="dim">X_v</div></div>'
    '<div class="box txt"><div class="bt">Linear 768→256<span class="frozen train">학습</span></div>'
    '<div class="dim">X_t</div></div>'
    '</div>')}
  {ARROW}
  {stage(4, "2단계 계층적 교차 어텐션 (4블록)",
    "모달리티 3개면 3쌍 × 양방향 = 6블록이 필요해 실시간 추론에 부담이다. 그래서 2단계로 나눴다",
    '<div class="row">'
    '<div class="box fus"><div class="bt">1단계 &nbsp; 오디오 ↔ 텍스트</div>'
    '<div class="bs">"무슨 말을 어떤 억양으로 했나" · 양방향 2블록</div></div>'
    '<div class="box fus"><div class="bt">2단계 &nbsp; 영상 ↔ (오디오+텍스트)</div>'
    '<div class="bs">"표정이 그걸 뒷받침하나" · 양방향 2블록</div></div>'
    '</div>'
    '<div class="note"><b>주의</b> — 이 순서는 텍스트를 앵커로 삼는다. '
    '애블레이션에서 텍스트 제거가 가장 아팠는데(−6.59%p) 짧은 발화 정확도는 오히려 높았다. '
    '텍스트의 <b>정보</b>가 아니라 <b>구조</b>가 기여하고 있을 가능성이 있다(8.30.2절, 검증 대기).</div>')}
  {ARROW}
  {stage(5, "하이브리드 결합 + 운율 게이트",
    "교차 어텐션 결과(미세 타이밍)와 백본 원본 평균(거시 분위기)을 함께 쓴다 — 처음 설계의 ⑥을 그대로 계승",
    '<div class="row">'
    '<div class="box aud"><div class="bt">concat[ z_cross_a ; mean(X_a) ]</div><div class="dim">512</div></div>'
    '<div class="box vis"><div class="bt">concat[ z_cross_v ; mean(X_v) ]</div><div class="dim">512</div></div>'
    '<div class="box txt"><div class="bt">concat[ z_cross_t ; mean(X_t) ]</div><div class="dim">512</div></div>'
    '</div>'
    '<div class="arrow">▼</div>'
    '<div class="box wide pro"><div class="bt">운율 게이트 (오디오에만) &nbsp; g·z + (1−g)·Linear(운율)</div>'
    '<div class="bs">억양이 잡음 많으면 안정적인 통계 쪽으로 비중을 자동 조절 — '
    '<b>다만 소음 조건에서 기여가 0으로 측정됐다</b>(8.30.3절)</div></div>')}
  {ARROW}
  {stage(6, "분류기", "세 모달리티 512차원씩을 이어 붙여 한 번에 판단한다",
    '<div class="box wide cls">'
    '<div class="bt">concat[ 영상 512 ; 오디오 512 ; 텍스트 512 ] = 1536 '
    '→ Linear(1536→512) → GELU → Dropout → Linear(512→7)</div></div>')}
  {ARROW}
  <div class="box wide out"><div class="bt">7클래스 로짓 → softmax</div>
    <div class="bs">분노 · 혐오 · 공포 · 행복 · 슬픔 · 놀람 · 중립 &nbsp;— &nbsp;
      확신도 0.5 미만이면 판단 보류 시 정확도 60.66%(8.30.5절)</div></div>

  <div class="note" style="margin-top:18px">
    <b>처음 설계에서 이어진 것</b> — Temporal Conv + 위치 인코딩, 양방향 교차 어텐션,
    하이브리드 결합, 모달리티 드롭아웃(학습 시 25%).
    <b>바뀐 것</b> — 텍스트 추가, 백본 3종 모두 사전학습으로 교체 후 동결,
    융합이 양방향 1쌍에서 2단계 계층으로, 운율 게이트 신설.
  </div>

  <div class="legend">
    <span><i style="background:#0f8b7e"></i>오디오</span>
    <span><i style="background:#c2185b"></i>영상</span>
    <span><i style="background:#6d28d9"></i>텍스트</span>
    <span><i style="background:#d97706"></i>융합</span>
  </div>

  <div class="foot">configs/config_si_w2v.yaml · 통합기록 8.18절 · v11_모델카드.md 3장 ·
    scripts/build_architecture_diagrams.py로 생성</div>
</div>
"""

DIAGRAMS = {"v2": ("architecture_v2_bimodal.png", V2),
            "v11": ("architecture_v11.png", V11)}


def render(html: str, out: Path, width: int = 1180) -> None:
    if not Path(CHROME).exists():
        sys.exit(f"Chrome을 찾을 수 없다: {CHROME}")
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "d.html"
        src.write_text(f"<meta charset=utf-8><style>{CSS}</style>{html}", encoding="utf-8")
        # --window-size의 높이는 넉넉히 주고 --screenshot이 전체를 잡게 한다.
        subprocess.run(
            [CHROME, "--headless", "--disable-gpu", "--hide-scrollbars",
             "--force-device-scale-factor=2",          # 2배 해상도로 또렷하게
             f"--screenshot={out}", f"--window-size={width},2400",
             "--virtual-time-budget=3000", src.as_uri()],
            check=True, capture_output=True, timeout=180,
        )
    if not out.exists() or out.stat().st_size < 20_000:
        sys.exit(f"{out.name}: 렌더 실패로 보인다 ({out.stat().st_size if out.exists() else 0} bytes)")
    crop_bottom(out)


def crop_bottom(path: Path, margin: int = 56) -> None:
    """아래쪽 흰 여백을 잘라낸다.

    Chrome의 --screenshot은 창 크기 그대로 찍으므로, 내용보다 창을 크게 잡으면 아래가
    전부 흰색으로 남는다. 창 높이를 문서마다 손으로 맞추는 대신 여기서 잘라낸다 —
    내용이 늘어나도 스크립트를 안 고쳐도 된다.
    """
    from PIL import Image

    im = Image.open(path).convert("RGB")
    w, h = im.size
    px = im.load()
    last = 0
    for y in range(h - 1, -1, -1):
        # 가로로 훑다가 흰색이 아닌 픽셀이 하나라도 있으면 거기가 내용의 끝이다.
        if any(px[x, y] != (255, 255, 255) for x in range(0, w, 4)):
            last = y
            break
    if last == 0:
        sys.exit(f"{path.name}: 내용이 없다 — 렌더가 비어 있다")
    new_h = min(h, last + margin)
    if new_h < h:
        im.crop((0, 0, w, new_h)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(DIAGRAMS), default=None)
    args = ap.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    targets = {args.only: DIAGRAMS[args.only]} if args.only else DIAGRAMS
    for key, (name, html) in targets.items():
        out = ASSETS / name
        render(html, out)
        print(f"[diagram] {out.relative_to(ROOT)}  {out.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
