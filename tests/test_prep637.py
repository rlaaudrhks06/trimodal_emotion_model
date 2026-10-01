"""637 전처리 파싱 검증 (13.17절).

  ① script.txt 4줄 형식 파싱 — ID·감정 대분류·소분류·전사문
  ② 끊어읽기 태그(||HL, |||M, ||||LHL)가 전사문에서 지워진다
  ③ 우리 7클래스로 매핑되는 감정만 통과하고, 나머지(KIND 등)는 EMO_MAP에 없다
  ④ 매핑 결과가 labels.normalize_label을 통과하는 유효한 라벨이다

    python tests/test_prep637.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.prepare_aihub637 import parse_script, EMO_MAP
from src.datasets.labels import EMOTION_LABELS, normalize_label

SAMPLE = """F0001_000001 NEUTRAL #지문
전화가 끊어지자||HL 한숨을 내쉬는||M 대규||||HL
저놔가 끄너지자||HL 한수믈 네쉬는||M 데규||||HL

F0001_100001 KIND #차분하게
할머니~|||HL 왜요?|||LH 잠이 안 오세요?||||LH
할머니~|||HL 왜요?|||LH 자미 안 오세요?||||LH

M0003_104413 SURPRISE #놀란듯
할머니!|||HL 또 아퍼요?||||M
할머니!|||HL 또 아퍼요?||||M
"""


def main() -> int:
    print("=== 637 전처리 파싱 ===")
    d = parse_script(SAMPLE)
    assert set(d) == {"F0001_000001", "F0001_100001", "M0003_104413"}, set(d)
    assert d["F0001_000001"][:2] == ("NEUTRAL", "#지문")
    assert d["M0003_104413"][:2] == ("SURPRISE", "#놀란듯")
    print("  ✅ 4줄 형식 파싱 · 남녀 화자 ID 모두")
    for k, v in d.items():
        assert "|" not in v[2], (k, v[2])
    assert d["F0001_100001"][2] == "할머니~ 왜요? 잠이 안 오세요?", d["F0001_100001"][2]
    print("  ✅ 끊어읽기 태그 제거")
    assert "KIND" not in EMO_MAP and "ANXIOUS" not in EMO_MAP, "대응 없는 감정이 매핑돼 있다"
    assert set(EMO_MAP) == {"NEUTRAL", "SAD", "ANGRY", "JOY", "SURPRISE", "FEAR", "UNPLEASURE"}
    print(f"  ✅ 매핑 대상은 7종뿐: {sorted(EMO_MAP)}")
    for v in EMO_MAP.values():
        assert normalize_label(v) in EMOTION_LABELS, v
    print("  ✅ 매핑 결과가 전부 유효한 7클래스 라벨")
    print("전부 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
