"""중복·화자누수 검사 로직 검증 (13.16절).

  ① 텍스트 정규화: 공백·문장부호·전각 차이를 흡수하고, 다른 문장은 안 묶는다
  ② speaker_of: utt_id에서 person_id를 뽑는다(형식이 다르면 utt_id 전체)
  ③ 텍스트 중복 검출: 겹치는 것만 제외 목록에 오른다
  ④ 화자 누수 검출: 임계값 이상이면 그 화자의 **모든 발화**가 제외된다
  ⑤ 빈 텍스트는 서로 묶지 않는다 (전사 실패가 전부 중복으로 잡히면 데이터가 통째로 날아간다)

얼굴 임베딩은 합성 벡터로 주입한다(GPU·이미지 불필요).
    python tests/test_overlap_check.py
"""
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import scripts.check_dataset_overlap as ov


def main() -> int:
    print("=== 중복·화자누수 검사 ===")
    # ①
    assert ov.norm_text("안녕, 하세요!") == ov.norm_text("안녕하세요")
    assert ov.norm_text("Hello  World.") == ov.norm_text("helloworld")
    assert ov.norm_text("밥 먹었어?") != ov.norm_text("밥 먹을래?")
    assert ov.norm_text("   ") == ""
    print("  ✅ 텍스트 정규화")
    # ②
    assert ov.speaker_of("1000_49_1127_1217") == "49"
    assert ov.speaker_of("KETI_MULTIMODAL_0001") == "KETI_MULTIMODAL_0001"
    print("  ✅ speaker_of")

    d = Path(tempfile.mkdtemp())
    ours = pd.DataFrame({"utt_id": ["1_7_0_10", "1_7_10_20", "2_8_0_10"],
                         "text": ["밥 먹었어?", "응 먹었지", "오늘 날씨 좋다"],
                         "face_frames_dir": ["a", "b", "c"]})
    new = pd.DataFrame({"utt_id": ["9_70_0_5", "9_70_5_9", "9_71_0_5", "9_72_0_5"],
                        "text": ["밥 먹었어?!", "처음 보는 문장", "", ""],
                        "face_frames_dir": ["x", "y", "z", "w"]})
    ours.to_csv(d / "ours.csv", index=False); new.to_csv(d / "new.csv", index=False)

    # ③ ⑤ 텍스트만 (얼굴 생략)
    sys.argv = ["x", "--ours", str(d / "ours.csv"), "--new", str(d / "new.csv"),
                "--out", str(d / "o1.json"), "--skip-face"]
    ov.main()
    r = json.loads((d / "o1.json").read_text())
    assert r["exclude_utts"] == ["9_70_0_5"], r["exclude_utts"]
    assert r["n_keep"] == 3
    print("  ✅ 텍스트 중복만 제외 · 빈 텍스트는 안 묶임")

    # ④ 화자 누수: 새 화자 71이 우리 화자 7과 같은 사람(코사인 0.9)
    v7, v8 = np.zeros(512), np.zeros(512); v7[0] = 1.0; v8[1] = 1.0
    close = v7 * 0.9 + v8 * np.sqrt(1 - 0.81)
    emb = {"7": v7, "8": v8, "70": v8 * 0.3 + np.roll(v7, 5) * 0.95, "71": close, "72": np.roll(v7, 9)}
    for k in emb: emb[k] = emb[k] / np.linalg.norm(emb[k])
    orig = ov.face_embeddings
    ov.face_embeddings = lambda df, root, **kw: {s: emb[s] for s in
                                                 {ov.speaker_of(u) for u in df["utt_id"].astype(str)}}
    try:
        sys.argv = ["x", "--ours", str(d / "ours.csv"), "--new", str(d / "new.csv"),
                    "--out", str(d / "o2.json"), "--face-threshold", "0.62"]
        ov.main()
    finally:
        ov.face_embeddings = orig
    r = json.loads((d / "o2.json").read_text())
    assert "9_71_0_5" in r["exclude_utts"], r["exclude_utts"]          # 누수 화자
    assert "9_70_0_5" in r["exclude_utts"]                              # 텍스트 중복
    assert "9_72_0_5" not in r["exclude_utts"], r["exclude_utts"]       # 무관한 화자는 남는다
    assert r["leak_pairs"][0]["new_speaker"] == "71" and r["leak_pairs"][0]["cosine"] >= 0.9
    print(f"  ✅ 화자 누수: 71(코사인 {r['leak_pairs'][0]['cosine']}) 제외 · 무관한 화자 보존")
    print("전부 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
