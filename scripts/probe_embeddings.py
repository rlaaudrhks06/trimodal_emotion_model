"""프로빙(probing): 모델이 만든 벡터에 어떤 정보가 들어있는지 선형 분류기로 재본다.

**무엇을 알아내려는 것인가**

v11은 train 60.6% / test 46.2%로 14.4%p 벌어졌다. 흔한 설명은 "화자를 외웠다"인데,
지금까지 이건 추측이었다. 프로빙은 이걸 숫자로 바꾼다 — 벡터에서 **감정**을 얼마나
읽어낼 수 있는지와 **화자 신원**을 얼마나 읽어낼 수 있는지를 같은 방식으로 재서
비교한다. 화자 쪽이 압도적으로 잘 맞으면, 그 표현은 감정보다 "누가 말했는가"를
주로 담고 있다는 뜻이다.

방법은 표준적이다: 표현을 고정한 채 그 위에 **선형 분류기 하나만** 학습시킨다.
비선형 분류기를 쓰면 "표현에 정보가 있어서"가 아니라 "프로브가 똑똑해서" 맞히는
것과 구분이 안 되므로 선형으로 제한한다.

**모달리티별로 따로 재는 게 핵심이다.** 예를 들어 시각 브랜치(z_v)가 화자는 80%로
맞히고 감정은 30%밖에 못 맞힌다면, 그 브랜치는 감정 인식이 아니라 사실상 얼굴
인식을 하고 있다는 직접 증거가 된다.

**주의 — 화자 프로빙 결과를 과대해석하지 말 것.** 목소리와 얼굴에서 화자 특징이
읽히는 건 당연하다(사람도 그렇다). 문제는 "읽히느냐"가 아니라 **"감정보다 훨씬 잘
읽히느냐"** 이므로, 항상 두 값을 나란히 놓고 우연 수준(chance) 대비로 비교한다.

실행 예:
    python scripts/probe_embeddings.py --emb results/embeddings/v11_test.npz
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

VECTOR_NAMES = ["z_v", "z_a", "z_t", "h"]
VECTOR_DESC = {
    "z_v": "시각 브랜치",
    "z_a": "오디오 브랜치",
    "z_t": "텍스트 브랜치",
    "h": "최종 결합(분류 직전)",
}


def probe(X: np.ndarray, y: np.ndarray, seed: int, max_iter: int) -> tuple[float, float, int]:
    """X로 y를 맞히는 선형 프로브를 학습하고 (정확도, 우연 수준, 버린 표본 수)를 돌려준다.

    우연 수준 = "무조건 최다 클래스만 찍기" 정확도. 클래스 수가 다른 과제끼리
    (감정 7개 vs 화자 수십 명) 비교하려면 절대 정확도가 아니라 이 기준선 대비
    상승폭을 봐야 한다.

    stratify로 층화 분할하는 이유: 화자별 발화 수가 고르지 않아서 무작위로 나누면
    프로브 학습셋에 아예 안 나온 화자가 평가셋에 생길 수 있다(그러면 그 화자는
    원리적으로 못 맞힌다 — 표현의 문제가 아니라 분할의 문제가 된다).

    다만 stratify는 표본이 1개뿐인 클래스가 있으면 ValueError로 죽는다. 그런
    클래스는 어차피 학습/평가 어느 쪽에도 제대로 못 들어가므로 미리 빼고,
    몇 개를 뺐는지 호출부에 알려 결과 해석 시 참고하게 한다.
    """
    uniq, cnt = np.unique(y, return_counts=True)
    usable = set(uniq[cnt >= 2])
    keep = np.array([v in usable for v in y])
    dropped = int((~keep).sum())
    X, y = X[keep], y[keep]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    # 표준화: 차원마다 스케일이 다르면 L2 규제가 특정 차원만 과하게 누른다.
    # 통계는 프로브 학습셋에서만 fit한다(평가셋 정보가 새지 않게).
    scaler = StandardScaler().fit(X_tr)
    # n_jobs는 넣지 않는다 — sklearn 1.8부터 무효고 1.10에서 제거 예정이라
    # FutureWarning만 뜬다. 기본 solver(lbfgs)의 다항 로지스틱은 어차피 이 인자를 안 쓴다.
    clf = LogisticRegression(max_iter=max_iter)
    clf.fit(scaler.transform(X_tr), y_tr)
    acc = clf.score(scaler.transform(X_te), y_te)

    _, counts = np.unique(y_te, return_counts=True)
    return acc, counts.max() / len(y_te), dropped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", required=True, help="extract_embeddings.py가 만든 .npz")
    parser.add_argument("--min-utts", type=int, default=20,
                        help="화자 프로빙에 포함할 최소 발화 수 — 발화가 몇 개뿐인 화자는 "
                             "층화 분할이 불가능하고 결과만 흔든다")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000)
    args = parser.parse_args()

    d = np.load(args.emb, allow_pickle=False)
    labels, person_ids = d["labels"], d["person_ids"]
    label_names = [str(x) for x in d["label_names"]]
    n = len(labels)

    print(f"[probe] {Path(args.emb).name} — 발화 {n:,}개 / 화자 {len(set(person_ids)):,}명\n")

    # 화자 과제는 발화가 충분한 화자로 제한한다(위 --min-utts 설명 참고)
    uniq, cnt = np.unique(person_ids, return_counts=True)
    keep = set(uniq[cnt >= args.min_utts])
    spk_mask = np.array([p in keep for p in person_ids])
    print(f"[probe] 화자 과제 대상: 발화 {args.min_utts}개 이상인 화자 {len(keep)}명, "
          f"발화 {spk_mask.sum():,}개 ({100*spk_mask.sum()/n:.1f}%)\n")

    print(f"{'표현':22} {'과제':6} {'정확도':>8} {'우연':>8} {'상승폭':>9}")
    print("─" * 60)
    results = {}
    for name in VECTOR_NAMES:
        X = d[name]
        for task, y, mask in [("감정", labels, None),
                              ("화자", person_ids, spk_mask)]:
            Xt, yt = (X, y) if mask is None else (X[mask], y[mask])
            acc, chance, dropped = probe(Xt, yt, args.seed, args.max_iter)
            results[(name, task)] = (acc, chance)
            note = f"  (표본 1개뿐인 클래스 {dropped}개 제외)" if dropped else ""
            print(f"{VECTOR_DESC[name]+' ('+name+')':22} {task:6} "
                  f"{100*acc:>7.2f}% {100*chance:>7.2f}% {100*(acc-chance):>+8.2f}%p{note}")
        print()

    print("─" * 60)
    print("해석: 같은 표현에서 '화자 상승폭'이 '감정 상승폭'보다 크게 높으면,")
    print("      그 브랜치는 감정보다 화자 개인 특성을 주로 담고 있다는 뜻이다.")
    print()
    for name in VECTOR_NAMES:
        le = results[(name, "감정")][0] - results[(name, "감정")][1]
        ls = results[(name, "화자")][0] - results[(name, "화자")][1]
        # 상승폭이 0 이하면 프로브가 우연 수준도 못 넘은 것이다. 이때 비율을 내면
        # inf나 음수가 나와서 "화자 쪽이 무한히 강하다"처럼 잘못 읽히므로 따로 처리한다.
        if le <= 0 and ls <= 0:
            ratio_s, verdict = "  —  ", "둘 다 우연 수준 — 표현에 정보 없음"
        elif le <= 0:
            ratio_s, verdict = "  —  ", "감정은 우연 수준, 화자만 읽힘"
        elif ls <= 0:
            ratio_s, verdict = "  —  ", "화자는 우연 수준, 감정만 읽힘"
        else:
            r = ls / le
            ratio_s = f"{r:5.2f}"
            verdict = ("화자 쪽이 훨씬 강함" if r > 2 else
                       "화자 쪽이 다소 강함" if r > 1.2 else
                       "감정 쪽이 강함" if r < 0.8 else "비슷함")
        print(f"  {VECTOR_DESC[name]:16} 감정 {100*le:+6.2f}%p / 화자 {100*ls:+6.2f}%p "
              f"(비 {ratio_s}) -> {verdict}")

    print()
    print("참고: 감정 클래스별 분포")
    for i, nm in enumerate(label_names):
        c = int((labels == i).sum())
        print(f"  {nm:9} {c:>6,}개 ({100*c/n:5.2f}%)")


if __name__ == "__main__":
    main()
