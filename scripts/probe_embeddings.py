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


def _drop_rare_classes(X, y, extra):
    """표본이 1개뿐인 클래스를 뺀다 — stratify가 그런 클래스에서 ValueError로 죽는다.

    그런 클래스는 어차피 학습/평가 어느 쪽에도 제대로 들어가지 못하므로 빼는 게 맞다.
    extra(예: person_ids)도 같은 마스크로 잘라 인덱스 정합을 유지한다.
    """
    uniq, cnt = np.unique(y, return_counts=True)
    usable = set(uniq[cnt >= 2])
    keep = np.array([v in usable for v in y])
    return X[keep], y[keep], (None if extra is None else extra[keep]), int((~keep).sum())


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
    X, y, _, dropped = _drop_rare_classes(X, y, None)

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


def probe_within_speaker(
    X: np.ndarray, y: np.ndarray, person_ids: np.ndarray,
    speakers: list, seed: int, max_iter: int,
) -> tuple[float, float, int]:
    """화자 한 명 안에서만 감정을 맞히는 프로브를 화자마다 돌려 평균낸다.

    돌려주는 값: (정규화 상승폭 평균, 평균 우연 수준, 화자당 평균 표본 수)

    화자를 고정하면 목소리·얼굴·배경이 상수가 되므로, 남는 변이는 감정(과 발화 내용)뿐이다.
    이 조건에서 감정이 훨씬 잘 읽히면 "화자 변이가 감정 신호를 덮고 있었다"는 뜻이고,
    별 차이가 없으면 화자 정보는 그냥 같이 들어있을 뿐 방해는 아니라는 뜻이다.

    화자마다 감정 분포가 달라 우연 수준도 제각각이므로, 절대 정확도가 아니라
    정규화 상승폭 (acc-chance)/(1-chance) 로 모아야 화자 간 평균이 의미를 갖는다.
    """
    lifts, chances, sizes = [], [], []
    for spk in speakers:
        m = person_ids == spk
        Xs, ys = X[m], y[m]
        if len(np.unique(ys)) < 2:
            continue  # 감정이 한 종류뿐인 화자는 분류 문제가 성립하지 않음
        try:
            acc, chance, _ = probe(Xs, ys, seed, max_iter)
        except ValueError:
            continue  # 층화 분할이 불가능할 만큼 표본이 적은 화자는 건너뜀
        lifts.append((acc - chance) / (1 - chance))
        chances.append(chance)
        sizes.append(len(ys))
    if not lifts:
        return float("nan"), float("nan"), 0
    return float(np.mean(lifts)), float(np.mean(chances)), int(np.mean(sizes))


def probe_speaker_normalized(
    X: np.ndarray, y: np.ndarray, person_ids: np.ndarray,
    seed: int, max_iter: int, min_for_mean: int = 5,
) -> tuple[float, float]:
    """화자별 평균 벡터를 뺀 뒤 프로빙한다. 반환: (정확도, 우연 수준).

    화자 고정 프로빙(probe_within_speaker)에서 "화자 변이가 감정을 덮는다"까지는
    확인됐지만, 그 변이가 **단순한 평행이동(화자마다 벡터가 통째로 다른 위치)**인지는
    별개 문제다. 평행이동이라면 화자 평균을 빼는 것만으로 지워진다. 그게 아니면
    (예: 화자마다 감정 축의 방향 자체가 다르면) 평균을 빼도 별로 안 좋아진다.
    이 차이가 v12 설계를 가른다 — 전자면 정규화/적대적 학습이 통하고, 후자면 안 통한다.

    **평균은 반드시 프로브 학습셋에서만 낸다.** 평가셋 표본까지 넣어 평균을 내면
    평가셋 정보가 전처리 단계로 새어 결과가 낙관적으로 부풀려진다.

    실사용 관점: 로봇은 같은 사람을 반복해 만나므로 그 사람의 평균을 쌓아둘 수 있다
    (라벨이 필요 없으니 정당하다). 다만 **처음 본 사람에게 즉시 판단할 때는 못 쓴다** —
    아직 그 사람의 평균이 없기 때문이다. 이 수치는 "평균을 확보한 뒤"의 성능이다.
    """
    X, y, pid, _ = _drop_rare_classes(X, y, person_ids)

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.3, random_state=seed, stratify=y)

    # 학습셋 표본이 너무 적은 화자는 평균이 불안정하므로 전체 평균으로 대체한다
    # (그 화자에 대해서는 화자 정규화를 사실상 안 하는 셈 — 과보정보다 안전하다).
    global_mean = X[tr].mean(axis=0)
    means = {}
    for spk in np.unique(pid):
        sel = tr[pid[tr] == spk]
        means[spk] = X[sel].mean(axis=0) if len(sel) >= min_for_mean else global_mean

    Xn = X - np.stack([means[p] for p in pid])

    scaler = StandardScaler().fit(Xn[tr])
    clf = LogisticRegression(max_iter=max_iter)
    clf.fit(scaler.transform(Xn[tr]), y[tr])
    acc = clf.score(scaler.transform(Xn[te]), y[te])

    _, counts = np.unique(y[te], return_counts=True)
    return acc, counts.max() / len(y[te])


def probe_size_matched(
    X: np.ndarray, y: np.ndarray, n_samples: int,
    seed: int, max_iter: int, n_repeats: int = 5,
) -> float:
    """화자를 섞은 채로 **화자 내 프로브와 같은 표본 수만** 써서 돌리는 대조군.

    이게 없으면 결론을 못 낸다. 화자 내 프로브는 데이터가 1/화자수로 줄어드는데,
    선형 프로브라도 512차원을 200여 개 표본으로 맞추면 성능이 떨어진다. 대조군 없이
    "화자 내 정확도가 낮다"를 보면 **화자 변이가 문제가 아니어서인지, 그냥 데이터가
    적어서인지** 구분할 수 없다. 같은 크기로 맞춰야 차이가 오롯이 '화자 고정' 효과가 된다.

    표본을 무작위로 뽑는 만큼 뽑기 운을 타므로 여러 번 반복해 평균낸다.
    """
    rng = np.random.default_rng(seed)
    lifts = []
    for r in range(n_repeats):
        idx = rng.choice(len(y), size=min(n_samples, len(y)), replace=False)
        try:
            acc, chance, _ = probe(X[idx], y[idx], seed + r, max_iter)
        except ValueError:
            continue
        lifts.append((acc - chance) / (1 - chance))
    return float(np.mean(lifts)) if lifts else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", required=True, help="extract_embeddings.py가 만든 .npz")
    parser.add_argument("--min-utts", type=int, default=20,
                        help="화자 프로빙에 포함할 최소 발화 수 — 발화가 몇 개뿐인 화자는 "
                             "층화 분할이 불가능하고 결과만 흔든다")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--speaker-normalize", action="store_true",
                        help="화자별 평균 벡터를 뺀 뒤 감정·화자를 다시 프로빙한다 — "
                             "화자 변이가 단순 평행이동인지 확인하고, 그걸 지웠을 때의 "
                             "감정 성능 상한을 재학습 없이 추정한다")
    parser.add_argument("--within-speaker", action="store_true",
                        help="화자를 고정한 채 감정을 프로빙하고, 같은 표본 수의 화자혼합 "
                             "대조군과 비교한다 — 화자 정보가 '같이 들어있는' 것인지 "
                             "'감정 판별을 방해하는' 것인지 가른다")
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

    if args.speaker_normalize:
        print()
        print("=" * 76)
        print("화자 평균 제거 후 재프로빙 — 화자 변이가 '단순 평행이동'인가")
        print("=" * 76)
        print("화자마다 벡터가 통째로 밀려 있는 것뿐이라면 평균을 빼면 지워진다.")
        print("평균은 프로브 학습셋에서만 낸다(평가셋 정보 누출 방지).")
        print()
        print(f"{'표현':22} {'감정 원본':>10} {'감정 정규화':>12} {'변화':>9} "
              f"{'화자 원본':>10} {'화자 정규화':>12}")
        print("─" * 76)
        for name in VECTOR_NAMES:
            X = d[name]
            e0 = results[(name, "감정")][0]
            s0 = results[(name, "화자")][0]
            e1, _ = probe_speaker_normalized(X, labels, person_ids, args.seed, args.max_iter)
            s1, _ = probe_speaker_normalized(
                X[spk_mask], person_ids[spk_mask], person_ids[spk_mask],
                args.seed, args.max_iter)
            print(f"{VECTOR_DESC[name]+' ('+name+')':22} {100*e0:>9.2f}% {100*e1:>11.2f}% "
                  f"{100*(e1-e0):>+8.2f}%p {100*s0:>9.2f}% {100*s1:>11.2f}%")
        print("─" * 76)
        print("해석: **'감정 변화' 열만 증거다.**")
        print("  뚜렷이 상승 -> 화자 변이가 평행이동이었다. 정규화/적대적 학습이 통한다")
        print("  0 근처      -> 평행이동이 아니다(화자마다 감정 축 방향 자체가 다름 등).")
        print("                 평균을 빼는 방식으로는 못 고치니 다른 접근이 필요하다")
        print()
        print("  화자 열은 증거가 아니라 뺄셈이 실행됐는지 보는 확인용이다 — 선형 프로브는")
        print("  주로 평균을 읽으므로, 평균을 뺀 이상 변이의 성격과 무관하게 우연 수준으로")
        print("  떨어진다(인공 데이터 검증에서 회전 세계도 똑같이 떨어짐을 확인).")
        print()
        print("주의: 이 수치는 '그 사람의 평균을 이미 확보한 뒤'의 성능이다.")
        print("      로봇이 같은 사람을 반복해 만나는 상황에는 그대로 쓸 수 있지만,")
        print("      처음 본 사람에게 즉시 판단할 때는 쓸 수 없다.")

    if args.within_speaker:
        print()
        print("=" * 72)
        print("화자 고정 감정 프로빙 — 화자 변이가 감정 신호를 덮고 있는가")
        print("=" * 72)
        spk_list = sorted(keep)
        print(f"대상 화자 {len(spk_list)}명. 정규화 상승폭 (정확도-우연)/(1-우연) 으로 비교한다.")
        print()
        # 화자당 표본 수는 X와 무관(person_ids만으로 정해짐)하므로 루프 밖에서 한 번만 구한다.
        # 루프 안에서 덮어쓴 값을 루프 뒤에서 출력하면 마지막 반복 값에 의존하게 되어 위험하다.
        avg_n = int(np.mean([(person_ids == s).sum() for s in spk_list]))

        print(f"{'표현':22} {'전체':>9} {'화자내':>9} {'크기맞춘대조':>13} {'화자내-대조':>12}")
        print("─" * 72)
        for name in VECTOR_NAMES:
            X = d[name]
            e_acc, e_ch = results[(name, "감정")]
            full = (e_acc - e_ch) / (1 - e_ch)
            wl, _, _ = probe_within_speaker(
                X[spk_mask], labels[spk_mask], person_ids[spk_mask],
                spk_list, args.seed, args.max_iter)
            ctrl = probe_size_matched(X, labels, avg_n, args.seed, args.max_iter)
            print(f"{VECTOR_DESC[name]+' ('+name+')':22} {100*full:>8.1f}% "
                  f"{100*wl:>8.1f}% {100*ctrl:>12.1f}% {100*(wl-ctrl):>+11.1f}%p")
        print("─" * 72)
        print(f"대조군은 화자를 섞은 채 화자당 평균 표본 수({avg_n}개)만 써서 5회 반복 평균.")
        print()
        print("해석: 마지막 열('화자내 - 대조')이 결론이다. 표본 수를 맞췄으므로")
        print("      이 차이는 오롯이 '화자를 고정한 효과'다.")
        print("  크게 양수  -> 화자 변이가 감정 신호를 덮고 있었다. 제거하면 이득이 크다")
        print("  0 근처     -> 화자 정보는 공존할 뿐 방해가 아니다. 제거해도 감정 성능은 안 오른다")
        print("  음수       -> 화자 간 대비 자체가 감정 판별에 쓰이고 있었다(제거하면 손해)")

    print()
    print("참고: 감정 클래스별 분포")
    for i, nm in enumerate(label_names):
        c = int((labels == i).sum())
        print(f"  {nm:9} {c:>6,}개 ({100*c/n:5.2f}%)")


if __name__ == "__main__":
    main()
