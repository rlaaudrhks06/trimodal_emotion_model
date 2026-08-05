"""매니페스트에 최소 품질 기준(텍스트 길이·얼굴 프레임 수·오디오 길이) 없이
들어간 저품질/노이즈 샘플이 실제로 얼마나 있는지 확인하는 진단 스크립트.

학습 파이프라인(manifest_dataset.py)은 지금 이런 기준이 전혀 없다 — 1프레임짜리
얼굴, "네"/"음" 같은 극단적으로 짧은 텍스트, 아주 짧은 오디오도 다른 정상 샘플과
동일한 가중치로 학습에 들어간다. 이 스크립트는 아무것도 수정하지 않고 분포만
보여준다 — 실제로 문제 수준인지 판단한 뒤에 필터링 여부를 결정하기 위함.

실행 예:
    python scripts/check_data_quality.py --manifest data/manifests/train.csv
"""
import argparse
from pathlib import Path

import pandas as pd
import soundfile as sf


def percentile_report(name: str, values: list[float], thresholds: list[float]) -> None:
    s = pd.Series(values)
    print(f"\n[{name}] n={len(s)}")
    print(f"  min={s.min():.2f} p1={s.quantile(0.01):.2f} p5={s.quantile(0.05):.2f} "
          f"median={s.median():.2f} p95={s.quantile(0.95):.2f} max={s.max():.2f}")
    for t in thresholds:
        n_below = (s < t).sum()
        print(f"  {t} 미만: {n_below}개 ({100 * n_below / len(s):.2f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--audio-sample-rate", type=float, default=1.0, help="오디오 길이 체크에 쓸 샘플 비율(0~1) — 전체 다 읽으면 느려서 기본은 표본만")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    print(f"[check_data_quality] {args.manifest}: 총 {len(df)}개 발화")

    # 1. 텍스트 길이 (문자 수, 공백 제거 기준)
    text_lens = df["text"].astype(str).str.strip().str.len().tolist()
    percentile_report("텍스트 길이(문자)", text_lens, thresholds=[1, 2, 3, 5])

    very_short_text = df[df["text"].astype(str).str.strip().str.len() <= 2]
    if len(very_short_text) > 0:
        print(f"  예시(길이<=2): {very_short_text['text'].astype(str).head(10).tolist()}")

    # 2. 얼굴 프레임 수
    frame_counts = []
    missing_dirs = 0
    for d in df["face_frames_dir"]:
        p = Path(d)
        if not p.exists():
            missing_dirs += 1
            frame_counts.append(0)
            continue
        frame_counts.append(len(list(p.glob("*"))))
    percentile_report("얼굴 프레임 수", frame_counts, thresholds=[2, 3, 5])
    if missing_dirs:
        print(f"  [경고] face_frames_dir 자체가 없는 행: {missing_dirs}개")

    # 3. 오디오 길이(초) — 전체 다 열면 느리므로 표본만 (soundfile.info는 헤더만 읽어 빠름)
    sample_df = df.sample(frac=args.audio_sample_rate, random_state=args.seed) if args.audio_sample_rate < 1.0 else df
    durations = []
    audio_missing = 0
    for wav_path in sample_df["wav_path"]:
        try:
            info = sf.info(wav_path)
            durations.append(info.frames / info.samplerate)
        except Exception:
            audio_missing += 1
    percentile_report(f"오디오 길이(초, 표본 {len(durations)}개)", durations, thresholds=[0.2, 0.3, 0.5, 1.0])
    if audio_missing:
        print(f"  [경고] 오디오 파일을 못 연 행(표본 중): {audio_missing}개")

    # 4. 완전 중복 발화(같은 텍스트가 몇 번이나 반복되는지 상위 목록만 — 그 자체가 문제는 아니지만 참고용)
    top_dup = df["text"].astype(str).str.strip().value_counts().head(10)
    print("\n[가장 많이 반복되는 텍스트 top10] (라벨 다양성 확인용 — 반복 자체는 정상일 수 있음)")
    print(top_dup.to_string())


if __name__ == "__main__":
    main()
