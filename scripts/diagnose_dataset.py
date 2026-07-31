"""매니페스트의 모든 발화를 순회하며 비정상적으로 느리거나 멈추는 샘플을 찾아낸다.

검증(val) 단계에서 학습(train)보다 훨씬 오래 걸리다 못해 GPU 사용률이 0%에
가깝게 떨어지는 현상이 batch_size를 16→128로 바꿔도 똑같이 재현되어, 배치
크기 문제가 아니라 **특정 샘플(손상되었거나 비정상적으로 큰 캐시 파일 등)에서
멈추는 문제**로 의심하고 작성했다. feature_cache가 이미 채워져 있어 정상
샘플은 np.load만 하면 되므로 매우 빨라야 한다 — 그런데도 느리거나 아예 응답이
없는(타임아웃) 샘플이 있다면 그게 범인이다.

사용 예:
    python scripts/diagnose_dataset.py --manifest data/manifests/val.csv
    python scripts/diagnose_dataset.py --manifest data/manifests/train.csv --timeout-sec 15
"""
import argparse
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.datasets.manifest_dataset import ManifestEmotionDataset


class _Timeout(Exception):
    pass


def _handler(signum, frame):
    raise _Timeout()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--timeout-sec", type=int, default=10, help="이 시간(초) 안에 안 끝나면 의심 샘플로 기록")
    parser.add_argument("--slow-threshold-sec", type=float, default=2.0, help="이보다 오래 걸리면 '느림'으로 출력(에러는 아님)")
    args = parser.parse_args()

    cfg = load_config()
    cache_dir = cfg.raw["train"].get("feature_cache_dir")
    ds = ManifestEmotionDataset(args.manifest, cfg, cache_dir=cache_dir)
    print(f"[diagnose] {args.manifest}: 총 {len(ds)}개 발화 검사 시작 (샘플당 타임아웃 {args.timeout_sec}초)")

    signal.signal(signal.SIGALRM, _handler)

    problem_ids = []
    for i in range(len(ds)):
        row = ds.df.iloc[i]
        signal.alarm(args.timeout_sec)
        t0 = time.time()
        try:
            item = ds[i]
            dt = time.time() - t0
            if dt > args.slow_threshold_sec:
                print(f"[느림] ({i}/{len(ds)}) {row['utt_id']}: {dt:.1f}초 | mel={item['mel'].shape} frames={item['frames'].shape}")
        except _Timeout:
            print(f"[타임아웃!] ({i}/{len(ds)}) {row['utt_id']}: {args.timeout_sec}초 초과 -> 의심 샘플 (wav={row['wav_path']}, face_dir={row['face_frames_dir']})")
            problem_ids.append(row["utt_id"])
        except Exception as e:
            print(f"[에러] ({i}/{len(ds)}) {row['utt_id']}: {e}")
            problem_ids.append(row["utt_id"])
        finally:
            signal.alarm(0)

        if (i + 1) % 2000 == 0:
            print(f"[diagnose] 진행: {i + 1}/{len(ds)}")

    print(f"\n[diagnose] 완료. 문제(타임아웃/에러) 샘플 {len(problem_ids)}개")
    for uid in problem_ids[:50]:
        print(f"  - {uid}")


if __name__ == "__main__":
    main()
