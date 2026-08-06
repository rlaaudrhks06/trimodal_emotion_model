"""feature_cache의 얼굴 프레임을 float32 -> uint8로 다시 저장해 용량을 약 1/4로 줄인다.

배경: 캐시에 프레임을 0~1 float32로 저장하고 있었는데, 원본 픽셀은 0~255 정수라
4바이트가 필요 없다. 실측 결과 캐시 303GB 중 93%가 이 프레임이었고, uint8로 바꾸면
약 201GB가 확보된다(값 손실은 전혀 없다 — 0~1 변환은 읽는 시점에 한다).

캐시를 지우고 재생성하지 않는 이유: prosody 추출(librosa.pyin)이 매우 느려서
79,601개를 다시 계산하면 몇 시간이 걸린다. 이 스크립트는 mel/prosody는 그대로 두고
frames만 변환해 덮어쓴다.

안전장치:
- 이미 uint8인 파일은 건너뛴다(중단 후 재실행해도 안전).
- 임시 파일에 쓴 뒤 원자적으로 rename — 중간에 죽어도 반쯤 쓰인 캐시가 남지 않는다.
- --dry-run으로 예상 절감량만 먼저 확인할 수 있다.

실행 예:
    python scripts/shrink_feature_cache.py --cache-dir data/feature_cache --dry-run
    python scripts/shrink_feature_cache.py --cache-dir data/feature_cache
"""
import argparse
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--dry-run", action="store_true", help="변환하지 않고 예상 절감량만 계산")
    parser.add_argument("--limit", type=int, default=None, help="테스트용: 앞의 N개만 처리")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    files = sorted(cache_dir.glob("*.npz"))
    files = [f for f in files if not f.name.endswith(".tmp.npz")]
    if args.limit:
        files = files[:args.limit]
    print(f"[shrink] 대상 {len(files):,}개 (.npz)")

    before = after = 0
    converted = skipped = failed = 0
    t0 = time.time()

    for i, path in enumerate(files, 1):
        try:
            size_before = path.stat().st_size
            with np.load(path) as d:
                frames = d["frames"]
                if frames.dtype == np.uint8:
                    skipped += 1
                    before += size_before
                    after += size_before
                    continue
                mel, prosody = d["mel"], d["prosody"]
                # 0~1 float32 -> 0~255 uint8. round 없이 자르면 값이 1씩 어긋날 수 있어
                # 반올림 후 클리핑한다(원본 픽셀값을 정확히 복원하기 위함).
                frames_u8 = np.clip(np.rint(frames * 255.0), 0, 255).astype(np.uint8)

            before += size_before
            if args.dry_run:
                after += size_before // 4  # frames가 대부분이므로 대략 1/4로 추정
                converted += 1
            else:
                tmp = path.with_name(path.stem + ".tmp.npz")
                np.savez(str(tmp), mel=mel, prosody=prosody, frames=frames_u8)
                tmp.replace(path)
                after += path.stat().st_size
                converted += 1
        except Exception as e:
            failed += 1
            print(f"  [실패] {path.name}: {type(e).__name__} {e}")

        if i % 5000 == 0:
            el = (time.time() - t0) / 60
            print(f"  {i:,}/{len(files):,} ({100*i/len(files):.0f}%) "
                  f"변환 {converted:,} / 이미완료 {skipped:,} / 실패 {failed} — 경과 {el:.1f}분", flush=True)

    gb = 1024 ** 3
    print(f"\n[shrink] {'예상' if args.dry_run else '완료'}: "
          f"변환 {converted:,} / 이미 uint8 {skipped:,} / 실패 {failed}")
    print(f"  이전 {before/gb:.1f} GB -> 이후 {after/gb:.1f} GB "
          f"(절감 {(before-after)/gb:.1f} GB)")
    if args.dry_run:
        print("  * --dry-run 이므로 실제로 변경하지 않았습니다.")


if __name__ == "__main__":
    main()
