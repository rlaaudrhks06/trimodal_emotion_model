"""실시간 데모 — 마이크로 말하면 v11이 감정을 판단해 화면에 띄운다.

흐름: 마이크 VAD로 발화 경계를 잡고 → 그 구간의 얼굴 프레임을 함께 꺼내고 →
Whisper로 전사한 뒤 → 세 가지를 v11에 한 번에 넣는다.

실행:
    python robot/demo.py --checkpoint checkpoints/v11_best.pt
    python robot/demo.py --checkpoint ... --no-camera     # 마이크만(얼굴 없이)
    python robot/demo.py --checkpoint ... --threshold 0.6 # 판단 보류 기준 조정

종료: 화면에서 q
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.brain.capture import FaceBuffer, MicVAD, VADConfig   # noqa: E402
from robot.brain.engine import EmotionEngine                    # noqa: E402
from robot.brain.preprocess import FaceCropper                  # noqa: E402

# 감정별 색(BGR). 설계도·대시보드와 같은 계열로 맞춘다.
COLOR = {"분노": (60, 24, 194), "혐오": (5, 92, 168), "공포": (172, 32, 109),
         "행복": (126, 139, 15), "슬픔": (201, 126, 74), "놀람": (6, 119, 217),
         "중립": (128, 114, 107)}
# 로봇 행동은 3클래스로 결정되므로 화면도 그쪽을 크게 보여준다.
COARSE_COLOR = {"긍정": (126, 175, 15), "부정": (60, 40, 200), "중립": (140, 128, 118)}


class Demo:
    def __init__(self, args):
        self.args = args
        self.engine = EmotionEngine(args.config, args.checkpoint,
                                    device=args.device,
                                    temperature=args.temperature,
                                    threshold=args.threshold,
                                    audio_pretrained=args.audio_pretrained,
                                    decide_on=args.decide_on)
        self.cropper = None if args.no_camera else FaceCropper()
        self.faces = FaceBuffer()
        self.vad = MicVAD(VADConfig(), self.faces if not args.no_camera else None)

        print("[demo] Whisper 로딩...")
        import whisper
        self.stt = whisper.load_model(args.whisper)

        self.log = deque(maxlen=6)
        self.state = "대기 중"
        self.last: object = None
        self.lock = threading.Lock()

    # ---------------------------------------------------------------- 발화 처리
    def on_utterance(self, utt) -> None:
        # Whisper는 빈·무음 오디오에서 "cannot reshape tensor of 0 elements"로 터진다.
        # VAD가 길이는 걸러주지만 전부 0인 블록(마이크 순간 끊김)은 통과하므로 여기서 막는다.
        import numpy as _np
        if utt.wav.size < 1600 or float(_np.abs(utt.wav).max()) < 1e-4:
            return
        with self.lock:
            self.state = "전사 중"
        try:
            r = self.stt.transcribe(utt.wav, language="ko", fp16=False)
            text = (r.get("text") or "").strip()
        except Exception as e:
            print(f"[demo] 전사 실패: {e}")
            with self.lock:
                self.state = "대기 중"
            return

        if not text:
            with self.lock:
                self.state = "대기 중"
            return

        with self.lock:
            self.state = "판단 중"
        try:
            res = self.engine.infer(utt.wav, utt.faces_bgr, text)
        except Exception as e:
            print(f"[demo] 추론 실패: {type(e).__name__}: {e}")
            with self.lock:
                self.state = "대기 중"
            return

        mark = "" if res.answered else "  [보류]"
        line = (f"{res.coarse} {100*res.coarse_confidence:.0f}%  "
                f"({res.emotion} {100*res.confidence:.0f}%){mark}  | {text[:24]}")
        print(f"[demo] {line}  ({res.latency_ms:.0f}ms · 얼굴 {len(utt.faces_bgr)}장 · "
              f"{utt.duration:.1f}초)")
        with self.lock:
            self.last = res
            self.log.appendleft(line)
            self.state = "대기 중"

    # ---------------------------------------------------------------- 화면
    def draw(self, frame):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 96), (28, 26, 24), -1)

        with self.lock:
            state, last, log = self.state, self.last, list(self.log)

        if last is not None and last.answered:
            col = COARSE_COLOR.get(last.coarse, (200, 200, 200))
            cv2.putText(frame, last.coarse, (18, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.7, col, 3)
            cv2.putText(frame,
                        f"{100*last.coarse_confidence:.0f}%   "
                        f"{last.emotion} {100*last.confidence:.0f}%", (18, 86),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
        elif last is not None:
            cv2.putText(frame, "HOLD", (18, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (150, 150, 150), 3)
            cv2.putText(frame, f"{100*last.confidence:.0f}% < "
                               f"{100*self.args.threshold:.0f}%", (18, 86),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

        # 마이크 레벨 막대 — 말하는 중이면 초록
        lv = min(1.0, self.vad.level / 3000.0)
        bar_col = (110, 190, 100) if self.vad.is_speaking else (90, 90, 90)
        cv2.rectangle(frame, (w - 230, 30), (w - 30, 46), (70, 70, 70), 1)
        cv2.rectangle(frame, (w - 229, 31),
                      (w - 229 + int(198 * lv), 45), bar_col, -1)
        cv2.putText(frame, state, (w - 230, 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (215, 215, 215), 1)

        for i, line in enumerate(log):
            cv2.putText(frame, line, (18, h - 18 - i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (235, 235, 235) if i == 0 else (150, 150, 150), 1)
        return frame

    # ---------------------------------------------------------------- 루프
    def run(self) -> None:
        self.vad.start(self.on_utterance)
        print("[demo] 준비됐다. 말해보라 — 종료는 q\n")

        if self.args.no_camera:
            try:
                while True:
                    time.sleep(0.3)
            except KeyboardInterrupt:
                pass
            self.vad.stop()
            return

        cap = cv2.VideoCapture(self.args.camera)
        if not cap.isOpened():
            print(f"[demo] 카메라 {self.args.camera}를 열 수 없다. --no-camera로 실행해보라")
            self.vad.stop()
            return

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                face = self.cropper(frame)
                if face is not None:
                    self.faces.push(face)
                    # 검출된 얼굴을 우측 상단에 미리보기로
                    frame[10:10 + face.shape[0], -10 - face.shape[1]:-10] = face
                cv2.imshow("v11 realtime", self.draw(frame))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.vad.stop()
            cap.release()
            cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "config_si_w2v.yaml"))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-camera", action="store_true",
                    help="얼굴 없이 오디오+텍스트만. 카메라가 없을 때 확인용")
    ap.add_argument("--whisper", default="small",
                    help="tiny/base/small/medium — 클수록 정확하지만 느리다")
    ap.add_argument("--temperature", type=float, default=1.17,
                    help="신뢰도 보정 온도(8.30.5절에서 val로 적합한 값)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="이 확신도 미만이면 판단을 보류한다")
    ap.add_argument("--decide-on", choices=["coarse", "7class"], default="coarse",
                    help="응답 여부를 무엇으로 판단할지. coarse는 긍정/부정/중립 그룹의 "
                         "확률 합을 쓴다 — 로봇 행동이 그 해상도로 결정되고 정확도도 "
                         "67.34%로 높다(8.29.3절)")
    ap.add_argument("--audio-pretrained", default=None,
                    help="config의 wav2vec2 경로를 대체한다. 서버의 변환본이 없는 맥에서는 "
                         "facebook/wav2vec2-large-xlsr-53 을 주면 HF 캐시로 돈다")
    args = ap.parse_args()

    if not Path(args.checkpoint).exists():
        raise SystemExit(f"체크포인트가 없다: {args.checkpoint}\n"
                         f"서버 archived_runs/checkpoint_v11_best/best_model.pt를 받아와야 한다")
    Demo(args).run()


if __name__ == "__main__":
    main()
