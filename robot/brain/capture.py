"""마이크·카메라에서 **발화 단위**로 오디오와 얼굴을 함께 잘라낸다.

핵심은 "발화 하나"의 경계를 정하는 일이다. 모델은 발화 단위로 학습됐으므로
실시간에서도 같은 단위로 잘라 줘야 한다.

VAD는 `robot_project/robot_companion_brain/core.py`의 적응형 노이즈 플로어 방식을
계승했다 — 고정 임계값은 방마다 다시 맞춰야 하는데, 조용한 구간의 RMS를 천천히
따라가며 기준을 스스로 올리고 내리면 환경이 바뀌어도 버틴다.

**얼굴은 오디오와 같은 구간에서만 모은다.** 발화가 끝난 뒤 프레임을 모으면
"말할 때의 표정"이 아니라 "말이 끝난 뒤의 표정"이 들어가 학습 조건과 어긋난다.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class VADConfig:
    sample_rate: int = 16_000
    chunk: int = 1024                 # 약 64ms
    # 조용한 구간의 RMS를 천천히 따라간다(1 - 0.98 = 2%씩 반영).
    noise_decay: float = 0.98
    # 말이라고 볼 기준 — 노이즈 플로어의 배수 + 절대 여유분.
    speech_mult: float = 2.5
    speech_floor: float = 80.0
    # 이만큼 조용하면 발화가 끝난 것으로 본다(청크 수 × 64ms).
    silence_chunks: int = 12
    min_speech_sec: float = 0.4       # 이보다 짧으면 잡음으로 보고 버린다
    max_speech_sec: float = 8.0       # 학습 최대 길이와 맞춘다


@dataclass
class Utterance:
    """발화 하나 — 오디오와 그 구간의 얼굴 프레임."""
    wav: np.ndarray                   # float32 -1~1, 16kHz
    faces_bgr: list = field(default_factory=list)
    started_at: float = 0.0
    duration: float = 0.0


class FaceBuffer:
    """카메라 스레드가 채우고, 발화가 끝날 때 그 구간만 꺼내 쓴다.

    (타임스탬프, 얼굴크롭)을 최근 것만 들고 있다가 구간으로 잘라 준다.
    검출 실패한 프레임은 아예 넣지 않는다 — 빈 크롭을 넣으면 모델이 검은 화면을
    표정으로 읽는다.
    """

    def __init__(self, seconds: float = 12.0, fps_hint: int = 15):
        self.buf = deque(maxlen=int(seconds * fps_hint) + 30)
        self.lock = threading.Lock()

    def push(self, face_bgr: np.ndarray) -> None:
        with self.lock:
            self.buf.append((time.time(), face_bgr))

    def slice(self, t0: float, t1: float, min_frames: int = 1) -> list:
        with self.lock:
            got = [f for (ts, f) in self.buf if t0 <= ts <= t1]
        if len(got) < min_frames:
            # 구간에 얼굴이 없으면 직전 프레임이라도 준다 — 아예 없는 것보다는 낫다.
            with self.lock:
                got = [f for (_, f) in list(self.buf)[-min_frames:]]
        return got


class MicVAD:
    """마이크에서 발화를 잘라 콜백으로 넘긴다. 별도 스레드에서 돈다."""

    def __init__(self, cfg: VADConfig, face_buffer: FaceBuffer | None = None):
        self.cfg = cfg
        self.faces = face_buffer
        self.running = False
        self.noise_floor = 150.0
        self.level = 0.0              # 화면 표시용 현재 RMS
        self.is_speaking = False
        self._thread: threading.Thread | None = None

    def start(self, on_utterance) -> None:
        self.running = True
        self._thread = threading.Thread(
            target=self._loop, args=(on_utterance,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False

    def _loop(self, on_utterance) -> None:
        # pyaudio 대신 sounddevice를 쓴다 — 휠에 portaudio가 들어 있어 brew 설치가
        # 필요 없다. 젯슨에서도 같은 이유로 설치가 간단하다.
        import sounddevice as sd

        c = self.cfg
        try:
            stream = sd.InputStream(samplerate=c.sample_rate, channels=1,
                                    dtype="int16", blocksize=c.chunk)
            stream.start()
        except Exception as e:
            print(f"[VAD] 마이크를 열 수 없다: {e}")
            print("      맥이라면 시스템 설정 > 개인정보 보호 > 마이크에서 터미널을 허용해야 한다")
            return

        chunks: list[np.ndarray] = []
        silent = 0
        recording = False
        t_start = 0.0
        max_chunks = int(c.max_speech_sec * c.sample_rate / c.chunk)

        while self.running:
            try:
                block, overflowed = stream.read(c.chunk)
            except Exception:
                continue
            data = block[:, 0].copy()            # [chunk] int16
            x = data.astype(np.float64)
            rms = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
            self.level = rms

            # 조용할 때만 바닥을 갱신한다 — 말소리로 바닥이 올라가면 안 된다.
            if 10 < rms < self.noise_floor:
                self.noise_floor = self.noise_floor * c.noise_decay + rms * (1 - c.noise_decay)
            speaking = rms > (self.noise_floor * c.speech_mult + c.speech_floor)
            self.is_speaking = speaking

            if speaking:
                if not recording:
                    recording, chunks, silent = True, [], 0
                    t_start = time.time()
                chunks.append(data)
                silent = 0
                if len(chunks) >= max_chunks:      # 너무 길면 여기서 끊는다
                    self._emit(chunks, t_start, on_utterance)
                    recording, chunks = False, []
            elif recording:
                chunks.append(data)
                silent += 1
                if silent > c.silence_chunks:
                    self._emit(chunks, t_start, on_utterance)
                    recording, chunks = False, []

        stream.stop()
        stream.close()

    def _emit(self, chunks: list, t_start: float, on_utterance) -> None:
        wav = np.concatenate(chunks).astype(np.float32) / 32768.0
        dur = len(wav) / self.cfg.sample_rate
        if dur < self.cfg.min_speech_sec:
            return
        t_end = time.time()
        faces = self.faces.slice(t_start, t_end) if self.faces else []
        # 추론은 오래 걸리므로 다른 스레드로 넘긴다 — 여기서 붙잡으면 다음 발화를 놓친다.
        threading.Thread(
            target=on_utterance,
            args=(Utterance(wav=wav, faces_bgr=faces, started_at=t_start, duration=dur),),
            daemon=True,
        ).start()
