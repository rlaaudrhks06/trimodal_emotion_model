"""v11 추론 엔진 — 발화 하나를 받아 감정과 **행동 결정**을 돌려준다.

정확도만 내놓지 않고 판단 보류까지 하는 이유(8.30.5절): test 46.19%를 그대로
쓰면 두 번에 한 번은 틀린다. 확신도 0.5 이상만 응답하면 응답률 40.4%에 정확도
60.66%가 된다. **틀리게 확신하고 반응하는 로봇이 가만히 있는 로봇보다 신뢰를
빨리 잃는다.**

온도 보정도 여기서 한다. 보정 없이 확신도로 임계값을 걸면 같은 0.5가 다른 뜻이
된다 — v11은 살짝 과신하는 편이라 T=1.17로 눌러야 문서의 수치와 맞는다.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config                        # noqa: E402
from src.datasets.labels import EMOTION_LABELS            # noqa: E402
from src.model import TrimodalEmotionModel                # noqa: E402
from robot.brain.preprocess import build_batch            # noqa: E402

KO = {"angry": "분노", "disgust": "혐오", "fear": "공포", "happy": "행복",
      "sad": "슬픔", "surprise": "놀람", "neutral": "중립"}

# 8.29.3절에서 측정한 매핑. 로봇 행동은 이 해상도면 충분할 수 있다(3클래스 67.34%).
COARSE = {"happy": "긍정", "surprise": "긍정",
          "angry": "부정", "disgust": "부정", "fear": "부정", "sad": "부정",
          "neutral": "중립"}


@dataclass
class Result:
    emotion: str            # 7클래스 (한글)
    coarse: str             # 긍정/부정/중립
    confidence: float       # 7클래스 최대 확률 (보정 후)
    coarse_confidence: float  # 그 coarse 그룹의 확률 합
    probs: dict             # 감정 -> 확률 (보정 후)
    coarse_probs: dict      # 긍정/부정/중립 -> 확률 합
    answered: bool          # 임계값을 넘겨 실제로 응답하는가
    decided_on: str         # 응답 판단에 쓴 기준 ("7class" | "coarse")
    latency_ms: float
    text: str


class EmotionEngine:
    def __init__(self, config_path: str, checkpoint: str,
                 device: str | None = None, temperature: float = 1.17,
                 threshold: float = 0.5, audio_pretrained: str | None = None,
                 decide_on: str = "coarse"):
        self.cfg = load_config(Path(config_path))
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu"))
        self.temperature = temperature
        self.threshold = threshold
        # 응답 여부를 7클래스 최대 확률로 볼지, 3클래스 그룹 합으로 볼지.
        # 로봇 행동이 "좋음/나쁨/보통"으로 결정된다면 coarse가 맞는 기준이다.
        assert decide_on in ("7class", "coarse")
        self.decide_on = decide_on

        # ---- 준비물 검사를 **모델 로딩 전에** 한다. 백본 로딩만 수십 초라
        #      나중에 실패하면 그 시간을 통째로 버린다.
        self.prosody_stats = self._load_prosody_stats()
        if audio_pretrained:
            # config는 서버의 변환본 경로를 가리킨다. 맥에서는 HF 허브 id로 대체할 수
            # 있게 열어둔다 — 가중치는 같고 로딩 경로만 다르다.
            #
            # Config는 dataclass라 값이 로드 시점에 고정된다. raw만 고치면 모델은
            # 여전히 옛 경로를 읽는다(실제로 그렇게 한 번 실패했다). 필드를 바꾼다.
            print(f"[engine] 오디오 백본 경로 대체: "
                  f"{self.cfg.audio_pretrained} -> {audio_pretrained}")
            self.cfg.audio_pretrained = audio_pretrained
            if isinstance(self.cfg.raw.get("audio"), dict):
                self.cfg.raw["audio"]["pretrained_model"] = audio_pretrained
            assert self.cfg.audio_pretrained == audio_pretrained

        print(f"[engine] 모델 로딩 — {self.device}")
        self.model = TrimodalEmotionModel(self.cfg).to(self.device).eval()
        state = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(state)

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.text_pretrained)

        n = sum(x.numel() for x in self.model.parameters())
        print(f"[engine] 준비 완료 — 파라미터 {n:,} · 온도 {temperature} · "
              f"임계값 {threshold} ({decide_on} 기준)")

    def _load_prosody_stats(self) -> dict | None:
        """학습이 운율을 정규화했다면 추론도 **같은 통계**를 써야 한다.

        없는 채로 그냥 돌리면 운율 벡터의 스케일이 학습 때와 완전히 달라지는데
        (f0는 수백 단위, jitter는 0.01 단위) **에러 없이 그럴듯한 답이 나온다.**
        이 프로젝트가 반복해서 당한 "조용히 틀리는" 유형이라 여기서 막는다.
        """
        p = self.cfg.raw["train"].get("prosody_stats_path")
        if not p:
            return None
        path = ROOT / p
        if not path.exists():
            raise SystemExit(
                f"\n[engine] prosody 정규화 통계가 없다: {p}\n"
                f"  이 config는 학습 때 운율을 정규화했으므로 추론도 같은 통계를 써야 한다.\n"
                f"  없는 채로 돌리면 에러 없이 틀린 답이 나온다.\n\n"
                f"  서버에서 받아올 것:\n"
                f"    scp tta@<서버>:/data/aihub_download/trimodal_emotion_model/{p} {p}\n"
            )
        print(f"[engine] prosody 정규화 통계 적용: {p}")
        return json.loads(path.read_text(encoding="utf-8"))

    @torch.no_grad()
    def infer(self, wav: np.ndarray, faces_bgr: list, text: str) -> Result:
        t0 = time.perf_counter()
        batch = build_batch(wav, faces_bgr, text, self.tokenizer,
                            self.cfg, self.device, self.prosody_stats)
        logits = self.model(**batch)

        # 온도 스케일링은 순위를 바꾸지 않는다 — 예측은 그대로고 확신도만 조정된다.
        p = torch.softmax(logits.float() / self.temperature, dim=-1)[0].cpu().numpy()
        i = int(p.argmax())
        name = EMOTION_LABELS[i]
        conf = float(p[i])

        # coarse 확률은 **그룹에 속한 감정들의 확률을 더한 값**이다. 7클래스에서
        # 슬픔 30·혐오 25·분노 15로 흩어져 있어도 "부정"으로는 70이 된다.
        # 로봇 행동이 3클래스 해상도로 결정된다면 이쪽이 실제 확신도에 가깝고,
        # 정확도도 7클래스 46.19%가 아니라 3클래스 67.34%가 적용된다(8.29.3절).
        cp: dict[str, float] = {}
        for j, e in enumerate(EMOTION_LABELS):
            cp[COARSE[e]] = cp.get(COARSE[e], 0.0) + float(p[j])
        top_coarse = max(cp, key=cp.get)
        coarse_conf = cp[top_coarse]

        if self.decide_on == "coarse":
            answered, shown_coarse = coarse_conf >= self.threshold, top_coarse
        else:
            answered, shown_coarse = conf >= self.threshold, COARSE[name]

        return Result(
            emotion=KO[name],
            coarse=shown_coarse,
            confidence=conf,
            coarse_confidence=coarse_conf,
            probs={KO[e]: float(p[j]) for j, e in enumerate(EMOTION_LABELS)},
            coarse_probs=cp,
            answered=answered,
            decided_on=self.decide_on,
            latency_ms=(time.perf_counter() - t0) * 1000,
            text=text,
        )
