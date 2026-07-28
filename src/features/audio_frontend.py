"""원본 파형(waveform) -> 멜-스펙트로그램 변환. 설계 v3 §4 ①~② 전처리 단계.

nn.Module이 아니라 순수 함수인 이유: 멜 변환 자체는 학습 파라미터가 없는
고정 신호처리 단계이며, 데이터 로더(datasets/kemdy_dataset.py)에서
__getitem__ 시점에 호출된다.
"""
import numpy as np
import librosa


def waveform_to_mel(
    y: np.ndarray,
    sr: int,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
) -> np.ndarray:
    """[T_samples] 파형 -> [T_a, n_mels] 로그 멜스펙트로그램."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)  # [n_mels, T_a]
    # 발화 단위 정규화 (평균 0, 표준편차 1)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    return log_mel.T.astype(np.float32)  # [T_a, n_mels]
