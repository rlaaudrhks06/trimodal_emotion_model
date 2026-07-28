"""음성 운율(prosody) 특징 벡터 p_a 추출. 설계 v3 §0/§5.2.

기존 robot_project 코드는 STT로 텍스트만 뽑고 "어떻게 말했는가"(억양·톤)는
전혀 쓰지 않았다는 것이 설계 변경의 핵심 동기였다. 여기서 F0(피치)·jitter·
shimmer·HNR 등 발화 단위 통계 특징을 뽑아 고정 길이 벡터로 반환한다.

주의: jitter/shimmer/HNR은 Praat(parselmouth)로 계산하는 것이 표준이지만,
이 프로젝트는 추가 시스템 의존성(Praat 바이너리)을 피하기 위해 librosa만으로
근사치를 계산한다. 정밀도가 중요해지면 `praat-parselmouth` 패키지로 교체할 것
(design v3 참고문헌의 eGeMAPS/openSMILE 방식이 더 정확함).
"""
import numpy as np
import librosa

PROSODY_DIM = 10
PROSODY_FEATURE_NAMES = [
    "f0_mean", "f0_std", "jitter_approx", "shimmer_approx", "hnr_approx_db",
    "energy_mean", "energy_std", "voiced_ratio", "duration_sec", "zcr_mean",
]


def extract_prosody(y: np.ndarray, sr: int) -> np.ndarray:
    """[T_samples] 파형 -> [PROSODY_DIM] 고정 길이 통계 벡터 p_a."""
    duration_sec = len(y) / sr

    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
    )
    voiced_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])

    if voiced_f0.size > 1:
        f0_mean = float(np.mean(voiced_f0))
        f0_std = float(np.std(voiced_f0))
        # jitter 근사: 연속 피치 주기 간 상대적 변동
        periods = 1.0 / (voiced_f0 + 1e-6)
        jitter_approx = float(np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + 1e-6))
    else:
        f0_mean, f0_std, jitter_approx = 0.0, 0.0, 0.0

    voiced_ratio = float(np.mean(voiced_flag)) if voiced_flag is not None and len(voiced_flag) > 0 else 0.0

    # shimmer 근사: 프레임 RMS 진폭의 연속 변동
    rms = librosa.feature.rms(y=y)[0]
    if rms.size > 1 and np.mean(rms) > 1e-8:
        shimmer_approx = float(np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + 1e-6))
    else:
        shimmer_approx = 0.0
    energy_mean, energy_std = float(np.mean(rms)), float(np.std(rms))

    # HNR 근사: harmonic-percussive 분리 후 조화/비조화 에너지비 (Praat HNR 대체 근사치)
    harmonic, percussive = librosa.effects.hpss(y)
    h_energy = float(np.sum(harmonic ** 2))
    p_energy = float(np.sum(percussive ** 2)) + 1e-8
    hnr_approx_db = float(10 * np.log10(h_energy / p_energy + 1e-8))

    zcr_mean = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    return np.array(
        [f0_mean, f0_std, jitter_approx, shimmer_approx, hnr_approx_db,
         energy_mean, energy_std, voiced_ratio, duration_sec, zcr_mean],
        dtype=np.float32,
    )
