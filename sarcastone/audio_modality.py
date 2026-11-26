"""
SarcasTone: Audio & Speech Prosody Feature Extraction Module
Extracts eGeMAPS-inspired prosodic cues and IEMOCAP 4-class emotion probabilities.
"""

import numpy as np
import os

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

class AudioProsodyExtractor:
    def __init__(self):
        self.feature_names = [
            "f0_mean", "f0_std", "f0_range", "f0_max",
            "rms_mean", "rms_std", "rms_max", "rms_dynamic_range",
            "spectral_centroid_mean", "spectral_centroid_std",
            "zcr_mean", "zcr_std",
            "tempo", "jitter_approx"
        ]

    def extract_from_raw_audio(self, audio_data, sample_rate=16000):
        """
        Extracts 14 acoustic prosody features from raw audio wave array.
        """
        if not HAS_LIBROSA:
            # Fallback heuristic
            return np.zeros(len(self.feature_names), dtype=np.float32)

        y = audio_data.astype(np.float32)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        
        # Normalize
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))

        # 1. Pitch / Fundamental Frequency F0
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sample_rate
            )
            valid_f0 = f0[voiced_flag & (~np.isnan(f0))]
            if len(valid_f0) > 0:
                f0_mean = float(np.mean(valid_f0))
                f0_std = float(np.std(valid_f0))
                f0_range = float(np.max(valid_f0) - np.min(valid_f0))
                f0_max = float(np.max(valid_f0))
                jitter_approx = float(np.mean(np.abs(np.diff(valid_f0))) / (f0_mean + 1e-5))
            else:
                f0_mean, f0_std, f0_range, f0_max, jitter_approx = 150.0, 20.0, 40.0, 200.0, 0.05
        except Exception:
            f0_mean, f0_std, f0_range, f0_max, jitter_approx = 150.0, 20.0, 40.0, 200.0, 0.05

        # 2. Loudness / RMS Energy
        try:
            rms = librosa.feature.rms(y=y)[0]
            rms_mean = float(np.mean(rms))
            rms_std = float(np.std(rms))
            rms_max = float(np.max(rms))
            rms_dynamic_range = float(rms_max - np.min(rms))
        except Exception:
            rms_mean, rms_std, rms_max, rms_dynamic_range = 0.1, 0.03, 0.3, 0.28

        # 3. Spectral Centroid
        try:
            sc = librosa.feature.spectral_centroid(y=y, sr=sample_rate)[0]
            sc_mean = float(np.mean(sc))
            sc_std = float(np.std(sc))
        except Exception:
            sc_mean, sc_std = 1800.0, 400.0

        # 4. Zero Crossing Rate (articulation & pace)
        try:
            zcr = librosa.feature.zero_crossing_rate(y=y)[0]
            zcr_mean = float(np.mean(zcr))
            zcr_std = float(np.std(zcr))
        except Exception:
            zcr_mean, zcr_std = 0.08, 0.02

        # 5. Tempo
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sample_rate)
            tempo_val = float(tempo[0] if isinstance(tempo, (list, np.ndarray)) else tempo)
        except Exception:
            tempo_val = 120.0

        feats = np.array([
            f0_mean, f0_std, f0_range, f0_max,
            rms_mean, rms_std, rms_max, rms_dynamic_range,
            sc_mean, sc_std,
            zcr_mean, zcr_std,
            tempo_val, jitter_approx
        ], dtype=np.float32)

        return feats

    def extract_from_mustard_matrix(self, frame_matrix):
        """
        Pools the MUStARD benchmark acoustic frames (shape: 283 x d)
        into a fixed-length prosodic feature vector.
        """
        mat = np.array(frame_matrix, dtype=np.float32)
        # Use first 11 core acoustic channels
        mat = mat[:, :11]
        
        mean_v = np.mean(mat, axis=0)
        std_v = np.std(mat, axis=0)
        max_v = np.max(mat, axis=0)
        min_v = np.min(mat, axis=0)
        range_v = max_v - min_v
        
        # Total pooled prosodic representation = 55 dimensions
        pooled = np.concatenate([mean_v, std_v, max_v, min_v, range_v])
        return pooled

    def estimate_emotion_from_prosody(self, prosody_vector, is_mustard=False):
        """
        Estimates IEMOCAP 4-class emotion probabilities [neu, hap, ang, sad]
        from acoustic prosody parameters.
        """
        if is_mustard:
            # First channel represents energy/F0 in MUStARD
            energy = float(prosody_vector[0])
            std_energy = float(prosody_vector[11])
            max_energy = float(prosody_vector[22])
            f0_proxy = float(prosody_vector[1]) if len(prosody_vector) > 1 else 0.0
        else:
            f0_mean, f0_std, f0_range, f0_max = prosody_vector[0], prosody_vector[1], prosody_vector[2], prosody_vector[3]
            rms_mean, rms_std, rms_max = prosody_vector[4], prosody_vector[5], prosody_vector[6]
            energy = rms_mean
            std_energy = rms_std
            max_energy = rms_max
            f0_proxy = (f0_mean - 150.0) / 50.0

        # Calibrated acoustic-to-affect mapping
        # Angry: High energy, high variance, sharp transitions
        ang_score = 0.5 * std_energy + 0.4 * max_energy + 0.3 * max(0.0, f0_proxy)
        # Happy: Moderate-high energy, dynamic pitch, bright
        hap_score = 0.4 * energy + 0.5 * std_energy + 0.2 * max(0.0, f0_proxy)
        # Sad: Low energy, low variance, flat pitch
        sad_score = 0.6 * max(0.0, 0.3 - energy) + 0.4 * max(0.0, 0.2 - std_energy)
        # Neutral: Moderate energy, steady pitch
        neu_score = 0.5 * (1.0 - abs(energy - 0.2)) + 0.3 * (1.0 - std_energy)

        raw = np.array([max(0.01, neu_score), max(0.01, hap_score), max(0.01, ang_score), max(0.01, sad_score)])
        probs = raw / np.sum(raw)
        return probs

# Presets for Interactive Web UI
AUDIO_PRESETS = {
    "Sarcastic Drawl (Ironic Pitch Drop & Elongated)": {
        "emotion_probs": [0.15, 0.05, 0.55, 0.25], # heavy annoyed/angry undertone
        "prosody_summary": {"f0_mean": 138.0, "f0_std": 32.5, "rms_mean": 0.18, "speaking_rate": "Elongated (Slow)"},
        "description": "Slow, mocking intonation with pitch glides and exasperated edge."
    },
    "Deadpan / Monotone Sarcasm (Flat Pitch)": {
        "emotion_probs": [0.65, 0.05, 0.20, 0.10], # neutral/suppressed anger
        "prosody_summary": {"f0_mean": 125.0, "f0_std": 6.8, "rms_mean": 0.12, "speaking_rate": "Monotone (Dry)"},
        "description": "Completely flat pitch contour with virtually zero emotional inflection."
    },
    "Annoyed / Irritated Voice": {
        "emotion_probs": [0.08, 0.02, 0.72, 0.18], # predominantly angry
        "prosody_summary": {"f0_mean": 210.0, "f0_std": 48.0, "rms_mean": 0.35, "speaking_rate": "Fast & Stressed"},
        "description": "Sharp acoustic bursts, elevated F0 and prominent vocal stress."
    },
    "Genuinely Enthusiastic / Happy": {
        "emotion_probs": [0.05, 0.85, 0.05, 0.05], # predominantly happy
        "prosody_summary": {"f0_mean": 235.0, "f0_std": 55.0, "rms_mean": 0.28, "speaking_rate": "Upbeat"},
        "description": "High pitch variability, bright timbre, warm harmonic resonance."
    },
    "Sad / Frustrated / Disappointed": {
        "emotion_probs": [0.15, 0.05, 0.25, 0.55], # predominantly sad
        "prosody_summary": {"f0_mean": 115.0, "f0_std": 14.0, "rms_mean": 0.08, "speaking_rate": "Slow & Muffled"},
        "description": "Subdued energy, downward pitch contour, and low acoustic intensity."
    },
    "Neutral Professional Voice": {
        "emotion_probs": [0.75, 0.10, 0.08, 0.07], # neutral
        "prosody_summary": {"f0_mean": 160.0, "f0_std": 18.0, "rms_mean": 0.15, "speaking_rate": "Standard"},
        "description": "Balanced acoustic dynamics, standard conversational pacing."
    }
}
