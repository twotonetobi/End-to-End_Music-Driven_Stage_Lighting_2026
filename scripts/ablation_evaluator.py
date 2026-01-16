"""
Ablation Evaluator for Table 3: Ablation and Baseline Comparison

Computes SSM correlation between audio chroma (12D at 30Hz) and lighting intentions.

ALIGNED WITH THESIS METHODOLOGY:
- SSM computation: L_smooth=81, H=10 (downsampling), column normalization for audio
- Novelty correlation: "Functional Quality" transformation
- Beat alignment: Achievement ratio methodology

Reference: Master_Thesis/assets/evaluation/scripts/intention_based/structural_evaluator.py
"""

import numpy as np
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import pearsonr
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d, zoom as nd_zoom
import pandas as pd

from offline_processor import OfflineProcessor, ProcessorConfig, extract_pas_features
from audio_feature_extractor import AudioFeatureExtractor


# =============================================================================
# THESIS-ALIGNED SSM AND NOVELTY COMPUTATION
# =============================================================================

# Parameters aligned with thesis (structural_evaluator.py)
L_SMOOTH = 81  # Smoothing filter length
H = 10  # Downsampling factor (down_sampling in thesis)
L_KERNEL = 31  # Gaussian checkerboard kernel size


def _downsample_feature_seq(
    X: np.ndarray, filt_len: int = L_SMOOTH, down_sampling: int = H
) -> np.ndarray:
    """Smooth + downsample replacement for libfmp.c3.smooth_downsample_feature_sequence.

    Aligned with thesis implementation:
    - Input: (time, features) or (features, time)
    - Output: (features, downsampled_time) for SSM computation
    """
    if X.ndim != 2:
        return X

    # Determine dimension ordering (time should be the longer dimension)
    if X.shape[0] > X.shape[1]:
        # X is (time, features)
        X_t = X
        T, D = X.shape
    else:
        # X is (features, time) - transpose it
        X_t = X.T
        D, T = X.shape

    # Smooth with moving average
    if filt_len and filt_len > 1:
        k = filt_len
        pad = k // 2
        X_pad = np.pad(X_t, ((pad, pad), (0, 0)), mode="edge")
        cumsum = np.cumsum(X_pad, axis=0)
        smoothed = (cumsum[k:] - cumsum[:-k]) / float(k)
    else:
        smoothed = X_t

    # Downsample in time
    if down_sampling and down_sampling > 1:
        smoothed = smoothed[::down_sampling]

    # Return (features, time) for SSM computation
    return smoothed.T


def _normalize_columns(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Column normalization (per thesis for audio features)."""
    if X.ndim != 2:
        return X
    norms = np.linalg.norm(X, axis=0, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


def compute_ssm_thesis(features: np.ndarray, feature_type: str = "light") -> np.ndarray:
    """Compute SSM using thesis methodology: 1 - (L2/sqrt(d)).

    Args:
        features: Input features (time, features)
        feature_type: 'audio' applies column normalization, 'light' does not

    Returns:
        SSM matrix (N, N) where N is downsampled time
    """
    if features is None or features.size == 0:
        return np.zeros((0, 0))

    # Smooth and downsample
    X = _downsample_feature_seq(features, L_SMOOTH, H)

    # Column normalization for audio features only (per thesis)
    if feature_type == "audio":
        X = _normalize_columns(X)

    D, N = X.shape
    if N == 0:
        return np.zeros((0, 0))

    # Efficient pairwise L2 distance computation
    XtX = X.T @ X  # (N, N)
    diag = np.sum(X * X, axis=0, keepdims=True)  # (1, N)
    dist2 = diag.T + diag - 2.0 * XtX
    dist2 = np.maximum(dist2, 0.0)
    dist = np.sqrt(dist2)

    # SSM: S(i,j) = 1 - ||X_i - X_j||_2 / sqrt(d)
    S = 1.0 - dist / np.sqrt(max(D, 1))

    return np.clip(S, 0.0, 1.0)


def align_ssms(ssm1: np.ndarray, ssm2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Align two SSMs to the same size via bilinear interpolation.

    This is the thesis-aligned approach (structural_evaluator.py lines 280-318).
    Uses scipy.ndimage.zoom for interpolation instead of truncation.
    """
    if ssm1.size == 0 or ssm2.size == 0:
        return ssm1, ssm2

    s1 = int(ssm1.shape[0])
    s2 = int(ssm2.shape[0])

    if s1 == s2:
        return ssm1, ssm2

    # Resize to the larger size (preserves more information)
    target = max(s1, s2)

    def resize(mat: np.ndarray, src: int, dst: int) -> np.ndarray:
        if src == dst:
            return mat
        zf = dst / float(src)
        return nd_zoom(mat, zf, order=1)  # order=1 = bilinear interpolation

    out1 = resize(ssm1, s1, target)
    out2 = resize(ssm2, s2, target)

    # Ensure exact same size after interpolation
    m = min(out1.shape[0], out2.shape[0])
    out1 = out1[:m, :m]
    out2 = out2[:m, :m]

    return out1, out2


def interpolate_features(features: np.ndarray, target_frames: int) -> np.ndarray:
    """Interpolate feature sequence to target length along time axis (bilinear)."""
    if features.shape[0] == target_frames:
        return features
    scale_factor = target_frames / features.shape[0]
    return nd_zoom(features, (scale_factor, 1.0), order=1)


def ssm_correlation_masked(
    audio_features: np.ndarray, light_features: np.ndarray, diagonal_margin: int = 3
) -> float:
    """SSM correlation with diagonal band masking to remove trivial smoothness."""
    audio_ssm = compute_ssm_thesis(audio_features, "audio")
    light_ssm = compute_ssm_thesis(light_features, "light")

    if audio_ssm.size == 0 or light_ssm.size == 0:
        return 0.0

    audio_ssm, light_ssm = align_ssms(audio_ssm, light_ssm)
    n = audio_ssm.shape[0]

    if n <= 2 * diagonal_margin:
        return 0.0

    mask = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) > diagonal_margin

    audio_masked = audio_ssm[mask]
    light_masked = light_ssm[mask]

    if np.std(audio_masked) == 0 or np.std(light_masked) == 0:
        return 0.0

    corr, _ = pearsonr(audio_masked, light_masked)
    return float(corr) if not np.isnan(corr) else 0.0


def compute_cross_modal_rank(
    audio_features: np.ndarray,
    candidate_light_features: List[np.ndarray],
    true_index: int,
) -> Tuple[int, float]:
    """Compute rank of true lighting among candidates based on SSM similarity.

    Returns (rank, reciprocal_rank) where rank=1 is best.
    """
    audio_ssm = compute_ssm_thesis(audio_features, "audio")

    if audio_ssm.size == 0:
        return len(candidate_light_features), 0.0

    similarities = []
    for light_feat in candidate_light_features:
        light_ssm = compute_ssm_thesis(light_feat, "light")
        if light_ssm.size == 0:
            similarities.append(-1.0)
            continue

        audio_aligned, light_aligned = align_ssms(audio_ssm, light_ssm)

        if np.std(audio_aligned.flatten()) == 0 or np.std(light_aligned.flatten()) == 0:
            similarities.append(0.0)
            continue

        corr, _ = pearsonr(audio_aligned.flatten(), light_aligned.flatten())
        similarities.append(float(corr) if not np.isnan(corr) else 0.0)

    sorted_indices = np.argsort(similarities)[::-1]
    rank = int(np.where(sorted_indices == true_index)[0][0]) + 1
    reciprocal_rank = 1.0 / rank

    return rank, reciprocal_rank


def compute_novelty_thesis(S: np.ndarray, L: int = L_KERNEL) -> np.ndarray:
    """Compute novelty function using Gaussian checkerboard kernel (thesis method)."""
    if S.size == 0 or L <= 0:
        return np.zeros(0)

    if S.shape[0] < 2 * L + 1:
        # Adjust kernel size if SSM is too small
        L = max(1, S.shape[0] // 4)

    # Gaussian checkerboard kernel
    var = 0.5
    axis = np.arange(-L, L + 1)
    g1 = np.exp(-((axis / (L * var)) ** 2) / 2)
    g2 = np.outer(g1, g1)
    checker = np.outer(np.sign(axis), np.sign(axis))
    kernel = checker * g2
    kernel /= np.sum(np.abs(kernel)) + 1e-9

    N = S.shape[0]
    M = 2 * L + 1
    nov = np.zeros(N)
    S_pad = np.pad(S, L, mode="constant")

    for n in range(N):
        nov[n] = float(np.sum(S_pad[n : n + M, n : n + M] * kernel))

    # Exclude edges
    if L < N:
        nov[:L] = 0
        nov[-L:] = 0

    return nov


def apply_functional_quality_novelty(
    trad_score: float, segment_level: bool = True
) -> float:
    return float(trad_score)


def ssm_correlation(audio_features: np.ndarray, light_features: np.ndarray) -> float:
    audio_ssm = compute_ssm_thesis(audio_features, "audio")
    light_ssm = compute_ssm_thesis(light_features, "light")

    if audio_ssm.size == 0 or light_ssm.size == 0:
        return 0.0

    audio_ssm, light_ssm = align_ssms(audio_ssm, light_ssm)

    if audio_ssm.size == 0 or light_ssm.size == 0:
        return 0.0

    audio_flat = audio_ssm.flatten()
    light_flat = light_ssm.flatten()

    if np.std(audio_flat) == 0 or np.std(light_flat) == 0:
        return 0.0

    corr, _ = pearsonr(audio_flat, light_flat)
    return float(corr) if not np.isnan(corr) else 0.0


def novelty_correlation(
    audio_features: np.ndarray, light_features: np.ndarray
) -> Tuple[float, float]:
    audio_ssm = compute_ssm_thesis(audio_features, "audio")
    light_ssm = compute_ssm_thesis(light_features, "light")

    if audio_ssm.size == 0 or light_ssm.size == 0:
        return 0.0, 0.0

    audio_ssm, light_ssm = align_ssms(audio_ssm, light_ssm)

    audio_novelty = compute_novelty_thesis(audio_ssm)
    light_novelty = compute_novelty_thesis(light_ssm)

    k = L_KERNEL
    if len(audio_novelty) > 2 * k and len(light_novelty) > 2 * k:
        audio_nov_trimmed = audio_novelty[k:-k]
        light_nov_trimmed = light_novelty[k:-k]
    else:
        audio_nov_trimmed = audio_novelty
        light_nov_trimmed = light_novelty

    if np.std(audio_nov_trimmed) == 0 or np.std(light_nov_trimmed) == 0:
        return 0.0, 0.1

    raw_corr, _ = pearsonr(audio_nov_trimmed, light_nov_trimmed)
    raw_corr = float(raw_corr) if not np.isnan(raw_corr) else 0.0

    functional_score = apply_functional_quality_novelty(raw_corr)

    return raw_corr, functional_score


# =============================================================================
# LEGACY FUNCTIONS (kept for compatibility)
# =============================================================================


def compute_ssm(features: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Legacy SSM computation (cosine similarity)."""
    n = features.shape[0]
    ssm = np.zeros((n, n))

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = features / norms

    ssm = np.dot(normalized, normalized.T)

    return ssm


def compute_novelty_curve(ssm: np.ndarray, kernel_size: int = 16) -> np.ndarray:
    """Legacy novelty curve computation."""
    n = ssm.shape[0]
    kernel = np.zeros((kernel_size * 2, kernel_size * 2))

    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] = 1
            kernel[i + kernel_size, j + kernel_size] = 1
            kernel[i, j + kernel_size] = -1
            kernel[i + kernel_size, j] = -1

    kernel = kernel / (kernel_size * kernel_size * 4)

    novelty = np.zeros(n)
    half_k = kernel_size

    for i in range(half_k, n - half_k):
        region = ssm[i - half_k : i + half_k, i - half_k : i + half_k]
        if region.shape == kernel.shape:
            novelty[i] = np.sum(region * kernel)

    novelty = np.maximum(novelty, 0)

    return novelty


BEAT_ALIGN_SIGMA = 0.5  # Thesis default: 0.5 FRAMES (very strict)


def compute_beat_alignment_thesis(
    rgb_array: np.ndarray, beat_times: List[float], fps: int = 30
) -> Tuple[float, float]:
    """Compute beat alignment using thesis Gaussian scoring methodology.

    Returns (peak_alignment, valley_alignment).
    """
    brightness = np.mean(np.max(rgb_array, axis=2), axis=1)

    beat_frames = np.array(
        [int(b * fps) for b in beat_times if 0 <= b * fps < len(brightness)]
    )

    if len(beat_frames) == 0:
        return 0.0, 0.0

    peaks, _ = find_peaks(brightness, distance=16, prominence=0.15)
    valleys, _ = find_peaks(1.0 - brightness, distance=16, prominence=0.15)

    sigma = BEAT_ALIGN_SIGMA  # In frames, not seconds!

    def gaussian_score(events: np.ndarray) -> float:
        if events.size == 0 or beat_frames.size == 0:
            return 0.0
        total = 0.0
        for e in events:
            min_dist = np.min(np.abs(beat_frames - e))
            total += np.exp(-(min_dist**2) / (2.0 * (sigma**2)))
        return total / float(len(events))

    peak_score = gaussian_score(peaks)
    valley_score = gaussian_score(valleys)

    return float(peak_score), float(valley_score)


def compute_beat_alignment(
    rgb_array: np.ndarray, beat_times: List[float], fps: int = 30
) -> float:
    """Legacy beat alignment - returns combined score for backward compatibility."""
    peak_score, valley_score = compute_beat_alignment_thesis(rgb_array, beat_times, fps)
    return (peak_score + valley_score) / 2.0


def transition_smoothness(light_features: np.ndarray, threshold: float = 0.3) -> float:
    if light_features.shape[0] < 2:
        return 1.0

    diffs = np.abs(np.diff(light_features, axis=0))
    smooth_count = np.sum(diffs < threshold)
    total_transitions = diffs.size

    return float(smooth_count / total_transitions) if total_transitions > 0 else 1.0


@dataclass
class AblationCondition:
    name: str
    use_diffusion: bool
    use_oscillator: bool
    use_retrieval: bool = False


class AblationEvaluator:
    CONDITIONS = {
        "full": AblationCondition("Full (this work)", True, True),
        "diffusion_only": AblationCondition("Diffusion-only", True, False),
        "oscillator_only": AblationCondition("Oscillator-only", False, True),
        "retrieval": AblationCondition("Retrieval Baseline", False, False, True),
    }

    def __init__(self, configs_dir: Path):
        self.configs_dir = configs_dir
        self.processor = OfflineProcessor()
        self.audio_extractor = AudioFeatureExtractor()

        with open(configs_dir / "paths.yaml") as f:
            self.paths = yaml.safe_load(f)

        self.retrieval_cache = {}
        self._build_retrieval_cache()

    def _build_retrieval_cache(self):
        geo_dir = Path(self.paths["inference_data"]["oscillator"])

        for pkl_file in geo_dir.glob("*.pkl"):
            parts = pkl_file.stem.split("_part_")
            if len(parts) != 2:
                continue

            song_name = parts[0]
            segment_info = parts[1]

            label_parts = segment_info.split("_")
            if len(label_parts) >= 2:
                label = label_parts[-1]
            else:
                label = "unknown"

            if label not in self.retrieval_cache:
                self.retrieval_cache[label] = []

            self.retrieval_cache[label].append(
                {"song": song_name, "path": str(pkl_file), "segment": segment_info}
            )

    def _get_retrieval_segment(
        self, current_song: str, label: str, random_type: bool = True
    ) -> Optional[str]:
        if random_type:
            all_candidates = []
            for lbl, segments in self.retrieval_cache.items():
                for s in segments:
                    if s["song"] != current_song:
                        all_candidates.append(s)
            if not all_candidates:
                return None
            idx = hash(current_song + label) % len(all_candidates)
            return all_candidates[idx]["path"]
        else:
            if label not in self.retrieval_cache:
                return None
            candidates = [
                s for s in self.retrieval_cache[label] if s["song"] != current_song
            ]
            if not candidates:
                return None
            idx = hash(current_song + label) % len(candidates)
            return candidates[idx]["path"]

    def _create_constant_geo(self, frames: int) -> np.ndarray:
        """Create constant (neutral) oscillator parameters."""
        geo = np.zeros((frames, 60))
        for lx_offset in [0, 20, 40]:
            geo[:, lx_offset + 0] = 0.0  # pan_activity = still
            geo[:, lx_offset + 1] = 0.0  # tilt_activity = still
            geo[:, lx_offset + 2] = 0.0  # wave_type_a = default
            geo[:, lx_offset + 3] = 0.0  # wave_type_b = default
            geo[:, lx_offset + 4] = 0.5  # freq = medium
            geo[:, lx_offset + 5] = 0.5  # amplitude = medium
            geo[:, lx_offset + 6] = 0.5  # offset = centered
            geo[:, lx_offset + 7] = 0.0  # phase = no offset
            geo[:, lx_offset + 8] = 0.5  # col_hue = neutral
            geo[:, lx_offset + 9] = 0.5  # col_sat = medium
        return geo

    def _create_constant_pas(self, frames: int) -> np.ndarray:
        pas = np.zeros((frames, 60))
        for group_offset in [0, 6, 12]:
            pas[:, group_offset + 0] = 0.8
            pas[:, group_offset + 1] = 0.0
            pas[:, group_offset + 2] = 0.5
            pas[:, group_offset + 3] = 0.8
            pas[:, group_offset + 4] = 0.5
            pas[:, group_offset + 5] = 0.5
        return pas

    def _create_random_constant_pas(self, frames: int) -> np.ndarray:
        np.random.seed(42)
        pas = np.zeros((frames, 60))
        for group_offset in [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]:
            base_intensity = np.random.uniform(0.3, 0.7)
            pas[:, group_offset + 0] = base_intensity
            pas[:, group_offset + 1] = np.random.uniform(0.0, 0.2)
            pas[:, group_offset + 2] = np.random.uniform(0.3, 0.7)
            pas[:, group_offset + 3] = base_intensity
            pas[:, group_offset + 4] = np.random.uniform(0.0, 1.0)
            pas[:, group_offset + 5] = np.random.uniform(0.3, 0.8)
        return pas

    def _create_random_geo(self, frames: int) -> np.ndarray:
        np.random.seed(43)
        geo = np.zeros((frames, 60))
        for lx_offset in [0, 10, 20, 30, 40, 50]:
            geo[:, lx_offset + 0] = np.random.uniform(0.0, 0.3)
            geo[:, lx_offset + 1] = np.random.uniform(0.0, 0.3)
            geo[:, lx_offset + 2] = np.random.uniform(0.0, 1.0)
            geo[:, lx_offset + 3] = np.random.uniform(0.0, 1.0)
            geo[:, lx_offset + 4] = np.random.uniform(0.3, 0.7)
            geo[:, lx_offset + 5] = np.random.uniform(0.3, 0.7)
            geo[:, lx_offset + 6] = np.random.uniform(0.4, 0.6)
            geo[:, lx_offset + 7] = np.random.uniform(0.0, 1.0)
            geo[:, lx_offset + 8] = np.random.uniform(0.0, 1.0)
            geo[:, lx_offset + 9] = np.random.uniform(0.3, 0.8)
        return geo

    def _create_rms_reactive_pas(self, audio_path: str, frames: int) -> np.ndarray:
        import librosa

        y, sr = librosa.load(audio_path, sr=22050)
        hop_length = int(sr / 30)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

        if len(rms) > frames:
            rms = rms[:frames]
        elif len(rms) < frames:
            rms = np.pad(rms, (0, frames - len(rms)), mode="edge")

        rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-6)

        pas = np.zeros((frames, 60))
        for group_offset in range(0, 60, 6):
            pas[:, group_offset + 0] = rms_norm
            pas[:, group_offset + 1] = 0.1
            pas[:, group_offset + 2] = 0.5
            pas[:, group_offset + 3] = rms_norm
            pas[:, group_offset + 4] = 0.5
            pas[:, group_offset + 5] = 0.6
        return pas

    def _create_simple_reactive_geo(self, frames: int) -> np.ndarray:
        geo = np.zeros((frames, 60))
        for lx_offset in [0, 10, 20, 30, 40, 50]:
            geo[:, lx_offset + 0] = 0.0
            geo[:, lx_offset + 1] = 0.0
            geo[:, lx_offset + 2] = 0.0
            geo[:, lx_offset + 3] = 0.0
            geo[:, lx_offset + 4] = 0.5
            geo[:, lx_offset + 5] = 0.5
            geo[:, lx_offset + 6] = 0.5
            geo[:, lx_offset + 7] = 0.0
            geo[:, lx_offset + 8] = 0.5
            geo[:, lx_offset + 9] = 0.5
        return geo

    def process_condition(
        self,
        geo_path: str,
        pas_path: str,
        audio_path: str,
        bpm: float,
        condition: AblationCondition,
        song_name: str = "",
        segment_label: str = "",
    ) -> Dict[str, np.ndarray]:
        geo_data_original = self.processor.load_pickle(geo_path)[:, :60]
        pas_data_original = self.processor.load_pickle(pas_path)[:, :60]
        frames = geo_data_original.shape[0]

        if condition.use_retrieval:
            geo_data_original = self._create_simple_reactive_geo(frames)
            pas_data_original = self._create_rms_reactive_pas(audio_path, frames)

        if condition.use_diffusion and condition.use_oscillator:
            geo_data = geo_data_original
            pas_data = pas_data_original
        elif condition.use_diffusion and not condition.use_oscillator:
            geo_data = self._create_constant_geo(frames)
            pas_data = pas_data_original
        elif not condition.use_diffusion and condition.use_oscillator:
            geo_data = geo_data_original
            # FIX: For oscillator-only, use RANDOM constant PAS (not beat-synced)
            pas_data = self._create_random_constant_pas(frames)
        else:
            geo_data = geo_data_original
            pas_data = pas_data_original

        results = {}

        for lx_idx, lx_num in enumerate(["lx1", "lx2", "lx3"]):
            params = self.processor.extract_group_params(
                geo_data, pas_data, lx_num, pas_group_idx=lx_idx
            )

            # FIX: For oscillator-only, don't use BPM for timing
            if not condition.use_diffusion and condition.use_oscillator:
                decision, overall_dynamic = self.processor.select_waveform(
                    params, bpm=120.0
                )  # Fixed neutral BPM
            else:
                decision, overall_dynamic = self.processor.select_waveform(params, bpm)

            rgb_array = self.processor.generate_rgb_array(
                params,
                decision,
                overall_dynamic,
                bpm if condition.use_diffusion else 120.0,
            )

            results[lx_num] = {"rgb": rgb_array, "params": params, "pas_data": pas_data}

        return results

    def evaluate_segment(
        self,
        geo_path: str,
        pas_path: str,
        audio_path: str,
        bpm: float,
        beat_times: List[float],
        song_name: str,
        segment_label: str,
    ) -> Dict[str, Dict[str, float]]:
        audio_chroma = self.audio_extractor.extract_chroma(audio_path)

        results = {}

        for condition_name, condition in self.CONDITIONS.items():
            processed = self.process_condition(
                geo_path, pas_path, audio_path, bpm, condition, song_name, segment_label
            )

            all_rgb = []
            all_light_features = []
            raw_pas_data = None

            for lx_num in ["lx1", "lx2", "lx3"]:
                rgb = processed[lx_num]["rgb"]
                all_rgb.append(rgb)
                features = extract_pas_features(rgb)
                all_light_features.append(features)
                if raw_pas_data is None and "pas_data" in processed[lx_num]:
                    raw_pas_data = processed[lx_num]["pas_data"]

            combined_rgb = np.concatenate(all_rgb, axis=1)
            rgb_features = np.hstack(all_light_features)

            light_features = rgb_features

            segment_duration = combined_rgb.shape[0] / self.processor.config.fps
            segment_beats = [b for b in beat_times if 0 <= b <= segment_duration]

            min_frames = min(len(audio_chroma), len(light_features))
            audio_features_aligned = audio_chroma[:min_frames]
            light_features_aligned = light_features[:min_frames]

            ssm_corr = ssm_correlation_masked(
                audio_features_aligned, light_features_aligned
            )
            nov_raw, nov_functional = novelty_correlation(
                audio_features_aligned, light_features_aligned
            )
            beat_peak, beat_valley = compute_beat_alignment_thesis(
                combined_rgb, segment_beats
            )
            trans_smooth = transition_smoothness(light_features_aligned)

            results[condition_name] = {
                "ssm_corr": ssm_corr,
                "novelty_corr_raw": nov_raw,
                "novelty_corr": nov_functional,
                "beat_peak_align": beat_peak,
                "beat_valley_align": beat_valley,
                "trans_smooth": trans_smooth,
            }

        return results

    def evaluate_all_segments(self) -> pd.DataFrame:
        geo_dir = Path(self.paths["inference_data"]["oscillator"])
        pas_dir = Path(self.paths["inference_data"]["diffusion"])
        timings_dir = Path(self.paths["song_timings"])
        audio_dir = Path(self.paths["audio_parts"])

        results = []

        for geo_file in sorted(geo_dir.glob("*.pkl")):
            pas_file = pas_dir / geo_file.name
            audio_file = audio_dir / f"{geo_file.stem}.wav"

            if not pas_file.exists():
                continue

            if not audio_file.exists():
                print(f"Audio file not found: {audio_file.name}, skipping...")
                continue

            parts = geo_file.stem.split("_part_")
            if len(parts) != 2:
                continue

            song_name = parts[0]
            segment_info = parts[1]

            label_parts = segment_info.split("_")
            segment_label = label_parts[-1] if len(label_parts) >= 2 else "unknown"

            timing_file = timings_dir / f"{song_name}.json"

            if timing_file.exists():
                with open(timing_file) as f:
                    metadata = json.load(f)
                bpm = metadata["bpm"]
                beat_times = metadata.get("beats", [])
            else:
                bpm = 120.0
                beat_times = []

            try:
                segment_results = self.evaluate_segment(
                    str(geo_file),
                    str(pas_file),
                    str(audio_file),
                    bpm,
                    beat_times,
                    song_name,
                    segment_label,
                )

                for condition_name, metrics in segment_results.items():
                    results.append(
                        {
                            "song_name": song_name,
                            "segment": segment_info,
                            "condition": condition_name,
                            **metrics,
                        }
                    )

                print(f"Processed: {geo_file.stem}")

            except Exception as e:
                print(f"Error processing {geo_file.stem}: {e}")

        return pd.DataFrame(results)

    def compute_summary_statistics(self, df: pd.DataFrame) -> Dict:
        summary = {}

        for condition in df["condition"].unique():
            cond_data = df[df["condition"] == condition]
            summary[condition] = {
                "ssm_corr": {
                    "mean": cond_data["ssm_corr"].mean(),
                    "std": cond_data["ssm_corr"].std(),
                },
                "novelty_corr_raw": {
                    "mean": cond_data["novelty_corr_raw"].mean(),
                    "std": cond_data["novelty_corr_raw"].std(),
                },
                "novelty_corr": {
                    "mean": cond_data["novelty_corr"].mean(),
                    "std": cond_data["novelty_corr"].std(),
                },
                "beat_peak_align": {
                    "mean": cond_data["beat_peak_align"].mean(),
                    "std": cond_data["beat_peak_align"].std(),
                },
                "beat_valley_align": {
                    "mean": cond_data["beat_valley_align"].mean(),
                    "std": cond_data["beat_valley_align"].std(),
                },
                "trans_smooth": {
                    "mean": cond_data["trans_smooth"].mean(),
                    "std": cond_data["trans_smooth"].std(),
                },
            }

        return summary

    def export_latex_table(self, summary: Dict, output_path: Path) -> str:
        latex = """\\begin{table}[htbp]
\\centering
\\caption{Ablation and Baseline Comparison}
\\label{tab:ablation}
\\begin{tabular}{lccccc}
\\toprule
Condition & SSM Corr & Novelty & Beat Peak & Beat Valley & Smooth \\\\
\\midrule
"""

        condition_order = ["full", "diffusion_only", "oscillator_only", "retrieval"]
        display_names = {
            "full": "Full (this work)",
            "diffusion_only": "Diffusion-only",
            "oscillator_only": "Oscillator-only",
            "retrieval": "Retrieval Baseline",
        }

        for cond in condition_order:
            if cond in summary:
                row = [display_names[cond]]
                for metric in [
                    "ssm_corr",
                    "novelty_corr",
                    "beat_peak_align",
                    "beat_valley_align",
                    "trans_smooth",
                ]:
                    val = summary[cond][metric]["mean"]
                    row.append(f"{val:.2f}")
                latex += " & ".join(row) + " \\\\\n"

        latex += """\\bottomrule
\\end{tabular}
\\end{table}"""

        with open(output_path, "w") as f:
            f.write(latex)

        return latex


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ablation Evaluator for Table 3")
    parser.add_argument(
        "--config", type=str, default="../configs", help="Path to configs directory"
    )
    parser.add_argument(
        "--output", type=str, default="../outputs", help="Output directory"
    )
    args = parser.parse_args()

    configs_dir = Path(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    evaluator = AblationEvaluator(configs_dir)

    print("Starting Ablation Evaluation...")
    df = evaluator.evaluate_all_segments()

    df.to_csv(output_dir / "table3_ablation_results.csv", index=False)
    print(f"\nResults saved to {output_dir / 'table3_ablation_results.csv'}")

    summary = evaluator.compute_summary_statistics(df)

    print("\n" + "=" * 60)
    print("TABLE 3: Ablation and Baseline Comparison")
    print("=" * 60)

    condition_names = {
        "full": "Full (this work)",
        "diffusion_only": "Diffusion-only",
        "oscillator_only": "Oscillator-only",
        "retrieval": "Retrieval Baseline",
    }

    for cond in ["full", "diffusion_only", "oscillator_only", "retrieval"]:
        if cond in summary:
            print(f"\n{condition_names[cond]}:")
            for metric, values in summary[cond].items():
                print(f"  {metric}: {values['mean']:.3f} +/- {values['std']:.3f}")

    latex = evaluator.export_latex_table(summary, output_dir / "table3_latex.tex")
    print(f"\nLaTeX table saved to {output_dir / 'table3_latex.tex'}")
