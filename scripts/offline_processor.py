"""
Offline Back-Converter for Paper Evaluation Pipeline

Ports the TouchDesigner back-converter logic to standalone Python
for offline processing of inference pickle files.

Data Layout (Oscillator Model - 60 params):
    [0:10]   lx1_standard
    [10:20]  lx1_highlight (NOT used)
    [20:30]  lx2_standard
    [30:40]  lx2_highlight (NOT used)
    [40:50]  lx3_standard
    [50:60]  lx3_highlight (NOT used)

GeoApproach Features (per 10-param block):
    0: pan_activity
    1: tilt_activity
    2: wave_type_a
    3: wave_type_b
    4: freq
    5: amplitude
    6: offset
    7: phase
    8: col_hue
    9: col_sat
"""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.signal import find_peaks


@dataclass
class ProcessorConfig:
    led_count: int = 33
    virtual_led_count: int = 8
    fps: int = 30
    max_cycles_per_second: float = 4.0
    max_phase_cycles_per_second: float = 8.0

    oscillation_threshold: float = 10
    geo_phase_threshold: float = 0.15
    geo_freq_threshold: float = 0.15
    geo_offset_threshold: float = 0.15

    bpm_low: int = 80
    bpm_high: int = 135

    decision_boundary_still: float = 0.1
    decision_boundary_sine: float = 0.3
    decision_boundary_pwm_basic: float = 0.5
    decision_boundary_pwm_extended: float = 0.7
    decision_boundary_odd_even: float = 0.9


class OfflineProcessor:
    def __init__(self, config: Optional[ProcessorConfig] = None):
        self.config = config or ProcessorConfig()

    def load_pickle(self, path: str) -> np.ndarray:
        with open(path, "rb") as f:
            return pickle.load(f)

    def load_song_metadata(self, json_path: str) -> Dict:
        with open(json_path, "r") as f:
            return json.load(f)

    def extract_group_params(
        self,
        geo_data: np.ndarray,
        pas_data: np.ndarray,
        lx_num: str,
        pas_group_idx: int = 0,
    ) -> Dict[str, Any]:
        """Extract parameters for a single luminaire group."""

        lx_offsets = {"lx1": 0, "lx2": 20, "lx3": 40}
        offset = lx_offsets[lx_num]

        standard_data = geo_data[:, offset : offset + 10]

        pas_start = pas_group_idx * 6
        pas_features = pas_data[:, pas_start : pas_start + 6]

        return {
            "pan_activity": standard_data[:, 0],
            "tilt_activity": standard_data[:, 1],
            "wave_type_a": standard_data[:, 2],
            "wave_type_b": standard_data[:, 3],
            "freq": standard_data[:, 4],
            "amplitude": standard_data[:, 5],
            "offset": standard_data[:, 6],
            "phase": standard_data[:, 7],
            "col_hue": standard_data[:, 8],
            "col_sat": standard_data[:, 9],
            "pas_intensity_peak": pas_features[:, 0],
            "pas_spatial_gradient": pas_features[:, 1],
            "pas_peak_density": pas_features[:, 2],
            "pas_intensity_min_inv": pas_features[:, 3],
            "pas_hue": pas_features[:, 4],
            "pas_sat": pas_features[:, 5],
            "frames": geo_data.shape[0],
        }

    def select_waveform(self, params: Dict, bpm: float) -> Tuple[str, float]:
        """Decision function for wave type selection."""

        intensity_peak = params["pas_intensity_peak"]
        intensity_min_inv = params["pas_intensity_min_inv"]

        peaks, _ = find_peaks(intensity_peak, height=0.6)
        oscillation_count = len(peaks)

        target_max = np.max(intensity_peak)
        target_min = 1.0 - np.max(intensity_min_inv)
        intensity_range = (
            max(0, target_max - target_min) if target_min <= target_max else target_max
        )

        pas_dynamic_score = oscillation_count / self.config.oscillation_threshold

        phase_geo = params["phase"]
        freq_geo = params["freq"]
        offset_geo = params["offset"]

        geo_phase_range = np.max(phase_geo) - np.min(phase_geo)
        geo_freq_range = np.max(freq_geo) - np.min(freq_geo)
        geo_offset_range = np.max(offset_geo) - np.min(offset_geo)

        geo_phase_norm = geo_phase_range / self.config.geo_phase_threshold
        geo_freq_norm = geo_freq_range / self.config.geo_freq_threshold
        geo_offset_norm = geo_offset_range / self.config.geo_offset_threshold

        overall_geo_dynamic = (geo_phase_norm + geo_freq_norm + geo_offset_norm) / 3.0
        overall_dynamic = (overall_geo_dynamic + pas_dynamic_score) / 2.0

        if intensity_range < self.config.decision_boundary_still:
            decision = "still"
        elif overall_dynamic < self.config.decision_boundary_sine:
            decision = "sine"
        elif overall_dynamic < self.config.decision_boundary_pwm_basic:
            decision = "pwm_basic"
        elif overall_dynamic < self.config.decision_boundary_pwm_extended:
            decision = "pwm_extended"
        elif overall_dynamic < self.config.decision_boundary_odd_even:
            decision = "odd_even"
        else:
            if bpm > self.config.bpm_high:
                decision = "square"
            else:
                decision = "random"

        return decision, overall_dynamic

    def _sine_basis(
        self, x: np.ndarray, phase_offset: float, freq_adj: float
    ) -> np.ndarray:
        peaks = (self.config.virtual_led_count - 4) / 2
        return 0.5 + 0.5 * np.sin(2 * np.pi * (freq_adj * peaks * x + phase_offset))

    def _square_basis(
        self, x: np.ndarray, phase_offset: float, freq_adj: float, divider: int = 32
    ) -> np.ndarray:
        squares_per_strip = self.config.virtual_led_count / divider
        return (
            np.mod(freq_adj * squares_per_strip * x + phase_offset, 1) < 0.5
        ).astype(float)

    def _still_basis(self, x: np.ndarray) -> np.ndarray:
        return np.full_like(x, 0.75)

    def _odd_even(self, frame: int, bpm: float) -> np.ndarray:
        frames_per_beat = self.config.fps * 60.0 / bpm
        half_beat_frames = frames_per_beat
        beat_index = int(frame / half_beat_frames)

        indices = np.arange(self.config.virtual_led_count)
        if beat_index % 2 == 0:
            virtual_pattern = (indices % 2).astype(float)
        else:
            virtual_pattern = ((indices + 1) % 2).astype(float)

        return self._map_virtual_to_physical(virtual_pattern)

    def _random_pattern(
        self, frame: int, bpm: float, update_modifier: int = 1
    ) -> np.ndarray:
        frames_per_beat = self.config.fps * 60.0 / bpm
        quarter_beat_frames = frames_per_beat / 4.0
        update_interval = int(quarter_beat_frames * update_modifier)
        update_index = frame // max(1, update_interval)

        rng = np.random.default_rng(update_index)
        virtual_pattern = rng.random(self.config.virtual_led_count)

        num_to_zero = int(np.ceil(0.3 * self.config.virtual_led_count))
        indices_to_zero = rng.choice(
            self.config.virtual_led_count, num_to_zero, replace=False
        )
        virtual_pattern[indices_to_zero] = 0.0

        return self._map_virtual_to_physical(virtual_pattern)

    def _pwm_basic(
        self, x: np.ndarray, phase_offset: float, freq_adj: float, bpm: float
    ) -> np.ndarray:
        frames_per_beat = (60 / bpm * 2) * self.config.fps
        frames_per_note = frames_per_beat / 4

        current_frame = int(phase_offset * 2 / (2 * np.pi) * self.config.fps)
        beat_position = (current_frame / frames_per_note) * (2 * np.pi)

        virtual_positions = np.linspace(0, 1, self.config.virtual_led_count)
        active_leds = int(self.config.virtual_led_count / 2)

        positions = np.mod(virtual_positions + beat_position / (2 * np.pi), 1)
        sorted_positions = np.sort(positions)
        threshold = sorted_positions[self.config.virtual_led_count - active_leds]

        virtual_pattern = (positions >= threshold).astype(float)
        return self._map_virtual_to_physical(virtual_pattern)

    def _pwm_extended(
        self, x: np.ndarray, phase_offset: float, freq_adj: float
    ) -> np.ndarray:
        virtual_positions = np.linspace(0, 1, self.config.virtual_led_count)
        indices = np.arange(self.config.virtual_led_count)

        base = (
            np.mod(
                freq_adj * self.config.max_cycles_per_second * virtual_positions
                + phase_offset,
                1,
            )
            < 0.5
        ).astype(float)
        odd_even_pattern = (indices % 2).astype(float)
        virtual_pattern = 0.5 * base + 0.5 * odd_even_pattern

        return self._map_virtual_to_physical(virtual_pattern)

    def _map_virtual_to_physical(self, virtual_pattern: np.ndarray) -> np.ndarray:
        n = self.config.led_count
        virtual_led_count = len(virtual_pattern)
        physical_pattern = np.zeros(n)

        for i in range(n):
            virtual_index = int(i * virtual_led_count / n)
            if virtual_index >= virtual_led_count:
                virtual_index = virtual_led_count - 1
            physical_pattern[i] = virtual_pattern[virtual_index]

        return physical_pattern

    def generate_rgb_array(
        self,
        params: Dict,
        decision: str,
        overall_dynamic: float,
        bpm: float,
        mirroring: bool = False,
    ) -> np.ndarray:
        """Generate RGB array for a segment."""

        frames = params["frames"]
        led_count = self.config.led_count

        mean_freq = np.mean(params["freq"])
        mean_phase = np.mean(params["phase"])

        if bpm < self.config.bpm_low:
            bpm_scale = 0.5
        elif bpm > self.config.bpm_high:
            bpm_scale = 2.0
        else:
            bpm_scale = 1.0

        f0 = mean_freq * bpm_scale
        freq_adj = f0 * overall_dynamic
        phase_movement = mean_phase * self.config.max_phase_cycles_per_second / frames

        col_hue = np.mean(params["pas_hue"])
        col_sat = np.mean(params["pas_sat"])

        waveform = np.zeros((frames, led_count))
        x = np.linspace(0, 1, led_count)

        for frame in range(frames):
            current_phase = phase_movement * frame

            if decision == "still":
                waveform[frame, :] = self._still_basis(x)
            elif decision == "sine":
                waveform[frame, :] = self._sine_basis(x, current_phase, freq_adj)
            elif decision == "square":
                waveform[frame, :] = self._square_basis(x, current_phase, freq_adj)
            elif decision == "odd_even":
                waveform[frame, :] = self._odd_even(frame, bpm)
            elif decision == "random":
                waveform[frame, :] = self._random_pattern(frame, bpm)
            elif decision == "pwm_basic":
                waveform[frame, :] = self._pwm_basic(x, current_phase, freq_adj, bpm)
            elif decision == "pwm_extended":
                waveform[frame, :] = self._pwm_extended(x, current_phase, freq_adj)
            else:
                waveform[frame, :] = self._sine_basis(x, current_phase, freq_adj)

        if mirroring:
            full_waveform = np.zeros((frames, led_count))
            for frame in range(frames):
                middle = (led_count + 1) // 2
                left_side = waveform[frame, :middle]
                if led_count % 2 == 0:
                    right_side = left_side[::-1]
                else:
                    right_side = left_side[:-1][::-1]
                full_waveform[frame, :] = np.concatenate([left_side, right_side])
            waveform = full_waveform

        rgb_array = np.zeros((frames, led_count, 3))

        for frame in range(frames):
            v = waveform[frame, :]
            c = v * col_sat
            h_prime = col_hue * 6.0
            x_color = c * (1 - np.abs(np.mod(h_prime, 2) - 1))
            m = v - c

            rgb = np.zeros((led_count, 3))

            if h_prime < 1:
                rgb = np.column_stack((c, x_color, np.zeros(led_count)))
            elif h_prime < 2:
                rgb = np.column_stack((x_color, c, np.zeros(led_count)))
            elif h_prime < 3:
                rgb = np.column_stack((np.zeros(led_count), c, x_color))
            elif h_prime < 4:
                rgb = np.column_stack((np.zeros(led_count), x_color, c))
            elif h_prime < 5:
                rgb = np.column_stack((x_color, np.zeros(led_count), c))
            else:
                rgb = np.column_stack((c, np.zeros(led_count), x_color))

            rgb += m[:, np.newaxis]
            rgb_array[frame] = np.clip(rgb, 0, 1)

        return rgb_array

    def process_segment(
        self,
        geo_path: str,
        pas_path: str,
        bpm: float,
        mirroring: Dict[str, bool] = None,
    ) -> Dict[str, np.ndarray]:
        """Process a single segment and return RGB arrays for all groups."""

        geo_data = self.load_pickle(geo_path)[:, :60]
        pas_data = self.load_pickle(pas_path)[:, :60]

        if mirroring is None:
            mirroring = {"lx1": False, "lx2": False, "lx3": False}

        results = {}

        for lx_idx, lx_num in enumerate(["lx1", "lx2", "lx3"]):
            params = self.extract_group_params(
                geo_data, pas_data, lx_num, pas_group_idx=lx_idx
            )
            decision, overall_dynamic = self.select_waveform(params, bpm)

            rgb_array = self.generate_rgb_array(
                params, decision, overall_dynamic, bpm, mirroring.get(lx_num, False)
            )

            results[lx_num] = {
                "rgb": rgb_array,
                "decision": decision,
                "overall_dynamic": overall_dynamic,
                "params": params,
            }

        return results


def extract_pas_features(rgb_array: np.ndarray) -> np.ndarray:
    """
    Re-extract PASv01-like features from RGB array.

    Args:
        rgb_array: Shape (frames, leds, 3)

    Returns:
        features: Shape (frames, 6)
    """
    frames, leds, _ = rgb_array.shape
    features = np.zeros((frames, 6))

    brightness = np.max(rgb_array, axis=2)

    features[:, 0] = np.max(brightness, axis=1)

    gradients = np.abs(np.diff(brightness, axis=1))
    features[:, 1] = np.mean(gradients, axis=1)

    for frame in range(frames):
        peaks, _ = find_peaks(brightness[frame], height=0.3)
        features[frame, 2] = len(peaks) / leds

    features[:, 3] = 1.0 - np.min(brightness, axis=1)

    for frame in range(frames):
        r, g, b = rgb_array[frame, :, 0], rgb_array[frame, :, 1], rgb_array[frame, :, 2]
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        delta = max_rgb - min_rgb

        hue = np.zeros(leds)
        mask = delta > 0

        r_max = (max_rgb == r) & mask
        g_max = (max_rgb == g) & mask
        b_max = (max_rgb == b) & mask

        hue[r_max] = ((g[r_max] - b[r_max]) / delta[r_max]) % 6
        hue[g_max] = ((b[g_max] - r[g_max]) / delta[g_max]) + 2
        hue[b_max] = ((r[b_max] - g[b_max]) / delta[b_max]) + 4
        hue = hue / 6.0

        features[frame, 4] = np.mean(hue[mask]) if np.any(mask) else 0.0

        sat = np.zeros(leds)
        sat[max_rgb > 0] = delta[max_rgb > 0] / max_rgb[max_rgb > 0]
        features[frame, 5] = np.mean(sat)

    return features


if __name__ == "__main__":
    import yaml

    config_path = Path(__file__).parent.parent / "configs" / "paths.yaml"
    with open(config_path) as f:
        paths = yaml.safe_load(f)

    processor = OfflineProcessor()

    geo_dir = Path(paths["inference_data"]["oscillator"])
    pas_dir = Path(paths["inference_data"]["diffusion"])
    timings_dir = Path(paths["song_timings"])

    sample_geo = list(geo_dir.glob("*.pkl"))[0]
    sample_pas = pas_dir / sample_geo.name

    song_name = "_".join(sample_geo.stem.split("_")[:-2])
    timing_file = timings_dir / f"{song_name}.json"

    if timing_file.exists():
        metadata = processor.load_song_metadata(str(timing_file))
        bpm = metadata["bpm"]
    else:
        bpm = 120.0

    print(f"Processing: {sample_geo.name}")
    print(f"BPM: {bpm}")

    results = processor.process_segment(str(sample_geo), str(sample_pas), bpm)

    for lx_num, data in results.items():
        print(f"\n{lx_num}:")
        print(f"  Decision: {data['decision']}")
        print(f"  Dynamic: {data['overall_dynamic']:.3f}")
        print(f"  RGB shape: {data['rgb'].shape}")
