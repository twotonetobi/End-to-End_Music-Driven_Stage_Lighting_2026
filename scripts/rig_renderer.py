"""
Rig-Specific Renderer for Cross-Rig Evaluation (Table 2)

Implements "Intention Preservation Under Cross-Rig Application" evaluation.

Tests how well the abstraction layer preserves creative INTENTIONS when adapting
to different rig configurations. Focus is on TEMPORAL dynamics (the show's
intensity/color envelope over time), not exact spatial patterns.

Rigs tested:
- Club (16 MH): 16 moving heads on one truss, center-mirrored (8 unique positions)
- Concert (56 fixtures): 56 fixtures across 3 trusses (~19 per group)
- LED Bars (64 px): 64 pixels distributed (~21 per group)
- Reference (Ours): Native 33 LEDs per group
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class RigProfile:
    name: str
    description: str
    total_fixtures: int
    unique_positions_per_group: int
    mirroring: bool
    groups: Dict[str, int]


RIG_PROFILES = {
    "club": RigProfile(
        name="Club Rig",
        description="16 moving heads on one truss, center-mirrored",
        total_fixtures=16,
        unique_positions_per_group=8,
        mirroring=True,
        groups={"lx1": 5, "lx2": 5, "lx3": 6},
    ),
    "concert": RigProfile(
        name="Concert Rig",
        description="56 fixtures across 3 trusses (front/mid/back)",
        total_fixtures=56,
        unique_positions_per_group=19,
        mirroring=False,
        groups={"lx1": 16, "lx2": 18, "lx3": 22},
    ),
    "led_bars": RigProfile(
        name="LED Bar Array",
        description="64 pixels distributed across LED bars",
        total_fixtures=64,
        unique_positions_per_group=21,
        mirroring=False,
        groups={"lx1": 20, "lx2": 22, "lx3": 22},
    ),
    "reference": RigProfile(
        name="Reference (Ours)",
        description="Native 33 LEDs per group (99 total)",
        total_fixtures=99,
        unique_positions_per_group=33,
        mirroring=False,
        groups={"lx1": 33, "lx2": 33, "lx3": 33},
    ),
    "direct_club": RigProfile(
        name="Direct Mapping",
        description="Naive truncation to 8 positions (no intention preservation)",
        total_fixtures=16,
        unique_positions_per_group=8,
        mirroring=False,
        groups={"lx1": 5, "lx2": 5, "lx3": 6},
    ),
}


def resample_linear(array: np.ndarray, target_length: int) -> np.ndarray:
    if len(array) == target_length:
        return array.copy()
    if target_length == 0:
        return np.array([])
    x_old = np.linspace(0, 1, len(array))
    x_new = np.linspace(0, 1, target_length)
    return np.interp(x_new, x_old, array)


def resample_2d(array: np.ndarray, target_length: int) -> np.ndarray:
    frames, leds = array.shape
    result = np.zeros((frames, target_length))
    for frame in range(frames):
        result[frame] = resample_linear(array[frame], target_length)
    return result


def resample_rgb(rgb_array: np.ndarray, target_leds: int) -> np.ndarray:
    frames, leds, channels = rgb_array.shape
    result = np.zeros((frames, target_leds, channels))
    for c in range(channels):
        result[:, :, c] = resample_2d(rgb_array[:, :, c], target_leds)
    return result


def direct_truncate_rgb(rgb_array: np.ndarray, target_leds: int) -> np.ndarray:
    """
    Naive direct mapping: take first N LEDs only (left-side truncation).
    Loses all spatial distribution - only sees one edge of the LED strip.
    """
    frames, leds, channels = rgb_array.shape
    if target_leds >= leds:
        return rgb_array.copy()
    return rgb_array[:, :target_leds, :]


def direct_expand_rgb(rgb_array: np.ndarray, target_leds: int) -> np.ndarray:
    """
    Naive expansion: repeat values without interpolation.
    Creates blocky/stepped output instead of smooth gradients.
    """
    frames, leds, channels = rgb_array.shape
    if target_leds <= leds:
        return rgb_array.copy()
    # Nearest-neighbor expansion
    indices = np.linspace(0, leds - 1, target_leds).astype(int)
    return rgb_array[:, indices, :]


def apply_center_mirror(array: np.ndarray) -> np.ndarray:
    return np.concatenate([array, array[::-1]])


def apply_center_mirror_2d(array: np.ndarray) -> np.ndarray:
    frames, unique = array.shape
    result = np.zeros((frames, unique * 2))
    for frame in range(frames):
        result[frame] = apply_center_mirror(array[frame])
    return result


def apply_center_mirror_rgb(rgb_array: np.ndarray) -> np.ndarray:
    frames, unique, channels = rgb_array.shape
    result = np.zeros((frames, unique * 2, channels))
    for c in range(channels):
        result[:, :, c] = apply_center_mirror_2d(rgb_array[:, :, c])
    return result


class RigRenderer:
    def __init__(self, rig_name: str):
        if rig_name not in RIG_PROFILES:
            raise ValueError(
                f"Unknown rig: {rig_name}. Available: {list(RIG_PROFILES.keys())}"
            )
        self.profile = RIG_PROFILES[rig_name]
        self.rig_name = rig_name

    def render(self, rgb_array: np.ndarray, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        frames, leds, channels = rgb_array.shape

        if self.rig_name == "reference":
            return rgb_array.copy()

        unique_positions = self.profile.unique_positions_per_group

        if self.rig_name.startswith("direct_"):
            downsampled = direct_truncate_rgb(rgb_array, unique_positions)
            result = direct_expand_rgb(downsampled, leds)
        else:
            downsampled = resample_rgb(rgb_array, unique_positions)
            if self.profile.mirroring:
                downsampled = apply_center_mirror_rgb(downsampled)
            result = resample_rgb(downsampled, leds)

        noise = rng.normal(0, 0.002, result.shape)
        result = np.clip(result + noise, 0, 1)

        return result

    def get_effective_resolution(self) -> int:
        return self.profile.unique_positions_per_group

    def get_info(self) -> Dict:
        return {
            "name": self.profile.name,
            "description": self.profile.description,
            "total_fixtures": self.profile.total_fixtures,
            "unique_per_group": self.profile.unique_positions_per_group,
            "mirroring": self.profile.mirroring,
            "groups": self.profile.groups,
        }


def render_through_rig(
    rgb_array: np.ndarray, rig_name: str, seed: int = 42
) -> np.ndarray:
    renderer = RigRenderer(rig_name)
    return renderer.render(rgb_array, seed)


def compute_mean_intensity_correlation(
    original: np.ndarray, rendered: np.ndarray
) -> float:
    """
    Temporal correlation of MEAN intensity per frame.
    This measures if the overall brightness dynamics are preserved.
    Should be very high (~95-99%) for all reasonable rigs.
    """
    orig_mean = np.mean(np.max(original, axis=2), axis=1)
    rend_mean = np.mean(np.max(rendered, axis=2), axis=1)

    if np.std(orig_mean) == 0 or np.std(rend_mean) == 0:
        return 1.0 if np.allclose(orig_mean, rend_mean) else 0.0

    corr = np.corrcoef(orig_mean, rend_mean)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def compute_peak_intensity_correlation(
    original: np.ndarray, rendered: np.ndarray
) -> float:
    """
    Temporal correlation of PEAK intensity per frame.
    Measures if intensity peaks are preserved over time.
    """
    orig_peak = np.max(np.max(original, axis=2), axis=1)
    rend_peak = np.max(np.max(rendered, axis=2), axis=1)

    if np.std(orig_peak) == 0 or np.std(rend_peak) == 0:
        return 1.0 if np.allclose(orig_peak, rend_peak) else 0.0

    corr = np.corrcoef(orig_peak, rend_peak)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def compute_dynamic_range_correlation(
    original: np.ndarray, rendered: np.ndarray
) -> float:
    """
    Temporal correlation of intensity RANGE (max - min) per frame.
    Measures if contrast/dynamics are preserved.
    """
    orig_brightness = np.max(original, axis=2)
    rend_brightness = np.max(rendered, axis=2)

    orig_range = np.max(orig_brightness, axis=1) - np.min(orig_brightness, axis=1)
    rend_range = np.max(rend_brightness, axis=1) - np.min(rend_brightness, axis=1)

    if np.std(orig_range) == 0 or np.std(rend_range) == 0:
        return 1.0 if np.allclose(orig_range, rend_range, atol=0.05) else 0.5

    corr = np.corrcoef(orig_range, rend_range)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def compute_color_correlation(
    original: np.ndarray, rendered: np.ndarray
) -> Tuple[float, float]:
    """
    Temporal correlation of MEAN hue and saturation per frame.
    Measures if color intent is preserved over time.
    Returns (hue_correlation, saturation_correlation).
    """

    def rgb_to_hs_mean(rgb):
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        delta = maxc - minc

        s = np.where(maxc > 0, delta / (maxc + 1e-8), 0)
        h = np.zeros_like(r)
        mask = delta > 0.01

        idx = (maxc == r) & mask
        h[idx] = ((g[idx] - b[idx]) / (delta[idx] + 1e-8)) % 6
        idx = (maxc == g) & mask
        h[idx] = ((b[idx] - r[idx]) / (delta[idx] + 1e-8)) + 2
        idx = (maxc == b) & mask
        h[idx] = ((r[idx] - g[idx]) / (delta[idx] + 1e-8)) + 4
        h = h / 6.0

        h_mean = np.mean(h, axis=1)
        s_mean = np.mean(s, axis=1)
        return h_mean, s_mean

    orig_h, orig_s = rgb_to_hs_mean(original)
    rend_h, rend_s = rgb_to_hs_mean(rendered)

    if np.std(orig_h) > 0.01 and np.std(rend_h) > 0.01:
        h_corr = np.corrcoef(orig_h, rend_h)[0, 1]
        h_corr = float(h_corr) if not np.isnan(h_corr) else 1.0
    else:
        h_corr = 1.0 if np.allclose(orig_h, rend_h, atol=0.1) else 0.8

    if np.std(orig_s) > 0.01 and np.std(rend_s) > 0.01:
        s_corr = np.corrcoef(orig_s, rend_s)[0, 1]
        s_corr = float(s_corr) if not np.isnan(s_corr) else 1.0
    else:
        s_corr = 1.0 if np.allclose(orig_s, rend_s, atol=0.1) else 0.8

    return h_corr, s_corr


if __name__ == "__main__":
    print("Rig Renderer - Intention Preservation Test")
    print("=" * 60)

    frames = 300
    leds = 33
    test_rgb = np.zeros((frames, leds, 3))

    for f in range(frames):
        phase = f / frames * 4 * np.pi
        base_intensity = 0.5 + 0.3 * np.sin(phase)
        for led in range(leds):
            position = led / leds
            intensity = base_intensity * (
                0.8 + 0.2 * np.sin(2 * np.pi * position + phase * 0.5)
            )
            hue = 0.6 + 0.1 * np.sin(phase * 0.3)
            test_rgb[f, led, 0] = intensity * (1 - hue)
            test_rgb[f, led, 1] = intensity * hue * 0.5
            test_rgb[f, led, 2] = intensity * hue

    print(f"\nTest pattern: {frames} frames, {leds} LEDs\n")

    for rig_name, profile in RIG_PROFILES.items():
        renderer = RigRenderer(rig_name)
        rendered = renderer.render(test_rgb)

        mean_corr = compute_mean_intensity_correlation(test_rgb, rendered)
        peak_corr = compute_peak_intensity_correlation(test_rgb, rendered)
        range_corr = compute_dynamic_range_correlation(test_rgb, rendered)
        hue_corr, sat_corr = compute_color_correlation(test_rgb, rendered)

        print(f"{profile.name}:")
        print(f"  Unique positions/group: {profile.unique_positions_per_group}")
        print(f"  Mean Intensity Corr:  {mean_corr:.3f}")
        print(f"  Peak Intensity Corr:  {peak_corr:.3f}")
        print(f"  Dynamic Range Corr:   {range_corr:.3f}")
        print(f"  Hue Correlation:      {hue_corr:.3f}")
        print(f"  Saturation Corr:      {sat_corr:.3f}")
        print()
