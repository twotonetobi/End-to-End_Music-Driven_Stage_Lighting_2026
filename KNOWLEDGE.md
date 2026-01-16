# Technical Knowledge Base - Paper Evaluation Pipeline

> **Last Updated:** 2026-01-14
> **Purpose:** Technical reference for the PETRA 2025 paper evaluation pipeline

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Table 2: Cross-Rig Evaluation](#table-2-cross-rig-evaluation)
3. [Table 3: Ablation Study](#table-3-ablation-study)
4. [Metric Implementations](#metric-implementations)
5. [Data Pipeline](#data-pipeline)
6. [Key Implementation Details](#key-implementation-details)

---

## System Overview

### Purpose

This repository contains **offline evaluation scripts** for the PETRA 2025 paper "End-to-End Music-Driven Stage Lighting: A Co-Creative Framework". The goal is to validate:

1. **Table 2 (Cross-Rig)**: Validate intention preservation across fixture configurations
2. **Table 3 (Ablation)**: Isolate contributions of Diffusion (PAS) and Oscillator (GEO) branches

### Architecture

```
                    AUDIO INPUT
                         │
           ┌─────────────┴─────────────┐
           ▼                           ▼
┌─────────────────────┐     ┌─────────────────────┐
│   DIFFUSION MODEL   │     │   OSCILLATOR MODEL  │
│   (PAS - Intensity) │     │   (GEO - Waveform)  │
│   60D per frame     │     │   60D per frame     │
└─────────────────────┘     └─────────────────────┘
           │                           │
           └───────────┬───────────────┘
                       ▼
              ┌────────────────┐
              │   INTERPRETER  │
              │  (RGB Output)  │
              └────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌──────────────────┐     ┌──────────────────┐
│   TABLE 2:       │     │   TABLE 3:       │
│   Cross-Rig      │     │   Ablation       │
│   Evaluation     │     │   Study          │
└──────────────────┘     └──────────────────┘
```

---

## Table 2: Cross-Rig Evaluation

### Purpose

Tests how well **temporal dynamics** (intensity envelope, color changes over time) survive when adapting lighting shows to different rig configurations.

### Key Insight

When adapting a show from 33 LEDs/group to smaller rigs:
- **Spatial patterns** inevitably degrade (fewer fixtures = less resolution)
- **Temporal dynamics** can be preserved with intelligent resampling

### Approach: Intention-Preserving Resampling

Our approach uses **linear interpolation** across the full LED strip:

```python
def resample_rgb(rgb_array, target_leds):
    # Resample using linear interpolation
    # Preserves weighted average of original pattern
    x_old = np.linspace(0, 1, original_leds)
    x_new = np.linspace(0, 1, target_leds)
    return np.interp(x_new, x_old, rgb_array)
```

### Baseline: Direct Mapping (Naive Truncation)

The baseline uses **naive truncation** (first N LEDs only):

```python
def direct_truncate_rgb(rgb_array, target_leds):
    # Take first N LEDs only - loses ~75% of spatial data
    return rgb_array[:, :target_leds, :]
```

### Why the Baseline Performs Poorly

| Factor | Explanation |
|--------|-------------|
| **Single-edge sampling** | Taking only first 8 LEDs misses patterns on LEDs 9-33 |
| **No spatial interpolation** | Lost information is irrecoverable |
| **No weighted averaging** | Doesn't preserve overall brightness/color |

### Rig Configurations

| Rig | Total Fixtures | Unique/Group | Method |
|-----|---------------|--------------|--------|
| **Direct Mapping** | 8 | 8 | Naive truncation (baseline) |
| **Club** | 16 MH | 8 | Linear resample + mirror |
| **Concert** | 56 | 19 | Linear resample |
| **LED Bars** | 64 px | 21 | Linear resample |
| **Reference** | 99 | 33 | Native (no resampling) |

### Temporal Metrics

| Metric | What It Measures |
|--------|------------------|
| **Mean Intensity** | Correlation of average brightness per frame |
| **Peak Intensity** | Correlation of maximum brightness per frame |
| **Dynamic Range** | Correlation of intensity spread (max - min) per frame |
| **Hue** | Correlation of mean hue per frame |
| **Saturation** | Correlation of mean saturation per frame |

---

## Table 3: Ablation Study

### Purpose

Isolate contributions of the Diffusion (PAS) and Oscillator (GEO) branches by comparing against an RMS baseline.

### Baseline: RMS-Reactive Lighting

The baseline represents a **naive sound-to-light** approach:

```python
def create_rms_baseline(audio_path, frames):
    y, sr = librosa.load(audio_path, sr=22050)
    rms = librosa.feature.rms(y=y, hop_length=735)[0]  # 30 fps
    rms_norm = normalize(rms)
    
    # Map RMS directly to intensity
    pas = zeros((frames, 60))
    for group in range(10):
        pas[:, group*6 + 0] = rms_norm  # I_peak = RMS
        pas[:, group*6 + 3] = rms_norm  # I_min_inv = RMS
    return pas
```

### Why Oscillator-only Beats Full on Beat Alignment

This is **expected and correct**:

1. **GEO is specifically trained for beat-synchronized patterns.** It generates phase-locked waveforms.

2. **When combined with PAS (Full), intensity modulation reduces beat clarity.** The diffusion model adds artistic variation that softens hard transitions.

3. **This is a trade-off, not a failure.** Professional lighting uses subtle dynamics rather than hard beat-syncing.

4. **The combination (Full) achieves best overall structure** despite lower beat sharpness.

### Ablation Conditions

| Condition | GEO (Oscillator) | PAS (Diffusion) | RGB Generation |
|-----------|------------------|-----------------|----------------|
| **Full** | Trained model | Trained model | Combined effect |
| **Diffusion-only** | Constant (neutral) | Trained model | PAS-driven intensity |
| **Oscillator-only** | Trained model | Constant (neutral) | GEO-driven waveform |
| **RMS Baseline** | Static | RMS-reactive | brightness = RMS |

---

## Metric Implementations

### Table 2: Temporal Correlation Metrics

```python
def compute_mean_intensity_correlation(original, rendered):
    orig_mean = np.mean(np.max(original, axis=2), axis=1)  # Mean brightness per frame
    rend_mean = np.mean(np.max(rendered, axis=2), axis=1)
    return np.corrcoef(orig_mean, rend_mean)[0, 1]

def compute_peak_intensity_correlation(original, rendered):
    orig_peak = np.max(np.max(original, axis=2), axis=1)  # Peak brightness per frame
    rend_peak = np.max(np.max(rendered, axis=2), axis=1)
    return np.corrcoef(orig_peak, rend_peak)[0, 1]

def compute_dynamic_range_correlation(original, rendered):
    orig_range = np.max(brightness, axis=1) - np.min(brightness, axis=1)
    rend_range = np.max(brightness, axis=1) - np.min(brightness, axis=1)
    return np.corrcoef(orig_range, rend_range)[0, 1]
```

### Table 3: SSM Correlation (Structural Correspondence)

```python
def ssm_correlation_masked(audio_features, light_features, margin=3):
    # 1. Smooth and downsample
    audio_ds = downsample(smooth(audio_features, L=81), H=10)
    light_ds = downsample(smooth(light_features, L=81), H=10)
    
    # 2. Compute SSMs
    audio_ssm = compute_ssm(audio_ds)  # S(i,j) = 1 - ||x_i - x_j||/sqrt(d)
    light_ssm = compute_ssm(light_ds)
    
    # 3. Mask diagonal band
    mask = np.abs(np.arange(n)[:, None] - np.arange(n)) > margin
    
    # 4. Pearson correlation on masked values
    return pearsonr(audio_ssm[mask], light_ssm[mask])[0]
```

### Table 3: Novelty Correlation (Transition Detection)

```python
def novelty_correlation(audio_features, light_features):
    # Gaussian checkerboard kernel novelty
    audio_nov = compute_novelty(audio_ssm, L_kernel=31)
    light_nov = compute_novelty(light_ssm, L_kernel=31)
    
    return pearsonr(audio_nov, light_nov)[0]
```

### Table 3: Beat Alignment

```python
def compute_beat_alignment(rgb, beat_times, sigma=0.5):
    brightness = np.mean(rgb, axis=(1, 2))
    peaks, _ = find_peaks(brightness, distance=16)
    
    score = 0
    for peak in peaks:
        min_dist = min(abs(peak/30 - beat) for beat in beat_times)
        score += exp(-min_dist**2 / (2 * sigma**2))
    
    return score / len(peaks) if peaks else 0
```

---

## Data Pipeline

### Input Data

```
InferenceSet_ConformerModelNoRF_2024-11_seed150/  # PAS (diffusion)
  └── Song_Name_part_01_intro.pkl  # Shape: (frames, 61)

InferenceSet_ConforModelNoRFSet_2025-01_seed150/  # GEO (oscillator)
  └── Song_Name_part_01_intro.pkl  # Shape: (frames, 61)

Audio_90s_Inference_Set_parts/
  └── Song_Name_part_01_intro.wav

Audio_90s_Inference_Set_SongTimings/
  └── Song_Name.json  # BPM, beats, segments
```

### Feature Layout

**PAS (Diffusion) - 60D = 10 groups x 6 features:**
```
Group i: pas[:, i*6:(i+1)*6]
  [0] I_peak       - Peak intensity
  [1] nabla_s_I    - Spatial gradient
  [2] rho_peak     - Peak density
  [3] I_min_inv    - Contrast (1 - min)
  [4] H_bar        - Mean hue
  [5] S_bar        - Mean saturation
```

**GEO (Oscillator) - 60D = 6 LX groups x 10 features:**
```
LX group i: geo[:, i*10:(i+1)*10]
  [0] pan_activity
  [1] tilt_activity
  [2] wave_type_a
  [3] wave_type_b
  [4] freq
  [5] amplitude
  [6] offset
  [7] phase
  [8] col_hue
  [9] col_sat
```

---

## Key Implementation Details

### RGB Generation

```python
def generate_rgb_array(params, decision, bpm):
    frames = params['pas_intensity_peak'].shape[0]
    
    # Select waveform based on dynamics
    if decision == 'sine':
        wave = 0.5 + 0.5 * sin(2*pi * freq * t + phase)
    elif decision == 'odd_even':
        beat_idx = int(frame / (fps * 60 / bpm))
        wave = 1.0 if beat_idx % 2 == 0 else 0.0
    
    # Apply PAS intensity envelope
    intensity = wave * params['pas_intensity_peak']
    
    # Convert to RGB via HSV
    rgb = hsv_to_rgb(params['hue'], params['sat'], intensity)
    
    return rgb
```

### Rig Rendering (Intention-Preserving)

```python
def render(self, rgb_array, seed=42):
    if self.rig_name == "reference":
        return rgb_array.copy()
    
    if self.rig_name.startswith("direct_"):
        # Baseline: naive truncation
        downsampled = direct_truncate_rgb(rgb_array, unique_positions)
        result = direct_expand_rgb(downsampled, leds)
    else:
        # Our approach: linear resampling
        downsampled = resample_rgb(rgb_array, unique_positions)
        if self.profile.mirroring:
            downsampled = apply_center_mirror_rgb(downsampled)
        result = resample_rgb(downsampled, leds)
    
    return result
```

---

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| FPS | 30 | Frame rate |
| hop_length | 735 | Audio hop (22050/30) |
| L_smooth | 81 | SSM smoothing window |
| H | 10 | SSM downsampling factor |
| L_kernel | 31 | Novelty kernel half-width |
| diagonal_margin | 3 | SSM diagonal mask |
| beat_sigma | 0.5 | Beat alignment tolerance (frames) |
| LED_COUNT | 33 | Physical LEDs per group |
