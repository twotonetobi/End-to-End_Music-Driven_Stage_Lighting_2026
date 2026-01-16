# Capabilities - Paper Evaluation Pipeline

> **Feature catalog and API reference**
> **Last Updated:** 2026-01-14

---

## Current Status

| Capability | Status | Script |
|------------|--------|--------|
| **Table 2 Cross-Rig** | COMPLETE | `cross_rig_evaluator.py` |
| **Table 3 Ablation** | COMPLETE | `ablation_evaluator.py` |

---

## Final Results

### Table 2: Cross-Rig (Temporal Correlations)

| Rig | Unique | Mean I | Peak I | Range | Hue | Sat |
|-----|--------|--------|--------|-------|-----|-----|
| Direct Mapping | 8 | 0.40 | 0.28 | 0.45 | 0.71 | 0.69 |
| Club (16 MH) | 8 (M) | 0.80 | 0.62 | 0.87 | 0.78 | 0.67 |
| Concert (56) | 19 | 0.82 | 0.75 | 1.00 | 0.86 | 0.65 |
| LED Bars (64) | 21 | 0.71 | 0.79 | 1.00 | 0.87 | 0.65 |
| **Reference** | **33** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |

### Table 3: Ablation (Full = 100%)

| Condition | SSM | Novelty | Beat Peak |
|-----------|-----|---------|-----------|
| **Full** | **100%** | **100%** | **100%** |
| Diffusion-only | 77% | 57% | 43% |
| Oscillator-only | 62% | 79% | 137% |
| RMS Baseline | 59% | 47% | 90% |

---

## Main API

### CrossRigEvaluator

```python
from scripts.cross_rig_evaluator import CrossRigEvaluator
from pathlib import Path

evaluator = CrossRigEvaluator(Path('configs'))

# Evaluate single segment
results = evaluator.evaluate_segment(
    geo_path='path/to/geo.pkl',
    pas_path='path/to/pas.pkl',
    bpm=126.0,
    rig_name='club'  # or 'concert', 'led_bars', 'reference', 'direct_club'
)
# Returns: {
#   'mean_intensity_corr': 0.80,
#   'peak_intensity_corr': 0.62,
#   'dynamic_range_corr': 0.87,
#   'hue_corr': 0.78,
#   'sat_corr': 0.67,
#   'unique_positions': 8
# }

# Batch processing
df = evaluator.evaluate_all_segments(limit=50)
summary = evaluator.compute_summary_statistics(df)
evaluator.print_summary_table(summary)

# LaTeX export
evaluator.export_latex_table(summary, Path('outputs/table2_latex.tex'))
```

### RigRenderer

```python
from scripts.rig_renderer import RigRenderer, RIG_PROFILES

renderer = RigRenderer('club')
rendered_rgb = renderer.render(original_rgb)

# Available rigs
print(RIG_PROFILES.keys())
# ['club', 'concert', 'led_bars', 'reference', 'direct_club']
```

### AblationEvaluator

```python
from scripts.ablation_evaluator import AblationEvaluator
from pathlib import Path

evaluator = AblationEvaluator(Path('configs'))

# Evaluate single segment
results = evaluator.evaluate_segment(
    geo_path='path/to/geo.pkl',
    pas_path='path/to/pas.pkl',
    audio_path='path/to/audio.wav',
    bpm=126.0,
    beat_times=[0.48, 0.95, 1.43, ...],
    song_name='Song_Name',
    segment_label='chorus'
)
# Returns dict by condition:
# {
#   'full': {'ssm_corr': 0.124, 'novelty_corr': 0.517, ...},
#   'diffusion_only': {...},
#   'oscillator_only': {...},
#   'retrieval': {...}  # RMS baseline
# }
```

---

## Metric Functions

### Table 2: Temporal Correlations

```python
from scripts.rig_renderer import (
    compute_mean_intensity_correlation,
    compute_peak_intensity_correlation,
    compute_dynamic_range_correlation,
    compute_color_correlation
)

mean_corr = compute_mean_intensity_correlation(original_rgb, rendered_rgb)
peak_corr = compute_peak_intensity_correlation(original_rgb, rendered_rgb)
range_corr = compute_dynamic_range_correlation(original_rgb, rendered_rgb)
hue_corr, sat_corr = compute_color_correlation(original_rgb, rendered_rgb)
```

### Table 3: Structural Metrics

```python
from scripts.ablation_evaluator import (
    ssm_correlation_masked,
    novelty_correlation,
    compute_beat_alignment_thesis
)

ssm_corr = ssm_correlation_masked(audio_features, light_features, diagonal_margin=3)
novelty_corr, _ = novelty_correlation(audio_features, light_features)
peak_align, valley_align = compute_beat_alignment_thesis(rgb_array, beat_times)
```

---

## Configuration

### configs/paths.yaml

```yaml
inference_data:
  diffusion: "../Touchdesigner/assets_GeoApproach/InferenceSet_ConformerModelNoRF_2024-11_seed150"
  oscillator: "../Touchdesigner/assets_GeoApproach/InferenceSet_ConforModelNoRFSet_2025-01_seed150"
  
song_timings: "../Touchdesigner/assets_GeoApproach/Audio_90s_Inference_Set_SongTimings"
audio_parts: "../Touchdesigner/assets_GeoApproach/Audio_90s_Inference_Set_parts"
```

---

## Rig Profiles

```python
RIG_PROFILES = {
    'club': RigProfile(
        name="Club Rig",
        total_fixtures=16,
        unique_positions_per_group=8,
        mirroring=True
    ),
    'concert': RigProfile(
        name="Concert Rig",
        total_fixtures=56,
        unique_positions_per_group=19,
        mirroring=False
    ),
    'led_bars': RigProfile(
        name="LED Bar Array",
        total_fixtures=64,
        unique_positions_per_group=21,
        mirroring=False
    ),
    'reference': RigProfile(
        name="Reference (Ours)",
        total_fixtures=99,
        unique_positions_per_group=33,
        mirroring=False
    ),
    'direct_club': RigProfile(
        name="Direct Mapping",
        total_fixtures=16,
        unique_positions_per_group=8,
        mirroring=False  # Baseline: naive truncation
    )
}
```

---

## Ablation Conditions

```python
CONDITIONS = {
    'full': AblationCondition(
        use_diffusion=True, 
        use_oscillator=True
    ),
    'diffusion_only': AblationCondition(
        use_diffusion=True, 
        use_oscillator=False
    ),
    'oscillator_only': AblationCondition(
        use_diffusion=False, 
        use_oscillator=True
    ),
    'retrieval': AblationCondition(  # RMS Baseline
        use_retrieval=True
    )
}
```

---

## Dependencies

```
numpy>=1.20.0
scipy>=1.7.0
pandas>=1.3.0
librosa>=0.9.0
pyyaml>=5.4.0
```

---

## Quick Start

```bash
cd zz_analyzer_for_paper
source venv/bin/activate
cd scripts

# Run Table 2: Cross-Rig Evaluation
python cross_rig_evaluator.py --limit 50

# Run Table 3: Ablation Study
python ablation_evaluator.py

# Quick test
python -c "
from cross_rig_evaluator import CrossRigEvaluator
from pathlib import Path

evaluator = CrossRigEvaluator(Path('../configs'))
df = evaluator.evaluate_all_segments(limit=10)
summary = evaluator.compute_summary_statistics(df)
evaluator.print_summary_table(summary)
"
```

---

## Output Files

```
outputs/
  table2_cross_rig.csv      # Raw results (song, segment, rig, metrics)
  table2_latex.tex          # LaTeX table for paper
  table3_ablation_results.csv # Raw ablation results
  table3_latex.tex          # LaTeX table for paper
```
