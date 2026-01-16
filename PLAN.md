# Implementation Plan - Paper Evaluation Pipeline

> **Last Updated:** 2026-01-14
> **Status:** TABLE 2 and TABLE 3 COMPLETE

---

## Current Status

### Table 3: Ablation Study - COMPLETE

| Task | Status |
|------|--------|
| Implement RMS-reactive baseline | Done |
| Fix Full to use combined GEO+PAS features | Done |
| Remove novelty score inflation | Done |
| Validate all metric rankings | Done |
| Document methodology | Done |

### Table 2: Cross-Rig - COMPLETE

| Task | Status |
|------|--------|
| Design temporal metrics approach | Done |
| Define 4 rig configs (Club, Concert, LED Bars, Reference) | Done |
| Implement intention-preserving resampling | Done |
| Add Direct Mapping baseline (naive truncation) | Done |
| Implement temporal correlation metrics | Done |
| Run evaluation on 50 segments | Done |
| Document methodology | Done |

---

## Final Results

### Table 2: Cross-Rig (Temporal Correlations)

| Rig | Unique | Mean I | Peak I | Range |
|-----|--------|--------|--------|-------|
| Direct Mapping | 8 | 0.40 | 0.28 | 0.45 |
| Club (16 MH) | 8 (M) | 0.80 | 0.62 | 0.87 |
| Concert (56) | 19 | 0.82 | 0.75 | 1.00 |
| LED Bars (64) | 21 | 0.71 | 0.79 | 1.00 |
| **Reference** | **33** | **1.00** | **1.00** | **1.00** |

**Key insight:** At same resolution (8 positions), our approach achieves 2x better correlation than naive baseline.

### Table 3: Ablation (Full = 100%)

| Condition | SSM | Novelty | Beat Peak |
|-----------|-----|---------|-----------|
| **Full** | **100%** | **100%** | **100%** |
| Diffusion-only | 77% | 57% | 43% |
| Oscillator-only | 62% | 79% | 137% |
| RMS Baseline | 59% | 47% | 90% |

---

## Fixes Applied (2026-01-13)

### 1. Combined GEO+PAS for Full Condition
- **Problem:** Full and Diffusion-only used same PAS features
- **Solution:** Full uses RGB-extracted features (captures combined GEO+PAS effect)

### 2. Removed Novelty Inflation
- **Problem:** "Functional Quality" transformation inflated scores 2-3x
- **Solution:** Return raw novelty correlation values

### 3. RMS-Reactive Baseline
- **Problem:** Previous baselines used trained models (spurious correlation)
- **Solution:** Simple `brightness = RMS(audio)` as true naive baseline

### 4. Fixed BPM for Oscillator-only
- **Problem:** Using actual BPM created artificial alignment
- **Solution:** Fixed BPM=120 reduces artificial beat correlation

---

## Methodology

### Baseline: Sound-to-Light
```python
brightness = normalize(librosa.feature.rms(audio))
pas_intensity = brightness  # Direct mapping
```

### Feature Extraction
- **Audio:** 12D chroma at 30fps
- **Lighting:** 18D RGB-extracted (6 features x 3 LX groups)

### SSM Computation
1. Smooth (L=81) -> Downsample (H=10)
2. SSM: S(i,j) = 1 - ||x_i - x_j||_2 / sqrt(d)
3. Mask diagonal (margin=3)
4. Pearson correlation

---

## Next Steps

1. ~~Complete Table 3 ablation~~ DONE
2. ~~Implement Table 2: Cross-Rig evaluation~~ DONE
3. Export LaTeX tables for paper - DONE (outputs/table2_latex.tex)
4. Generate visualizations (optional)

---

## File Structure

```
zz_analyzer_for_paper/
├── RESULTS.md          <- Current results with discussion
├── PLAN.md             <- This file
├── README.md           <- Quick start
├── KNOWLEDGE.md        <- Technical details
├── CAPABILITIES.md     <- API reference
├── configs/
│   └── paths.yaml
├── outputs/
│   ├── table2_cross_rig.csv   <- Table 2 raw results
│   ├── table2_latex.tex       <- Table 2 LaTeX
│   ├── table3_ablation_results.csv
│   └── table3_latex.tex
└── scripts/
    ├── cross_rig_evaluator.py  <- Table 2 evaluator (COMPLETE)
    ├── rig_renderer.py         <- Rig simulation + metrics
    ├── ablation_evaluator.py   <- Table 3 evaluator (COMPLETE)
    ├── offline_processor.py    <- RGB generation
    └── audio_feature_extractor.py
```
