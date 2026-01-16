# Supplementary Material: End-to-End Music-Driven Stage Lighting

**Paper:** End-to-End Music-Driven Stage Lighting: A Co-Creative Framework  
**Author:** Tobias Wursthorn  
**Conference:** PETRA 2025  
**Repository:** [https://github.com/twotonetobi/End-to-End_Music-Driven_Stage_Lighting_2026](https://github.com/twotonetobi/End-to-End_Music-Driven_Stage_Lighting_2026)

---

## Table of Contents

1. [Extraction Pipelines and Feature Definitions](#1-extraction-pipelines-and-feature-definitions)
2. [Hybrid Dynamic Scoring for Oscillator Labels](#2-hybrid-dynamic-scoring-for-oscillator-labels)
3. [Metric Definitions and Functional Quality](#3-metric-definitions-and-functional-quality)
4. [Baselines and Conditioning Ablations](#4-baselines-and-conditioning-ablations)
5. [Implementation Details](#5-implementation-details)
6. [Overall Quality Score Weights and Sensitivity](#6-overall-quality-score-weights-and-sensitivity)
7. [Cross-Rig Configuration Details](#7-cross-rig-configuration-details)

---

## 1. Extraction Pipelines and Feature Definitions

This section provides implementation-level definitions for the global intention features and oscillator representation.

### 1.1 Global Intention Features

For each semantic group $g$ and frame $t$, let $I_i(t) \in [0,1]$ denote the normalized intensity of fixture $i \in g$, and $H_i(t), S_i(t) \in [0,1]$ denote hue and saturation (HSV).

#### Peak Intensity Envelope

$$I_{\text{peak}}(g,t) = \max_{i \in g} I_i(t)$$

#### Spatial Intensity Gradient

$$\nabla_s I(g,t) = \frac{1}{N_g-1} \sum_{i=2}^{N_g} \bigl| I_i(t) - I_{i-1}(t) \bigr|$$

with fixtures ordered spatially and $N_g$ fixtures per group.

#### Peak Density (Active Fraction)

$$\rho_{\text{peak}}(g,t) = \frac{1}{N_g} \sum_{i \in g} \mathbf{1}[I_i(t) > \tau_{\text{active}}]$$

using $\tau_{\text{active}} = 0.1$.

#### Inverse Minima

$$I_{\text{min\_inv}}(g,t) = 1 - \min_{i \in g} I_i(t)$$

This feature serves as a proxy for contrast and negative-space phenomena. It is not a complete representation of intentional darkness.

#### Color Statistics

$$\bar{H}(g,t) = \operatorname{mean}_{i \in g} H_i(t), \quad \bar{S}(g,t) = \operatorname{mean}_{i \in g} S_i(t)$$

Hue is circular. In the present pipeline, arithmetic means were used because hue values in the dataset did not cluster near the wrap boundary. A circular mean is recommended for broader deployments.

### 1.2 Movement Activity Index (MAI)

Pan and tilt channels (when present) are normalized to $[0,1]$ to reduce dependence on fixture-specific ranges. MAI aggregates segment-level movement velocity:

$$\mathrm{MAI}(g,s) = \frac{1}{T}\sum_{t=1}^T \sqrt{\Bigl(\frac{\delta \mathrm{pan}}{\delta t}\Bigr)^2 + \Bigl(\frac{\delta \mathrm{tilt}}{\delta t}\Bigr)^2}$$

For fixtures or groups without pan/tilt channels, MAI defaults to 0.

---

## 2. Hybrid Dynamic Scoring for Oscillator Labels

Oscillator wave-type labels are derived via a Hybrid Dynamic Scoring method that combines an intention-based peak count with ranges of fitted oscillator parameters.

Let $N_{\text{peaks}}(g,s)$ denote the number of salient peaks in $I_{\text{peak}}(g,t)$ within segment $s$. Let $\Delta\phi(g,s)$, $\Delta\omega(g,s)$, and $\Delta O(g,s)$ denote within-segment ranges of fitted phase, frequency, and offset. Each range is normalized to $[0,1]$ based on empirical ranges. The composite score is:

$$S_{\text{dyn}}(g,s) = \frac{1}{2}\left( \frac{N_{\text{peaks}}(g,s)}{\theta_{\text{osc}}} + \frac{\operatorname{norm}(\Delta\phi) + \operatorname{norm}(\Delta\omega) + \operatorname{norm}(\Delta O)}{3} \right)$$

The score is mapped to a discrete wave family using empirically chosen thresholds:

| Score Range | Wave Type |
|-------------|-----------|
| 0.0 - 0.2 | Static |
| 0.2 - 0.4 | Sine |
| 0.4 - 0.6 | Triangle |
| 0.6 - 0.8 | Sawtooth |
| 0.8 - 1.0 | Square |

This procedure produces practical pseudo-labels but may introduce label noise. See the Discussion section in the main paper for limitations.

---

## 3. Metric Definitions and Functional Quality

This section summarizes the intention-based metrics adapted from Kohl et al. [1,2], including the functional-quality transformations applied to novelty and loudness/brightness correspondence.

### 3.1 Self-Similarity Matrix (SSM) Correlation

SSM correlation measures how well the lighting self-similarity structure matches the audio self-similarity structure:

$$\text{SSM}_{\text{corr}} = \text{Pearson}(\text{vec}(S_{\text{audio}}), \text{vec}(S_{\text{light}}))$$

where $S$ denotes the self-similarity matrix computed via cosine similarity on feature embeddings.

### 3.2 Novelty Correlation

Novelty functions capture structural boundaries. A Gaussian-kernel novelty function is computed along the SSM diagonal for both audio and lighting:

$$\text{Nov}_{\text{corr}} = \text{Pearson}(N_{\text{audio}}, N_{\text{light}})$$

### 3.3 Beat Peak Alignment

Beat peak alignment measures how well lighting intensity peaks align with musical beats:

$$\text{BeatAlign} = \frac{1}{|B|} \sum_{b \in B} \max_{t \in [b-\delta, b+\delta]} I_{\text{peak}}(t)$$

where $B$ is the set of beat times and $\delta = 50\text{ms}$ is the tolerance window.

### 3.4 Transition Smoothness

Transition smoothness penalizes abrupt changes at segment boundaries:

$$\text{TransSmooth} = 1 - \frac{1}{|S|} \sum_{s \in S} \| I(t_s^-) - I(t_s^+) \|_2$$

where $t_s^-$ and $t_s^+$ denote frames immediately before and after segment boundary $s$.

### 3.5 Achievement Ratio

For each metric $M$, the Achievement Ratio compares generated outputs to professional references:

$$\mathrm{AR}_M = \frac{\operatorname{median}(M_{\text{gen}})}{\operatorname{median}(M_{\text{gt}})}$$

Ratios are clipped to $[0, 2]$ before aggregation.

---

## 4. Baselines and Conditioning Ablations (Negative Results)

Two Seq2Seq baselines were implemented to regress 72D intention sequences from audio features at 30 Hz.

### 4.1 Transformer Seq2Seq

A vanilla encoder-decoder Transformer (4 layers, hidden size 512) learned coarse envelopes but produced temporally unstable outputs with high-frequency noise and weak long-range coherence on validation tracks. Scaling hidden size to 1024 did not resolve these issues.

### 4.2 LSTM Seq2Seq

An LSTM Seq2Seq baseline exhibited fast mode collapse during inference: output was varied for approximately 10-20 seconds and then became near-constant or silent for the remainder of the sequence.

### 4.3 Conditioning Features

Early attempts using only conventional MIR features (e.g., MFCCs, mel spectrograms, chromagrams) as conditioning context were inadequate for long-form coherent styles. Jukebox embeddings were retained to provide higher-level semantic context at increased preprocessing cost.

| Conditioning | SSM Corr | Novelty Corr | Beat Align |
|--------------|----------|--------------|------------|
| MIR-only | 0.089 | 0.612 | 1.102 |
| Jukebox-only | 0.134 | 0.778 | 0.987 |
| MIR + Jukebox | 0.162 | 0.822 | 1.257 |

---

## 5. Implementation Details

### 5.1 Global Intention Diffusion Model

| Parameter | Value |
|-----------|-------|
| Architecture | Stacked Conformer blocks with gated conditioning (EDGE-based) |
| Parameters | ~49 million |
| Training epochs | 200 |
| Optimizer | Adam |
| Learning rate | $10^{-4}$ |
| Batch size | 64 (reduced from 512 to fit 24GB VRAM) |
| Loss function | L2 reconstruction (MSE between predicted and actual noise) |
| Noise schedule | Linear beta schedule, tuned for lighting distribution |
| Dropout | 25% on music conditioning (classifier-free guidance) |
| Training time | ~4 days on NVIDIA RTX A5000 (24GB) |
| Inference | DDIM sampling with 50 steps (~1.2x real-time) |

### 5.2 Segment-Based Oscillator Model

| Parameter | Value |
|-----------|-------|
| Architecture | Conformer encoder with per-segment prediction heads |
| Attention heads | 8 (expanded channel capacity) |
| Training epochs | 150 |
| Optimizer | Adam |
| Learning rate | $5 \times 10^{-5}$ |
| Weight decay | 0.01 |
| Batch size | 64 |
| Training time | ~24 hours on NVIDIA RTX A5000 (24GB) |
| Inference | ~0.8x real-time |

### 5.3 Audio Feature Extraction

| Feature Type | Details |
|--------------|---------|
| Frame rate | 30 Hz (matched to lighting) |
| Rhythmic features | Onset curves, beat/downbeat positions (madmom) |
| Structural features | Chroma-based self-similarity, spectral flux |
| Semantic features | Jukebox embeddings (layer 36, 4800-dim) |
| Redundant onset features | Included for Conformer conditioning |

### 5.4 Additional Training Notes

- **Auxiliary losses:** Custom lighting coherence objectives enforce valid parameter ranges and consistency across fixture groups (replacing joint-velocity/contact losses from dance synthesis)
- **Temporal resolution:** All features aligned at consistent 30 FPS
- **Real-time feasibility:** System designed for offline pre-production; interpreter enables real-time playback of pre-generated abstractions

---

## 6. Overall Quality Score Weights and Sensitivity

The Overall Quality Score combines three components using predefined weights:

| Component | Weight | Description |
|-----------|--------|-------------|
| Structural/Temporal Analysis | 16% | SSM correlation, novelty correlation |
| Intention-Based Reference Comparison | 42% | Beat alignment, dynamic range, color fidelity |
| Oscillator Evaluation | 42% | Wave-type accuracy, parameter fidelity |

### Sensitivity Analysis

Sensitivity checks under alternative weightings yield variations of approximately ±3 percentage points:

| Weighting Scheme | Overall Quality Score |
|------------------|----------------------|
| Default (16/42/42) | 76.4% |
| Equal (33/33/33) | 78.3% |
| Structure-heavy (40/30/30) | 74.1% |
| Rhythm-heavy (16/25/59) | 75.8% |

---

## 7. Cross-Rig Configuration Details

This section documents the three rig configurations used for cross-rig retargeting validation (Table 1 in main paper).

### 7.1 Rig A: Beam-Focused (Robe BMFL)

| Property | Value |
|----------|-------|
| Primary fixtures | Robe BMFL Beam/Wash |
| Configuration | Three truss sections |
| Character | Beam/spotlight-focused generation |
| DMX footprint | Standard BMFL mode |

### 7.2 Rig B: LED Bar Array (GLP Bars)

| Property | Value |
|----------|-------|
| Primary fixtures | GLP Bars (LED linear fixtures) |
| Configuration | Multiple truss positions |
| Character | Continuous LED strip paradigm |
| DMX footprint | Different structure from conventional moving heads |

### 7.3 Rig C: Mixed Heterogeneous

| Property | Value |
|----------|-------|
| Primary fixtures | Mixed luminaire types |
| Configuration | Heterogeneous setup |
| Character | Real-world production scenario |
| DMX footprint | Varying footprints across fixture types |

---

## References

[1] Kohl, M., Wursthorn, T., and Weiss, C. (2025). Cross-Modal Metrics for Capturing Correspondences Between Music Audio and Stage Lighting Signals. In Proceedings of the 33rd ACM International Conference on Multimedia (MM '25), pp. 528-534. https://doi.org/10.1145/3746027.3755488

[2] Kohl, M. (2023). Generating Stage Lighting Controls from Music Audio Input. M.S. thesis, Julius Maximilians Universitat Wurzburg.

---

## 8. Dataset Provenance

The professional corpus used for training and evaluation consists of approximately 1,400 minutes (280 songs) of synchronized music and lighting control data from professional live performances, including festivals and touring productions.

### Data Sources

| Source | Content | Access |
|--------|---------|--------|
| Roskilde Festival | Subset of abstracted training data | Public (via Cross-Modal Metrics repo) |
| Touring Productions | Additional professional shows | Restricted (NDA) |
| Festival Archives | Multi-artist performances | Restricted (NDA) |

### Public Data Availability

A subset of the training data from the Roskilde Festival is publicly available through the cross-modal metrics repository:

**Cross-Modal Metrics Repository:** [https://github.com/MKKeys92/Cross-Modal-Metrics-for-Capturing-Correspondences-in-Stage-Performances](https://github.com/MKKeys92/Cross-Modal-Metrics-for-Capturing-Correspondences-in-Stage-Performances)

This repository includes:
- Abstracted lighting data from professional shows
- Audio feature representations
- Evaluation code for cross-modal metrics

### Metric Evaluation Framework

The cross-modal metric evaluation framework is available at:

**Metric Evaluation Repository:** [https://github.com/twotonetobi/MSc_thesis_metric_evaluation](https://github.com/twotonetobi/MSc_thesis_metric_evaluation)

---

## Code Repository

The abstraction and evaluation pipeline with example data is available at:

**GitHub:** [https://github.com/twotonetobi/End-to-End_Music-Driven_Stage_Lighting_2026](https://github.com/twotonetobi/End-to-End_Music-Driven_Stage_Lighting_2026)

The repository includes:
- Feature extraction code
- Metric computation scripts
- Example data for reproducing evaluation results
- Cross-rig interpreter configuration examples
