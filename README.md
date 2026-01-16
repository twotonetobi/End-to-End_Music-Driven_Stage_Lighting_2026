# End-to-End Music-Driven Stage Lighting

> **Supplementary Material for PETRAE 2025**  
> **Paper:** "End-to-End Music-Driven Stage Lighting: A Co-Creative Framework"  
> **Authors:** Tobias Wursthorn, Michael Kohl, Christof Weiß, Kai von Luck, Peer Stelldinger, Larissa Putzar

---

## Overview

This repository provides the evaluation pipeline and supplementary material for our PETRAE 2025 paper. The system generates retargetable lighting proposals through a **Dual Abstraction Layer**, combining:

- **Global Intention Model** (Diffusion-based) for slow-varying aesthetic dynamics
- **Segment-Based Oscillator Model** (Conformer-based) for rhythmic patterns

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/twotonetobi/End-to-End_Music-Driven_Stage_Lighting_2026.git
cd End-to-End_Music-Driven_Stage_Lighting_2026

# 2. Create and activate virtual environment
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# 3. Install dependencies
pip install numpy scipy librosa scikit-learn pyyaml
```

### Verify Installation

```bash
python -c "import numpy, scipy, librosa; print('All dependencies installed!')"
```

---

## Repository Structure

```
├── README.md                  # This file - start here
├── SUPPLEMENTARY.md           # Technical appendix (feature definitions, metrics)
├── RESULTS.md                 # Detailed evaluation results and discussion
│
├── scripts/                   # Evaluation scripts
│   ├── cross_rig_evaluator.py       # Table 2: Cross-rig retargeting
│   ├── ablation_evaluator.py        # Table 3: Ablation study
│   ├── rig_renderer.py              # Intention-preserving renderers
│   ├── audio_feature_extractor.py   # Audio feature extraction
│   └── offline_processor.py         # RGB generation from abstractions
│
├── outputs/                   # Pre-computed results
│   ├── table2_cross_rig.csv         # Cross-rig results (CSV)
│   ├── table2_latex.tex             # Cross-rig results (LaTeX)
│   ├── table3_ablation_results.csv  # Ablation results (CSV)
│   └── table3_latex.tex             # Ablation results (LaTeX)
│
├── configs/                   # Configuration files
│   ├── paths.yaml                   # Data paths
│   └── rigs/                        # Rig configuration examples
│       ├── rig_a_bmfl.yaml
│       ├── rig_b_bars.yaml
│       └── rig_c_mixed.yaml
│
└── assets/                    # Additional assets
```

---

## Running the Evaluation Scripts

### Table 2: Cross-Rig Retargeting Validation

Tests how well temporal dynamics survive when adapting to different rig configurations.

```bash
cd scripts
python cross_rig_evaluator.py
```

**Output:** `outputs/table2_cross_rig.csv` and `outputs/table2_latex.tex`

### Table 3: Ablation Study

Compares the full dual-branch system against ablated versions and baselines.

```bash
cd scripts
python ablation_evaluator.py
```

**Output:** `outputs/table3_ablation_results.csv` and `outputs/table3_latex.tex`

---

## Paper Results Summary

### Table 2: Intention Preservation Under Cross-Rig Application

| Rig Configuration | Positions | Mean I | Peak I | Range | Hue |
|-------------------|-----------|--------|--------|-------|-----|
| Direct Mapping | 8 | 0.40 | 0.28 | 0.45 | 0.71 |
| Club Rig | 8* | 0.80 | 0.62 | 0.87 | 0.78 |
| Concert Rig | 19 | 0.82 | 0.75 | 1.00 | 0.86 |
| LED Bar Array | 21 | 0.71 | 0.79 | 1.00 | 0.87 |
| **Reference (Ours)** | **33** | **1.00** | **1.00** | **1.00** | **1.00** |

*All values are Pearson correlations. \*Center-mirrored configuration.*

**Key Finding:** At the same spatial resolution (8 positions), intention-preserving resampling achieves **2x better** mean intensity correlation than naive mapping.

### Table 3: Ablation Study (Full = 100%)

| Condition | SSM Corr | Novelty | Beat Peak |
|-----------|----------|---------|-----------|
| **Full (Ours)** | **100%** | **100%** | 100% |
| Diffusion-only | 77% | 57% | 43% |
| Oscillator-only | 62% | 79% | 137% |
| RMS Baseline | 59% | 47% | 90% |

**Key Finding:** Dual-branch architecture outperforms individual branches on structural metrics.

---

## Documentation

| Document | Description |
|----------|-------------|
| [SUPPLEMENTARY.md](./SUPPLEMENTARY.md) | Technical appendix with feature definitions, metric formulas, implementation details |
| [RESULTS.md](./RESULTS.md) | Complete evaluation results with methodology and discussion |
| [KNOWLEDGE.md](./KNOWLEDGE.md) | Technical architecture documentation |
| [CAPABILITIES.md](./CAPABILITIES.md) | API reference for the evaluation scripts |

---

## Related Repositories

| Repository | Description |
|------------|-------------|
| [Cross-Modal Metrics](https://github.com/MKKeys92/Cross-Modal-Metrics-for-Capturing-Correspondences-in-Stage-Performances) | Evaluation framework and Roskilde Festival dataset |
| [Metric Evaluation](https://github.com/twotonetobi/MSc_thesis_metric_evaluation) | Extended metric evaluation pipeline |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{wursthorn2025lighting,
  author    = {Wursthorn, Tobias and Kohl, Michael and Wei{\ss}, Christof and 
               von Luck, Kai and Stelldinger, Peer and Putzar, Larissa},
  title     = {End-to-End Music-Driven Stage Lighting: A Co-Creative Framework},
  booktitle = {Proceedings of the 18th International Conference on PErvasive 
               Technologies Related to Assistive Environments (PETRAE '25)},
  year      = {2025},
  publisher = {Springer}
}
```

---

## License

This project is released for academic research purposes. The professional lighting corpus used for training is subject to NDAs and cannot be redistributed. A subset of abstracted data from the Roskilde Festival is available through the [Cross-Modal Metrics repository](https://github.com/MKKeys92/Cross-Modal-Metrics-for-Capturing-Correspondences-in-Stage-Performances).

---

## Contact

**Tobias Wursthorn**  
Hamburg University of Applied Sciences  
tobias.wursthorn@haw-hamburg.de
