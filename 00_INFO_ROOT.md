# Documentation Index

> **Start here.** This file maps all documentation in this repository.  
> **Repository:** [End-to-End Music-Driven Stage Lighting](https://github.com/twotonetobi/End-to-End_Music-Driven_Stage_Lighting_2026)  
> **Paper:** PETRA 2025 - "End-to-End Music-Driven Stage Lighting: A Co-Creative Framework"

---

## Quick Reference

| Document | Purpose | Audience |
|----------|---------|----------|
| [README.md](./README.md) | **Start here** - Overview, quick start, results summary | Reviewers, Users |
| [SUPPLEMENTARY.md](./SUPPLEMENTARY.md) | **Technical details** - Feature definitions, metrics, implementation | Researchers |
| [RESULTS.md](./RESULTS.md) | Complete evaluation results with methodology | Paper authors |
| [KNOWLEDGE.md](./KNOWLEDGE.md) | Technical architecture details | Developers |
| [CAPABILITIES.md](./CAPABILITIES.md) | API reference | Developers |
| [PLAN.md](./PLAN.md) | Implementation status tracking | Developers |

---

## Paper Results Summary

### Table 2: Cross-Rig Retargeting (Intention Preservation)

| Rig | Positions | Mean I | Peak I | Range | Hue |
|-----|-----------|--------|--------|-------|-----|
| Direct Mapping | 8 | 0.40 | 0.28 | 0.45 | 0.71 |
| Club Rig | 8* | 0.80 | 0.62 | 0.87 | 0.78 |
| Concert Rig | 19 | 0.82 | 0.75 | 1.00 | 0.86 |
| LED Bar Array | 21 | 0.71 | 0.79 | 1.00 | 0.87 |
| **Reference** | **33** | **1.00** | **1.00** | **1.00** | **1.00** |

**Key Finding:** 2x better temporal correlation than naive mapping at same spatial resolution.

### Table 3: Ablation Study (Full = 100%)

| Condition | SSM | Novelty | Beat Peak | Beat Valley |
|-----------|-----|---------|-----------|-------------|
| **Full (Ours)** | **100%** | **100%** | 100% | 100% |
| Diffusion-only | 77% | 57% | 43% | 64% |
| Oscillator-only | 62% | 79% | 137% | 136% |
| RMS Baseline | 59% | 47% | 90% | 98% |

**Key Finding:** Dual-branch architecture outperforms individual branches on structural metrics.

---

## Evaluation Scripts

| Script | Paper Table | Description |
|--------|-------------|-------------|
| [`scripts/cross_rig_evaluator.py`](./scripts/cross_rig_evaluator.py) | Table 2 | Cross-rig retargeting validation |
| [`scripts/ablation_evaluator.py`](./scripts/ablation_evaluator.py) | Table 3 | Dual-model ablation study |
| [`scripts/rig_renderer.py`](./scripts/rig_renderer.py) | - | Intention-preserving renderers |
| [`scripts/audio_feature_extractor.py`](./scripts/audio_feature_extractor.py) | - | Audio feature extraction |
| [`scripts/offline_processor.py`](./scripts/offline_processor.py) | - | RGB generation from abstractions |

---

## Reading Order by Goal

| Goal | Start With |
|------|------------|
| "What is this project?" | [README.md](./README.md) |
| "What are the paper results?" | [RESULTS.md](./RESULTS.md) |
| "How are metrics defined?" | [SUPPLEMENTARY.md](./SUPPLEMENTARY.md) |
| "How do I run evaluations?" | [README.md](./README.md) > Quick Start |
| "How does the code work?" | [KNOWLEDGE.md](./KNOWLEDGE.md) |

---

## Related Resources

- **Cross-Modal Metrics Repository:** [GitHub](https://github.com/MKKeys92/Cross-Modal-Metrics-for-Capturing-Correspondences-in-Stage-Performances)
- **Metric Evaluation Framework:** [GitHub](https://github.com/twotonetobi/MSc_thesis_metric_evaluation)
