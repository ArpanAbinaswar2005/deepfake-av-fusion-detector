# deepfake-av-fusion-detector

> **Deepfake Detection Under Common Real-World Video Degradation Using Quality-Aware Fusion Gate**  
> *Accepted — Under Publication at ICCCNET 2026, Springer*

---

## Overview

This repository contains the research implementation for a **dual-branch deepfake detection system** that remains robust under real-world video degradations such as social media compression and blurring — conditions where most existing detectors fail.

The core idea is simple: even when a video is heavily compressed and visual artifacts are destroyed, the **timing mismatch between lip movements and audio** in a deepfake still remains detectable. We exploit this using a quality-aware fusion strategy that adaptively shifts reliance from visual analysis to audio-visual synchronization based on input quality.

---

## The Problem

Modern lip-sync deepfakes — where only the mouth region is manipulated to match a target audio — are increasingly difficult to detect after they have been shared on platforms like WhatsApp, Facebook, or X (Twitter). Standard social media processing (H.264 compression, re-uploads, screen recording) destroys the subtle pixel-level fingerprints that most detectors rely on.

---

## Proposed Solution

We propose a **three-stage pipeline**:

1. **Degradation Module** — Simulates real-world social media laundering using FFmpeg (H.264 CRF 35, Gaussian Blur σ=3)
2. **Dual-Branch Detection**
   - **Spatial Artifact Branch** — ResNet-50 encoder + DCT/FFT frequency analysis for pixel-level forgery detection
   - **Complementary Sync Branch** — SyncNet-based audio-visual lip-sync consistency scoring using Modality Dissonance Score (MDS)
3. **Quality-Aware Fusion Gate** — Dynamically weights both branches using Laplacian Variance as a quality metric (α), following:

```
S_final = (α × S_spatial) + ((1 - α) × S_sync)
```

When video quality is low (heavy compression), the gate shifts ~90% weight to the sync branch, maintaining detection accuracy where spatial methods collapse.

---

## Results

Evaluated on a pilot batch of **100 videos (50 real, 50 fake)** from the [LAV-DF dataset](https://github.com/ControlNet/LAV-DF) under severe degradation:

| Metric | Proposed Method | SyncNet (Temporal Only) | ResNet-50 (Spatial Only) |
|---|---|---|---|
| **Accuracy** | **100%** | ~91% | ~58% |
| **AUROC** | **1.0** | 0.91 | 0.62 |
| **F1-Score** | **1.0** | 0.85 | 0.70 |
| **False Negative Rate** | **0%** | — | ~42% |

The fusion gate operated at an average Quality Alpha (α) of **0.10**, confirming the degradation module successfully reduced spatial reliability and forced reliance on the sync branch.

---

## Implementation

The full pipeline was implemented and run in **Google Colab** using the following tools and libraries:

- `Python`
- `PyTorch` — ResNet-50 spatial branch
- `SyncNet (V2)` — Audio-visual synchronization scoring
- `OpenCV` — Face detection and mouth ROI extraction
- `FFmpeg` — Automated degradation pipeline
- `NumPy`, `Matplotlib` — Data processing and visualization

---

## Repository Structure

```
deepfake-av-fusion-detector/
│
├── notebooks/
│   └── deepfake_detection_pipeline.ipynb   # Main Google Colab notebook
│
├── results/
│   ├── confusion_matrix.png
│   ├── fusion_gate_scatter.png
│   ├── accuracy_vs_crf.png
│   └── auroc_comparison.png
│
├── README.md
└── requirements.txt
```

---

## Research Paper

**"Deepfake Detection Under Common Real-World Video Degradation Using Quality-Aware Fusion Gate Performing Prediction on Metric-Based Analysis of Audio-Visual Lip-Sync as Auxiliary Signal"**

*Arpan Abinaswar, Armaan Jain, Arka Routh*  
*Kalinga Institute of Industrial Technology (KIIT), Bhubaneswar*  
*Accepted — Under Publication, ICCCNET 2026, Springer*

---

## Authors

| Name | Email |
|---|---|
| Arpan Abinaswar | 24155163@kiit.ac.in |
| Armaan Jain | 241551026@kiit.ac.in |
| Arka Routh | 24155162@kiit.ac.in |

---

## Acknowledgements

- [LAV-DF Dataset](https://github.com/ControlNet/LAV-DF) — Localized Audio-Visual Deepfake dataset
- [SyncNet](https://github.com/joonson/syncnet_python) — Audio-visual synchronization model by Chung & Zisserman
- [LipForensics](https://arxiv.org/abs/2012.07657) — Baseline comparison method
