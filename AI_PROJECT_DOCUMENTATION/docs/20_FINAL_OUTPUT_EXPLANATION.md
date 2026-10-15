# 20: Final Output & User Interface Guide

## 1. Streamlit Dashboard Artifacts (`demo/app.py`)
- **Console Readouts:** Large numerical metric readouts displaying predicted class name and confidence score percentage in teal (`#4fb8a8`) and amber (`#e8b23d`).
- **Dual Heatmaps:** Side-by-side colorized Grad-CAM heatmaps overlaid onto the original raw frame and the PDE-denoised frame.
- **Diagnostic Consensus Card:** Dynamic notification indicating whether the raw and PDE models agreed or diverged.
- **Root-Cause Analysis Panels:** Detailed clinical explanations breaking down why the raw model suffered attention leakage and how PDE diffusion achieved focal attention.

## 2. Machine-Readable Export Files
- `results/test_metrics_all_runs.csv`: Raw metrics across seeds.
- `results/test_metrics_aggregated.csv`: Summary statistics (mean and std).
- `results/performance_comparison.png`: Statistical bar chart.
- `results/confusion_matrix_*.png`: Class-wise normalized confusion heatmaps.
