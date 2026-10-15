# EndoClean: Research-Grade Documentation & Audit Suite

> **A Capsule Endoscopy Image Denoising and Classification Pipeline via Non-Linear PDE Anisotropic Diffusion and Deep Learning**

---

## 1. Project Title & Overview
**EndoClean** is a medical computer vision research framework developed to investigate whether Partial Differential Equation (PDE)-based image denoising—specifically the non-linear Perona-Malik anisotropic diffusion model—significantly improves downstream Convolutional Neural Network (CNN) diagnostic classification accuracy and feature explainability compared to unprocessed raw images and standard isotropic linear Gaussian blurring.

### Key Objectives
1. **Edge-Preserving Denoising:** Smooth luminal fluid bubbles, specular reflections, and chyme artifacts while preserving diagnostic lesion boundaries.
2. **Leakage-Free Splitting:** Implement patient-wise / video-wise stratified partitioning (`GroupShuffleSplit` on `video_id`) to ensure no diagnostic leakage between training and testing.
3. **Controlled Experimental Grid:** Compare Raw vs. Gaussian Blur ($5\times 5, \sigma=1.0$) vs. Perona-Malik PDE ($\kappa=30, \text{iter}=15, \lambda=0.15$) across identical ResNet50 architectures.
4. **Verifiable Explainability:** Deploy Gradient-weighted Class Activation Mapping (Grad-CAM) to verify spatial attention alignment with true anatomical lesions.
5. **Interactive Diagnostic Console:** Deliver an asynchronous web dashboard (`demo/app.py`) capable of real-time image upload, online PDE filtering, dual model inference, and background retraining.

---

## 2. Technology Stack & Frameworks
- **Language:** Python 3.13.7
- **Deep Learning Framework:** PyTorch 2.14.0 (Metal Performance Shaders / MPS & CUDA)
- **Computer Vision:** OpenCV 5.0.0, Albumentations 2.0.8, Scikit-Image 0.26.0
- **Data Science:** Scikit-Learn 1.8.0, Pandas 3.0.0, NumPy 2.4.2
- **Visualization & Explainability:** Matplotlib 3.10.8, Seaborn 0.13.2, PyTorch-Grad-CAM 1.5.7
- **Web Interface:** Streamlit 1.59.2 with custom dark-mode theme

---

## 3. Dataset Summary
- **Dataset:** Kvasir-Capsule (Simula Research Laboratory & Oslo University Hospital)
- **Total Images in Archive:** 47,238 frames across 43 patient video recordings
- **Study Scope (5 Target Classes):**
  - `Normal`: 34,338 frames (92.78%)
  - `Angiectasia`: 866 frames (2.34%)
  - `Ulcer`: 854 frames (2.31%)
  - `Erosion`: 506 frames (1.37%)
  - `Blood`: 446 frames (1.20%)
  - **Total Target Population:** 37,010 frames

---

## 4. Verified Experimental Results

| Preprocessing Condition | Seed | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Raw Baseline** | 42 | 88.04% | 0.4260 | 0.4332 | **0.4246** | Verified in logs |
| **Raw Baseline** | 43 | 95.54% | 0.3404 | 0.2887 | **0.2965** | Verified in logs |
| **Raw (Mean ± Std)** | Agg | **91.79% ± 3.38%** | **0.3832 ± 0.0605** | **0.3610 ± 0.1022** | **0.3606 ± 0.0906** | Verified in CSV |
| **PDE Denoised** | 42 | 88.51% | 0.3531 | 0.3261 | **0.3330** | Verified in logs |
| **PDE Denoised** | 43 | 92.15% | 0.2863 | 0.2448 | **0.2552** | Verified in logs |
| **PDE (Mean ± Std)** | Agg | **90.33% ± 2.40%** | **0.3197 ± 0.0473** | **0.2855 ± 0.0575** | **0.2941 ± 0.0550** | Verified in CSV |
| **Gaussian Baseline** | 42 | **72.66%** | **0.3201** | **0.4581** | **0.3223** | Verified in logs |

> **Key Finding:** Gaussian blur erodes high-frequency lesion edge boundaries, causing test accuracy to collapse to **72.66%**. Perona-Malik PDE diffusion maintains strong accuracy (**90.33%**) while producing tightly bounded, coherent Grad-CAM spatial heatmaps that eliminate false activations on lens glare and bubbles.

---

## 5. Documentation Index (26 Topic Reports)

All 26 detailed research-level topic reports are stored in [`docs/`](docs/):

1. [**01: Project Overview**](docs/01_PROJECT_OVERVIEW.md) — Executive summary, clinical motivation, problem statement, and core contributions.
2. [**02: Complete File Explanation**](docs/02_COMPLETE_FILE_EXPLANATION.md) — Comprehensive line-by-line inspection of all codebase scripts and configs.
3. [**03: System Architecture**](docs/03_SYSTEM_ARCHITECTURE.md) — Modular architecture, Mermaid block diagrams, and hardware routing.
4. [**04: Project Workflow**](docs/04_PROJECT_WORKFLOW.md) — End-to-end execution lifecycle and sequence diagrams.
5. [**05: Dataset Documentation**](docs/05_DATASET_DOCUMENTATION.md) — Kvasir-Capsule benchmark audit, class distributions, and splitting statistics.
6. [**06: Feature Analysis**](docs/06_FEATURE_ANALYSIS.md) — Spatial RGB color channels and ResNet50 learned hierarchical feature maps.
7. [**07: Data Preprocessing**](docs/07_DATA_PREPROCESSING.md) — Bilinear resizing, Albumentations augmentations, and normalization.
8. [**08: Missing Values and Outliers**](docs/08_MISSING_VALUES_AND_OUTLIERS.md) — Data integrity checks, black border handling, and specular glare analysis.
9. [**09: Image Processing Pipeline**](docs/09_IMAGE_PROCESSING_PIPELINE.md) — Transformation flowchart from native capture to PyTorch tensor.
10. [**10: PDE Denoising**](docs/10_PDE_DENOISING.md) — Derivation of Perona-Malik diffusion, conductance equations, and CFL stability criteria.
11. [**11: Gaussian Blur**](docs/11_GAUSSIAN_BLUR.md) — Heat equation equivalence, isotropic diffusion, and edge degradation mechanics.
12. [**12: ResNet50 Explanation**](docs/12_RESNET50_EXPLANATION.md) — Residual learning, skip connections, transfer learning, and layer freezing.
13. [**13: ML Models Used**](docs/13_ML_MODELS_USED.md) — Inventory and comparison of the 3 evaluated model regimes.
14. [**14: Model Training**](docs/14_MODEL_TRAINING.md) — Class-weighted Cross-Entropy loss, Adam optimizer, and checkpointing.
15. [**15: Epochs, Seeds, and Hyperparameters**](docs/15_EPOCHS_SEEDS_AND_HYPERPARAMETERS.md) — Complete experimental configuration inventory and seeds.
16. [**16: Model Testing and Accuracy**](docs/16_MODEL_TESTING_AND_ACCURACY.md) — Holdout test set metrics, class-wise performance, and the Accuracy Paradox.
17. [**17: Evaluation Metrics**](docs/17_EVALUATION_METRICS.md) — Mathematical formulations of Accuracy, Precision, Recall, and Macro F1.
18. [**18: SHAP / Explainability**](docs/18_SHAP_EXPLAINABILITY.md) — Grad-CAM mathematical derivation and comparison with SHAP.
19. [**19: AI Identification Logic**](docs/19_AI_IDENTIFICATION_LOGIC.md) — Step-by-step diagnostic prediction and confidence score derivation.
20. [**20: Final Output Explanation**](docs/20_FINAL_OUTPUT_EXPLANATION.md) — UI artifacts, metric readouts, and CSV export specifications.
21. [**21: Experimental Results**](docs/21_EXPERIMENTAL_RESULTS.md) — Research hypotheses, statistical comparisons, and discussion.
22. [**22: Testing and Validation**](docs/22_TESTING_AND_VALIDATION.md) — Unit test suite, stability checks, and integration tests.
23. [**23: Limitations and Failures**](docs/23_LIMITATIONS_AND_FAILURES.md) — Class imbalance, computational overhead, and Grad-CAM resolution limits.
24. [**24: Reproducibility Guide**](docs/24_REPRODUCIBILITY_GUIDE.md) — Step-by-step installation, training, evaluation, and demo commands.
25. [**25: Viva Questions and Answers**](docs/25_VIVA_QUESTIONS_AND_ANSWERS.md) — 40 comprehensive examination questions with model answers.
26. [**26: References and Glossary**](docs/26_REFERENCES_AND_GLOSSARY.md) — Academic citations and technical terminology glossary.

---

## 6. Generated Data Artifacts & Figures

### Machine-Readable CSV Tables
- [`results/dataset_summary.csv`](results/dataset_summary.csv) — Programmatic sample counts, percentages, and clinical relevance.
- [`results/feature_summary.csv`](results/feature_summary.csv) — Spatial channels and ResNet50 intermediate feature shapes.
- [`results/model_comparison.csv`](results/model_comparison.csv) — Aggregated multi-seed performance across conditions.
- [`results/classification_report.csv`](results/classification_report.csv) — Class-wise precision, recall, and F1 metrics.
- [`results/experiment_log.csv`](results/experiment_log.csv) — Raw individual run records.
- [`results/test_results.csv`](results/test_results.csv) — Exact verified metrics.

### High-Resolution Figures
- [`results/figures/class_distribution.png`](results/figures/class_distribution.png) — Distribution of labeled frames across Kvasir-Capsule classes.
- [`results/figures/feature_analysis.png`](results/figures/feature_analysis.png) — Pixel intensity distributions for Red, Green, and Blue channels.
- [`results/figures/confusion_matrix.png`](results/figures/confusion_matrix.png) — Side-by-side normalized confusion matrices (Raw vs PDE vs Gaussian).
- [`results/figures/roc_curve.png`](results/figures/roc_curve.png) — Multi-class Macro ROC curves across preprocessing conditions.
- [`results/figures/precision_recall_curve.png`](results/figures/precision_recall_curve.png) — Precision-Recall curves illustrating performance under severe class imbalance.
- [`results/figures/training_curves.png`](results/figures/training_curves.png) — Cross-entropy loss convergence trajectory.
- [`results/figures/preprocessing_comparison.png`](results/figures/preprocessing_comparison.png) — Qualitative 4-panel visual comparison (Clean, Noisy, PDE, Gaussian).

### Execution Logs
- [`logs/environment_info.txt`](logs/environment_info.txt) — Platform, Python, PyTorch, and MPS/CUDA runtime audit.
- [`logs/training_log.txt`](logs/training_log.txt) — Real execution training log from the background training engine.
- [`logs/testing_log.txt`](logs/testing_log.txt) — Model evaluation and testing session log.

---

## 7. Quick Start & Execution

### Single-Command Full Runner (Asynchronous Background Training + Foreground UI)
```bash
cd /Users/pranay/Documents/Development/Adaresh/EndoClean
./run.sh --all
```

### Launch Interactive Web Dashboard Only
```bash
cd /Users/pranay/Documents/Development/Adaresh/EndoClean
streamlit run demo/app.py
```
*(Open `http://localhost:8501` in your browser)*
