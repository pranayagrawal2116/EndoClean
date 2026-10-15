# 01: Project Overview — EndoClean

## 1. Project Title & Executive Summary
**EndoClean: Capsule Endoscopy Image Denoising and Classification Pipeline via PDE-Based Anisotropic Diffusion and Deep Learning**

EndoClean is a research-grade medical computer vision system designed to investigate whether Partial Differential Equation (PDE)-based image denoising (specifically, the non-linear Perona-Malik anisotropic diffusion model) improves the diagnostic accuracy and visual explainability of Convolutional Neural Networks (CNNs) in Wireless Capsule Endoscopy (WCE), compared against raw un-denoised images and traditional isotropic linear Gaussian filtering.

---

## 2. Problem Statement
Wireless Capsule Endoscopy (WCE) is a non-invasive gastrointestinal procedure where a patient swallows an ingestible pill camera. Over an 8 to 10-hour transit, the device records **50,000 to 100,000 high-resolution frames**. 

Clinical utilization faces two profound bottlenecks:
1. **Physician Fatigue & Diagnostic Miss Rates:** Manually analyzing 50,000+ frames takes a gastroenterologist 45 to 120 minutes per patient. Prolonged cognitive fatigue yields human error rates up to 15-20% on transient, subtle lesions (e.g., small angiectasias, flat mucosal ulcers).
2. **Environmental Noise & Edge Degradation:** The gastrointestinal tract is filled with digestive chyme, bile, bubbles, specular camera reflections, and uneven peristaltic illumination. Standard linear smoothing filters (like Gaussian blur) eliminate high-frequency noise, but they **destroy pathology boundaries** (edges) by diffusing pixel intensities indiscriminately across anatomical margins.

---

## 3. Project Objectives
1. **Mathematical Preprocessing:** Implement and optimize non-linear Perona-Malik anisotropic diffusion to smooth intra-tissue noise while preserving critical diagnostic lesion boundaries.
2. **Controlled Comparative Grid:** Train and evaluate pre-trained ResNet50 classifiers across three strictly matched data regimes:
   - **Condition 1 (Raw):** Original sensor captures from Kvasir-Capsule.
   - **Condition 2 (Gaussian):** Linearly blurred baseline ($5\times 5$ kernel, $\sigma=1.0$).
   - **Condition 3 (PDE):** Non-linearly diffused images ($\kappa=30, \text{iterations}=15, \lambda=0.15$).
3. **Prevention of Medical AI Leakage:** Enforce video-wise / patient-wise stratified data partitioning using `GroupShuffleSplit` on `video_id`.
4. **Explainable AI (XAI):** Deploy Gradient-weighted Class Activation Mapping (Grad-CAM) to verify whether PDE preprocessing directs deep feature maps toward actual pathological morphology rather than peripheral noise artifacts.
5. **Interactive Diagnostic Console:** Deliver an asynchronous web dashboard (Streamlit) supporting real-time image upload, online PDE filtering, dual model inference, and background retraining.

---

## 4. Input and Output Specifications
- **Input:** Single-frame or batch RGB endoscopy images (JPEG/PNG format, native resolution $336\times 336$, resized to $224\times 224\times 3$).
- **Output:**
  - 5-class predicted diagnostic probability distribution: `Normal`, `Angiectasia`, `Ulcer`, `Blood`, `Erosion`.
  - Confidence percentage score.
  - High-resolution Grad-CAM spatial activation heatmaps overlaid on the input image.
  - Comparative diagnostic consensus analysis explaining model reasoning pathways.

---

## 5. Technology Stack & Frameworks
- **Programming Language:** Python 3.13.7
- **Deep Learning Framework:** PyTorch 2.14.0 (accelerated with Apple Silicon Metal Performance Shaders / MPS)
- **Computer Vision & Augmentation:** OpenCV 5.0.0, Albumentations 2.0.8, Scikit-Image 0.26.0
- **Data Science & Metrics:** Scikit-Learn 1.8.0, Pandas 3.0.0, NumPy 2.4.2
- **Visualization:** Matplotlib 3.10.8, Seaborn 0.13.2, PyTorch-Grad-CAM 1.5.7
- **Interactive UI:** Streamlit 1.59.2 with custom dark-mode CSS

---

## 6. Key Findings & Verified Results Summary
- **Verified Accuracy:** Raw: $91.79\% \pm 3.38\%$; PDE: $90.33\% \pm 2.40\%$; Gaussian: $72.66\%$.
- **The Accuracy Paradox:** Raw accuracy is artificially elevated by majority class dominance (>90% Normal).
- **Macro F1-Score & Localization:** PDE denoising maintains high Macro F1 and produces dramatically tighter, more localized Grad-CAM heatmaps centered on genuine lesion borders, whereas Gaussian blurring collapses accuracy to 72.66% due to edge erosion.
