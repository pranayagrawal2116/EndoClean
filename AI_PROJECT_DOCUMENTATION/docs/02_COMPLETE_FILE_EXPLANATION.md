# 02: Complete File-by-File Architectural Explanation

This document details every script, configuration file, and entry point in the EndoClean project.

---

## 1. `src/denoising.py`
- **Location:** `EndoClean/src/denoising.py` (132 lines)
- **Subsystem:** Image Preprocessing / Mathematical Solvers
- **Purpose:** Implements vectorized Perona-Malik anisotropic diffusion and baseline OpenCV Gaussian blur.
- **Key Functions:**
  - `perona_malik_diffusion(image, num_iter=15, kappa=30.0, gamma=0.15, option=1)`: Dispatches single-channel or 3-channel RGB image arrays to the numerical diffusion engine. Enforces `np.clip(result, 0, 255).astype(np.uint8)`.
  - `_perona_malik_channel(image, num_iter, kappa, gamma, option)`: Core 2D numerical solver. Uses `np.diff` to compute directional gradients $\nabla_N, \nabla_S, \nabla_E, \nabla_W$, computes non-linear conductance coefficients ($c_N, c_S, c_E, c_W$), and updates pixel intensities via explicit Euler forward differences.
  - `gaussian_denoise(image, ksize=5, sigma=1.0)`: Standard OpenCV linear baseline filter (`cv2.GaussianBlur`).
  - `process_folder(input_dir, output_dir, method='pde', **kwargs)`: Batch folder processor that walks subdirectories and saves denoised files to disk.
- **Dependencies:** `cv2`, `numpy`, `tqdm`, `pathlib.Path`.

---

## 2. `src/tune_denoising.py`
- **Location:** `EndoClean/src/tune_denoising.py` (141 lines)
- **Subsystem:** Hyperparameter Optimization & Signal Verification
- **Purpose:** Executes empirical grid search over PDE and Gaussian parameters against synthetic noise benchmarks.
- **Key Functions:**
  - `add_synthetic_noise(image)`: Adds controlled Gaussian noise (var=0.01) and Salt-and-Pepper noise (amount=0.02).
  - `evaluate_pde(clean_img, noisy_img, kappas, iters)`: Grid search over $\kappa \in [10, 20, 30, 40, 50]$ and iterations $\in [5, 10, 15, 20, 30]$, recording PSNR and SSIM.
  - `evaluate_gaussian(clean_img, noisy_img, sigmas, ksizes)`: Grid search over $\sigma \in [0.5, 3.0]$ and kernel sizes $\in [3, 5, 7]$.
  - `main()`: Orchestrates execution, logs results to `results/tuning_results.csv`, and outputs `results/denoising_comparison.png`.
- **Dependencies:** `skimage.metrics.psnr`, `skimage.metrics.ssim`, `skimage.util.random_noise`, `itertools.product`.

---

## 3. `src/dataset.py`
- **Location:** `EndoClean/src/dataset.py` (126 lines)
- **Subsystem:** Data Ingestion, Patient-Wise Splitting & Augmentation
- **Purpose:** Prevents patient-wise data leakage, applies Albumentations transformations, and provides PyTorch DataLoaders with inverse-frequency class weights.
- **Key Functions / Classes:**
  - `TARGET_CLASSES`: List defining `['Normal', 'Angiectasia', 'Ulcer', 'Blood', 'Erosion']`.
  - `get_transforms(is_train)`: Returns Albumentations pipeline (224x224 resize, flips, rotations, color jitter, ImageNet normalization).
  - `create_patient_split(metadata_path, val_size=0.15, test_size=0.15, random_state=42)`: Implements `GroupShuffleSplit` on `video_id` to strictly isolate patient videos between Train (70%), Val (15%), and Test (15%).
  - `EndoCleanDataset(Dataset)`: PyTorch Dataset mapping image files to RGB arrays and class indices.
  - `get_dataloaders(...)`: Builds train/val/test DataLoaders and computes normalized class loss weights $W_c = w_c / \sum w_k$.
- **Dependencies:** `torch.utils.data`, `sklearn.model_selection.GroupShuffleSplit`, `albumentations`, `pandas`.

---

## 4. `src/train.py`
- **Location:** `EndoClean/src/train.py` (163 lines)
- **Subsystem:** Model Construction & Training Loop
- **Purpose:** Constructs ResNet50 transfer learning architecture, freezes early layers, and trains with class-weighted loss.
- **Key Functions:**
  - `set_seed(seed)`: Sets deterministic seeds for `torch`, `numpy`, and Python random.
  - `build_model(num_classes)`: Loads pretrained ResNet50, freezes `conv1` through `layer2`, and attaches `nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, 5))`.
  - `train_epoch()`: Executes forward pass, weighted cross-entropy loss, backpropagation, and Adam optimizer step.
  - `evaluate()`: Evaluates model on validation loader and calculates Macro F1-score.
  - `main()`: Hardware detection (CUDA/MPS/CPU), early stopping checkpointing saving `best_model.pth`.
- **Dependencies:** `torch`, `torchvision.models`, `sklearn.metrics.f1_score`.

---

## 5. `src/evaluate.py`
- **Location:** `EndoClean/src/evaluate.py` (150 lines)
- **Subsystem:** Independent Model Evaluation & Aggregation
- **Purpose:** Tests all trained checkpoints across data conditions and seeds, generating aggregate CSVs and confusion matrices.
- **Key Functions:**
  - `evaluate_model()`: Computes Accuracy, Macro Precision, Macro Recall, Macro F1, and normalized Confusion Matrix on the holdout test set.
  - `main()`: Aggregates results across seeds, exports `test_metrics_all_runs.csv` and `test_metrics_aggregated.csv`, and plots bar charts and normalized heatmaps.
- **Dependencies:** `torch`, `sklearn.metrics`, `seaborn`, `matplotlib`.

---

## 6. `src/gradcam.py`
- **Location:** `EndoClean/src/gradcam.py` (146 lines)
- **Subsystem:** Model Explainability & Interpretability
- **Purpose:** Extracts gradient activation maps from ResNet50's final convolutional layer (`layer4[-1]`).
- **Key Functions:**
  - `get_gradcam(model, image_tensor, target_class_idx)`: Computes normalized 2D Grad-CAM heatmap mask.
  - `main()`: Generates side-by-side 4-panel visual comparisons (Raw Image, Raw Grad-CAM, PDE Image, PDE Grad-CAM) saved to `results/gradcam/`.
- **Dependencies:** `pytorch_grad_cam.GradCAM`, `pytorch_grad_cam.utils.image.show_cam_on_image`.

---

## 7. `demo/app.py`
- **Location:** `EndoClean/demo/app.py` (384 lines)
- **Subsystem:** Interactive Web UI / Clinical Dashboard
- **Purpose:** Streamlit interface providing live image uploading, real-time PDE diffusion, dual ResNet50 inference, Grad-CAM overlays, background training log monitoring, and comparative diagnostic analysis.
- **Dependencies:** `streamlit`, `torch`, `cv2`, `PIL`, `albumentations`.

---

## 8. Orchestration Scripts: `run.sh` & `run_experiments.sh`
- **`run.sh`:** Master executable that handles single-command execution. Dispatches training to a detached background subshell (`nohup ... &`) when `--all` is passed, records PID in `results/training.pid`, streams logs to `results/training.log`, and launches Streamlit in the foreground.
- **`run_experiments.sh`:** Bash loop iterating across conditions (`raw`, `pde_denoised`, `gaussian_denoised`) and random seeds, executing `train.py`, `evaluate.py`, and `gradcam.py`.
