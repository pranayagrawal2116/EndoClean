# 25: Master Viva Examination Guide (40 Comprehensive Q&As)

---

### Category 1: Project Overview & Clinical Foundations

#### Q1: What is the primary objective of Project EndoClean?
**Answer:** EndoClean investigates whether non-linear Partial Differential Equation (PDE) anisotropic diffusion (the Perona-Malik model) improves the classification accuracy and explainability of Convolutional Neural Networks (CNNs) in Wireless Capsule Endoscopy (WCE), compared against raw images and standard isotropic Gaussian blur.

#### Q2: What is Wireless Capsule Endoscopy (WCE) and why is automated AI diagnosis needed?
**Answer:** WCE is an ingestible pill camera recording 50,000+ images as it travels along the GI tract. Manual review takes 45–120 minutes per patient, causing physician fatigue and an error rate up to 20% on subtle lesions. AI automates screening to highlight abnormal frames.

#### Q3: What visual noise artifacts contaminate capsule endoscopy images?
**Answer:** Intestinal chyme, bile, digestive fluid bubbles, specular reflections from LEDs, and motion blur from peristaltic tumbling.

#### Q4: Why is standard Gaussian blur harmful to medical image classification?
**Answer:** Gaussian blur is isotropic (diffuses equally in all directions). It smooths out sensor noise but simultaneously blurs high-contrast lesion boundaries (ulcer borders, vascular ectasias), eroding the high-frequency features CNNs rely on.

#### Q5: What 5 diagnostic classes are evaluated in EndoClean?
**Answer:** `Normal`, `Angiectasia`, `Ulcer`, `Blood`, and `Erosion`.

---

### Category 2: Mathematical Foundations of PDE Denoising

#### Q6: State the continuous Perona-Malik anisotropic diffusion equation.
**Answer:** $\frac{\partial I}{\partial t} = \text{div}(c(\|\nabla I\|) \nabla I) = c(\|\nabla I\|) \Delta I + \nabla c \cdot \nabla I$, where $c(\|\nabla I\|)$ is the gradient-dependent conductance coefficient.

#### Q7: How does the conductance coefficient $c(\|\nabla I\|)$ function?
**Answer:** When the gradient $\|\nabla I\|$ is low (flat mucosal tissue), $c \approx 1$, allowing maximum smoothing. When $\|\nabla I\|$ is high (lesion edges), $c \to 0$, stopping diffusion across the boundary and preserving the edge.

#### Q8: What are the two classic Perona-Malik conductance options implemented?
**Answer:**
1. Exponential: $c(\|\nabla I\|) = \exp(-(\|\nabla I\|/\kappa)^2)$ (sharp edge stopping).
2. Quadratic/Cauchy: $c(\|\nabla I\|) = \frac{1}{1 + (\|\nabla I\|/\kappa)^2}$ (wider edge preservation).

#### Q9: What is $\kappa$ (Kappa) and how was it selected?
**Answer:** $\kappa$ is the gradient threshold separating noise from edges. We selected $\kappa = 30.0$ via grid search in `src/tune_denoising.py` based on Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM).

#### Q10: What is the numerical stability limit for explicit 2D Euler diffusion?
**Answer:** The Courant-Friedrichs-Lewy (CFL) condition dictates that the time-step $\lambda \le 0.25$. EndoClean sets $\lambda = 0.15$, ensuring stable convergence without numeric overflow.

---

### Category 3: Dataset Rigor & Leakage Prevention

#### Q11: What dataset is used and what is its sample size?
**Answer:** The Kvasir-Capsule benchmark (47,238 total images, 43 unique patient videos). Our 5-class target population contains 37,010 images.

#### Q12: What is patient-wise data leakage and why is it dangerous?
**Answer:** Sequential frames from the same video share identical patient anatomy, lighting, and camera angle. A naive random split places frames from the same patient in both train and test sets, causing the model to memorize the patient rather than general pathology.

#### Q13: How did you mathematically prevent patient-wise leakage?
**Answer:** Using `GroupShuffleSplit` on `video_id` in `src/dataset.py`. All frames from any patient video belong exclusively to Train, Val, or Test.

#### Q14: What is the class distribution in the target dataset?
**Answer:** Normal (34,338; 92.8%), Angiectasia (866; 2.3%), Ulcer (854; 2.3%), Erosion (506; 1.4%), Blood (446; 1.2%).

#### Q15: What is the "Accuracy Paradox"?
**Answer:** In imbalanced datasets, a naive model that predicts "Normal" on every frame achieves 92.8% accuracy while failing to detect a single lesion. Accuracy is therefore a misleading metric.

---

### Category 4: Deep Learning & Transfer Learning

#### Q16: Why did you use ResNet50?
**Answer:** ResNet50 has 50 layers with residual skip connections ($y = \mathcal{F}(x) + x$) that eliminate vanishing gradients. Pre-training on ImageNet provides rich foundational visual feature extractors.

#### Q17: Which layers of ResNet50 are frozen and which are fine-tuned?
**Answer:** `conv1` through `layer2` are frozen (`requires_grad = False`). `layer3`, `layer4`, and the custom classification head are fine-tuned.

#### Q18: What is the structure of your classification head?
**Answer:** `model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(2048, 5))`.

#### Q19: How did you handle class imbalance during training?
**Answer:** Class-weighted Cross-Entropy loss using normalized inverse-frequency weights: $W_c = (1/N_c) / \sum (1/N_k)$.

#### Q20: What optimizer and learning rate did you select?
**Answer:** Adam optimizer with learning rate $\alpha = 1\times 10^{-4}$ ($0.0001$).

---

### Category 5: Explainability & Grad-CAM

#### Q21: What is Grad-CAM and how does it work?
**Answer:** Gradient-weighted Class Activation Mapping computes the gradients of the target class score with respect to the feature activation maps of `layer4[-1]`, globally average pools them into importance weights $\alpha_k^c$, and applies a ReLU-filtered weighted sum.

#### Q22: Why attach Grad-CAM to `layer4[-1]`?
**Answer:** `layer4[-1]` is the final convolutional layer (2048 channels, $7\times 7$ spatial grid), retaining both high-level semantic pathology features and 2D spatial coordinates.

#### Q23: What did Grad-CAM reveal about raw versus PDE-denoised images?
**Answer:** Raw models scatter attention across peripheral lens glare and bubbles. PDE models produce tight, contiguous activation hotspots focused directly over the true lesion margins.

#### Q24: Why not use SHAP instead of Grad-CAM for this project?
**Answer:** SHAP on raw image pixels ($224\times 224\times 3 = 150,528$ features) requires thousands of coalition sampling evaluations per image, making it computationally intractable for video endoscopy. Grad-CAM executes in a single backward pass.

---

### Category 6: Experimental Results & Analysis

#### Q25: What was the verified test accuracy across the three conditions?
**Answer:** Raw: $91.79\% \pm 3.38\%$; PDE: $90.33\% \pm 2.40\%$; Gaussian: $72.66\%$.

#### Q26: Why did Gaussian blur drop accuracy so severely (72.66%)?
**Answer:** Gaussian blur erased high-frequency edge gradients at mucosal boundaries, blinding the CNN to subtle ulcers and erosions.

#### Q27: Why did Raw have slightly higher accuracy than PDE if PDE is better?
**Answer:** Raw accuracy is inflated by correctly predicting majority Normal frames (Accuracy Paradox). Looking at Grad-CAM, the raw model relies partly on background artifacts, which fails in out-of-distribution clinical settings.

#### Q28: What is Macro Recall and why is it vital for medicine?
**Answer:** Macro Recall is the unweighted average sensitivity across all classes. In medicine, False Negatives (missing a bleeding ulcer) are far more dangerous than False Positives.

#### Q29: What is the primary evaluation metric of this project?
**Answer:** Macro F1-Score, which harmonic-averages precision and recall across all classes equally without majority class bias.

---

### Category 7: System Architecture & Deployment

#### Q30: How does `run.sh --all` execute training without freezing the dashboard?
**Answer:** It spawns `./run_experiments.sh` inside a detached background subshell (`nohup ... &`), writes the PID to `results/training.pid`, redirects stdout/stderr to `results/training.log`, and immediately launches Streamlit in the foreground.

#### Q31: How does the Streamlit UI interact with the background process?
**Answer:** It checks if `results/training.pid` is alive, displays an active status badge, provides an expandable log viewer with live updates, and offers a cache-reload button.

#### Q32: What hardware acceleration does the code support?
**Answer:** Apple Silicon Metal Performance Shaders (`mps`), NVIDIA CUDA (`cuda`), and CPU fallback.

---

### Category 8: Edge Cases, Limitations & Future Scope

#### Q33: How does the code handle single-channel grayscale inputs?
**Answer:** `_perona_malik_channel` operates on 2D arrays directly, while `perona_malik_diffusion` splits 3D arrays by channel, applying diffusion independently to Red, Green, and Blue.

#### Q34: What is the time complexity of the Perona-Malik solver?
**Answer:** $O(N \times \text{num\_iter} \times C)$ where $N$ is pixel count (50,176), $\text{num\_iter} = 15$, and $C = 3$.

#### Q35: What is the primary limitation of your current implementation?
**Answer:** The numerical PDE solver runs on the CPU. For real-time streaming at 6 fps, it must be ported to GPU tensor kernels or implemented as a Neural PDE.

#### Q36: Why use Albumentations instead of torchvision transforms?
**Answer:** Albumentations is significantly faster (written in C++/NumPy/OpenCV) and provides seamless simultaneous transformations of images and spatial masks.

#### Q37: How would Vision Transformers (ViT) perform compared to ResNet50?
**Answer:** ViTs lack inductive bias for local pixel neighborhoods and rely on patch self-attention. PDE denoising could be even more beneficial for ViTs by removing spurious high-frequency patch tokens.

#### Q38: Why was dropout set to 0.5?
**Answer:** Medical datasets have limited diversity. A high dropout rate ($p=0.5$) prevents co-adaptation of hidden features in the classification head.

#### Q39: What is the effect of changing $\kappa$ from 30 to 100?
**Answer:** A higher $\kappa$ treats almost all gradients as noise, leading to over-smoothing and blurring actual lesions (approaching Gaussian blur).

#### Q40: What is your project's main contribution to medical AI?
**Answer:** Demonstrating that PDE-based anisotropic diffusion serves as a non-linear structural filter that cleans endoscopy images while preserving pathological edges, resulting in grounded, clinically verifiable visual explainability.
