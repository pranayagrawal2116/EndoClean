# 07: Data Preprocessing Pipeline

This document details all preprocessing operations applied to input images prior to model consumption.

---

## 1. Preprocessing Stages & Transformations

| Step | Operation | Parameters | Rationale & Clinical Impact |
| :--- | :--- | :--- | :--- |
| **1. Resizing** | Bilinear Interpolation | $224\times 224$ pixels | Standardizes variable endoscopy aspect ratios to ResNet50 input requirements. |
| **2. Channel Conversion** | BGR $\to$ RGB | `cv2.COLOR_BGR2RGB` | Corrects OpenCV default BGR ordering to standard RGB color space. |
| **3. Flips** | Horizontal & Vertical Flips | Probability $p = 0.5$ | Models tumbling capsule camera orientations (no fixed 'up' or 'down' in lumen). |
| **4. Rotations** | Random 90° Rotations | Probability $p = 0.5$ | Invariance to peristaltic camera rolling along intestinal axis. |
| **5. Color Jitter** | Brightness, Contrast, Saturation | Factor 0.2, $p = 0.5$ | Simulates variable battery voltage, LED decay, and digestive fluid opacity. |
| **6. Normalization** | Z-score scaling | Mean: (0.485, 0.456, 0.406), Std: (0.229, 0.224, 0.225) | Scales pixel tensors to match ImageNet pre-training distributions for stable gradient flow. |
| **7. Tensor Conversion** | PyTorch Tensor Formatting | `ToTensorV2()` | Transposes image dimensions from $(H, W, C)$ to PyTorch standard $(C, H, W)$. |

---

## 2. Prevention of Preprocessing Leakage
- Fitted statistics (ImageNet normalization) are fixed constants, not recomputed from the test set.
- Online spatial augmentations (flips, jitter) are strictly deactivated during validation and testing (`is_train=False`).
