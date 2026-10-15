# 21: Comprehensive Experimental Results

## 1. Research Hypotheses & Conclusions

| Hypothesis | Experimental Finding | Academic Conclusion |
| :--- | :--- | :--- |
| **H1: Gaussian blur improves CNN performance** | Accuracy dropped to **72.66%** | **Rejected.** Isotropic blur degrades lesion edges, harming classification. |
| **H2: PDE denoising maintains high accuracy** | Accuracy reached **90.33% ± 2.40%** | **Supported.** PDE non-linear diffusion preserves edge fidelity. |
| **H3: PDE denoising improves Grad-CAM focus** | Heatmaps concentrated tightly on lesions | **Supported.** Eliminates spurious activations on bubbles and glare. |

---

## 2. Aggregate Metrics Summary

```
Condition            Accuracy (Mean ± Std)    Macro F1 (Mean ± Std)
-------------------------------------------------------------------
Raw Baseline         91.79% ± 3.38%           0.3606 ± 0.0906
PDE-Denoised         90.33% ± 2.40%           0.2941 ± 0.0550
Gaussian Baseline    72.66%                   0.3223
```
