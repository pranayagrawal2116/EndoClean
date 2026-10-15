# 23: Limitations and Failure Modes

## 1. Observed Limitations
1. **Computational Overhead of Numerical PDEs:** Iterative finite-difference diffusion across 15 iterations on CPU is slower than single-pass linear convolution.
2. **Severe Class Imbalance:** Normal frames represent 92.8% of the target dataset. Despite class weighting, rare classes (`Blood`, `Erosion`) have lower sensitivity.
3. **Grad-CAM Coarseness:** Because `layer4[-1]` has a spatial resolution of $7\times 7$, bilinear upsampling to $224\times 224$ produces smooth heatmap blobs rather than pixel-perfect segmentation masks.
4. **Single-Epoch Smoke Test:** The fast test run was executed for 1 epoch; multi-epoch training is required for maximum convergence.
