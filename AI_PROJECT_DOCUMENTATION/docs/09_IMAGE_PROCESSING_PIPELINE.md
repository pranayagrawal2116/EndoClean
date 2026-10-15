# 09: Image Processing Pipeline

## 1. End-to-End Image Processing Flow
```mermaid
flowchart LR
    A[Raw Endoscopy Frame 336x336] --> B{Preprocessing Branch}
    B -->|Path 1| C[Raw Pass-Through]
    B -->|Path 2| D[Gaussian Blur Filter 5x5, σ=1.0]
    B -->|Path 3| E[Perona-Malik PDE Diffusion κ=30, iter=15]
    C & D & E --> F[Albumentations Pipeline]
    F --> G[Resize 224x224]
    G --> H[ImageNet Normalization]
    H --> I[PyTorch Tensor Bx3x224x224]
    I --> J[ResNet50 Backbone]
```

## 2. Quantitative Image Quality Metrics
Evaluating denoising on synthetic noise benchmarks (`results/tuning_results.csv`):
- **Unprocessed Noisy Input:** PSNR: 18.21 dB, SSIM: 0.3080
- **Gaussian Blurred ($5\times 5, \sigma=1.0$):** PSNR: 25.36 dB, SSIM: 0.6042
- **Perona-Malik PDE ($\kappa=30, \text{iter}=15$):** PSNR: 20.64 dB, SSIM: 0.5211

*Key Insight:* While Gaussian blur achieves higher raw PSNR by aggressively smoothing global pixel variances, it blurs lesion edges. Perona-Malik preserves sharp edge discontinuities that are vital for CNN classification.
