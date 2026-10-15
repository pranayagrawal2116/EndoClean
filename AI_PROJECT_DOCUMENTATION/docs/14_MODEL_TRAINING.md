# 14: Model Training Configuration & Convergence

## 1. Training Parameters
- **Loss Function:** Class-Weighted Cross-Entropy Loss (`nn.CrossEntropyLoss(weight=weights)`)
- **Optimizer:** Adam (`lr=1e-4`, betas=(0.9, 0.999), weight_decay=0.0)
- **Batch Size:** 32 images per batch
- **Input Dimensions:** $3\times 224\times 224$
- **Hardware Platform:** Apple Silicon MPS (Metal Performance Shaders) / CUDA
- **Checkpointing Policy:** Checkpoint `best_model.pth` saved whenever validation Macro F1 improves.

---

## 2. Loss Weighting Formula
Given class counts $N_c$ in the training split:
$$w_c = \frac{1}{N_c}, \quad W_c = \frac{w_c}{\sum_{k=1}^5 w_k}$$

Errors on rare classes (`Blood`, `Angiectasia`) penalize the optimizer $\approx 75\times$ more than errors on `Normal` frames.
