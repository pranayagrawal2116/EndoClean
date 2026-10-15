# 19: AI Identification & Decision Pathway

This document tracks the step-by-step transformation of an uploaded endoscopy image into a clinical diagnosis.

## 1. Decision Pipeline
1. **Ingestion:** User uploads image ($336\times 336$ RGB).
2. **PDE Diffusion:** 15 iterations of Perona-Malik diffusion suppress fluid bubbles and specular glare while preserving lesion margins.
3. **Tensor Preprocessing:** Image resized to $224\times 224$, normalized via ImageNet constants, converted to tensor $(1, 3, 224, 224)$.
4. **ResNet50 Inference:** Forward pass outputs 5 raw logits $[z_1, z_2, z_3, z_4, z_5]$.
5. **Softmax Probabilities:**
   $$P(c) = \frac{\exp(z_c)}{\sum_{j=1}^5 \exp(z_j)}$$
6. **Class Decision:** $\hat{c} = \arg\max_c P(c)$
7. **Grad-CAM Synthesis:** Gradients backpropagated from $\hat{c}$ through `layer4[-1]` to synthesize spatial heatmap.
8. **Comparative Analysis:** Console generates diagnostic explanation comparing raw vs PDE attention boundaries.
