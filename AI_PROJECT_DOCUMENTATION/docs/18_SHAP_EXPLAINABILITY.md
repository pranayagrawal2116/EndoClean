# 18: Explainability Analysis: Grad-CAM vs. SHAP

## 1. Audit of Explainability in EndoClean
An inspection of the codebase confirms that EndoClean implements **Gradient-weighted Class Activation Mapping (Grad-CAM)** via `pytorch_grad_cam` on ResNet50's final convolutional layer (`layer4[-1]`).

---

## 2. Mathematical Formulation of Grad-CAM
Grad-CAM calculates the gradient of the class score $Y^c$ with respect to feature activation maps $A^k$:

$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial Y^c}{\partial A_{i,j}^k}$$

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

The ReLU filter ensures only features that positively correlate with the predicted diagnosis are visualized.

---

## 3. Comparison with SHAP (SHapley Additive exPlanations)
- **Why Grad-CAM was Chosen:** Convolutional feature maps maintain 2D spatial topography ($7\times 7$). Grad-CAM backpropagates in a single backward pass ($O(1)$ forward/backward passes).
- **Why SHAP is Challenging on Images:** SHAP (e.g. KernelSHAP) treats every pixel ($224\times 224\times 3 = 150,528$ dimensions) as a coalition member, requiring thousands of model evaluations per image. In real-time capsule streaming, SHAP is computationally intractable.
- **Empirical Grad-CAM Finding:** Grad-CAM reveals that raw models scatter attention across bubbles and dark edges, whereas PDE models concentrate activation tightly around true mucosal lesions.
