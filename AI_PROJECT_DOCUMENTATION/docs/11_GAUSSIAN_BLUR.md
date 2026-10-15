# 11: Gaussian Blur Filter Baseline

## 1. Mathematical Formulation
Linear isotropic Gaussian filtering is defined by the 2D Gaussian kernel:

$$G_\sigma(x, y) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2+y^2}{2\sigma^2}\right)$$

Filtering an image corresponds to convolving $I_0(x,y)$ with $G_\sigma$:
$$I(x,y,t) = I_0(x,y) * G_\sigma(x,y)$$

This is the exact analytical solution to the classical heat diffusion equation:
$$\frac{\partial I}{\partial t} = \Delta I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$$

---

## 2. Parameters & Downstream Impact
- **Implementation:** `cv2.GaussianBlur(image, (5, 5), sigmaX=1.0)`
- **The Degradation Mechanism:** Because diffusion speed is constant across all coordinates, edge boundaries between pathologies (red ulcers, vascular tufts) and healthy pink mucosa are smoothed away.
- **Empirical Proof:** ResNet50 trained on Gaussian-blurred images experienced a severe drop in classification accuracy to **72.66%** (compared to 90.33% for PDE and 91.79% for Raw).
