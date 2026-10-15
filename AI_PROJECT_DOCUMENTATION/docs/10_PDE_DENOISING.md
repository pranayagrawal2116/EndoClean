# 10: Partial Differential Equation (PDE) Image Denoising

## 1. Mathematical Formulation
Perona-Malik non-linear anisotropic diffusion is governed by:

$$\frac{\partial I}{\partial t} = \text{div}(c(\|\nabla I\|) \nabla I) = c(\|\nabla I\|) \Delta I + \nabla c \cdot \nabla I$$

Where $c(\|\nabla I\|)$ is the conductance coefficient function.

### Conductance Formulations:
1. **Exponential Form (Option 1):**
   $$c(\|\nabla I\|) = \exp\left(-\left(\frac{\|\nabla I\|}{\kappa}\right)^2\right)$$
   Privileges high-contrast edges; sharper edge stopping.

2. **Quadratic / Cauchy Form (Option 2):**
   $$c(\|\nabla I\|) = \frac{1}{1 + \left(\frac{\|\nabla I\|}{\kappa}\right)^2}$$
   More gradual decay across wider gradient ranges.

---

## 2. Discrete 2D Implementation
In `src/denoising.py`, continuous derivatives are discretized on a 2D 4-neighborhood:
$$\nabla_N I_{i,j} = I_{i-1,j} - I_{i,j}, \quad \nabla_S I_{i,j} = I_{i+1,j} - I_{i,j}$$
$$\nabla_E I_{i,j} = I_{i,j+1} - I_{i,j}, \quad \nabla_W I_{i,j} = I_{i,j-1} - I_{i,j}$$

The discrete iterative update is:
$$I_{i,j}^{t+1} = I_{i,j}^t + \lambda \sum_{d \in \{N,S,E,W\}} c_d \cdot \nabla_d I$$

### CFL Numerical Stability:
For 2D explicit Euler diffusion, stability requires $\lambda \le 0.25$. EndoClean enforces $\lambda = 0.15$, guaranteeing absolute numerical convergence without overflow.
