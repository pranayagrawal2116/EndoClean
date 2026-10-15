# 26: References and Glossary

## 1. Academic & Scientific References
1. **Perona-Malik Diffusion:** Perona, P., & Malik, J. (1990). *Scale-space and edge detection using anisotropic diffusion.* IEEE Transactions on Pattern Analysis and Machine Intelligence, 12(7), 629-639.
2. **ResNet Architecture:** He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition.* Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770-778.
3. **Grad-CAM Explainability:** Selvaraju, R. R., Cogswell, M., Das, A., Vedaldi, A., Parikh, D., & Batra, D. (2017). *Grad-CAM: Visual explanations from deep networks via gradient-based localization.* IEEE International Conference on Computer Vision (ICCV), 618-626.
4. **Kvasir-Capsule Benchmark:** Smedsrud, P. H., Thambawita, V., Hicks, S. A., Gjestang, H., Nedrejord, O. O., Næss, E., ... & Halvorsen, P. (2021). *Kvasir-Capsule, a video dataset for development of artificial intelligence for wireless capsule endoscopy.* Scientific Data, 8(1), 142.
5. **Albumentations Library:** Buslaev, A., Iglovikov, V. I., Khvedchenya, E., Parinov, A., Druzhinin, M., & Kalinin, A. A. (2020). *Albumentations: fast and flexible image augmentations.* Information, 11(2), 125.

---

## 2. Technical Glossary
- **Anisotropic Diffusion:** A non-linear image smoothing technique where diffusion rate varies spatially depending on gradient magnitudes, preserving edges.
- **Isotropic Diffusion:** Uniform diffusion in all directions (mathematically equivalent to the Gaussian heat equation).
- **Courant-Friedrichs-Lewy (CFL) Condition:** The numerical stability limit for explicit finite-difference time integration schemes ($\lambda \le 0.25$ on 2D square lattices).
- **Grad-CAM:** Gradient-weighted Class Activation Mapping; visualizes model attention using the gradients flowing into the final convolutional layer.
- **GroupShuffleSplit:** A cross-validation splitting strategy ensuring that all samples sharing the same group identifier (`video_id`) remain together in either train or test splits.
- **Macro F1-Score:** The unweighted average of F1-scores computed individually for each class, treating rare classes equally with majority classes.
- **PSNR (Peak Signal-to-Noise Ratio):** Logarithmic ratio between maximum possible signal power and corrupting noise power.
- **SSIM (Structural Similarity Index):** Perception-based metric measuring structural, luminance, and contrast similarity between images.
