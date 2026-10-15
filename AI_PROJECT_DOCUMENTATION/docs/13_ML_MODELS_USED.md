# 13: Machine Learning Models Inventory

EndoClean evaluates a $3 \times 3$ experimental matrix comprising 3 distinct model regimes across multiple random seeds:

| Model Identifier | Input Preprocessing | Architecture | Objective & Role |
| :--- | :--- | :--- | :--- |
| **Model 1: Raw ResNet50** | Unprocessed raw images | ResNet50 + 5-Class Head | Control baseline representing standard training without preprocessing. |
| **Model 2: Gaussian ResNet50** | Gaussian blur ($5\times 5, \sigma=1.0$) | ResNet50 + 5-Class Head | Comparative baseline testing impact of linear isotropic edge blurring. |
| **Model 3: PDE ResNet50** | Perona-Malik diffusion ($\kappa=30, \text{iter}=15$) | ResNet50 + 5-Class Head | Proposed model testing non-linear edge-preserving denoising. |

All three models share identical network weights initialization (ImageNet-1K), optimizer parameters, batch sizes, loss functions, and evaluation splits to ensure strict scientific comparability.
