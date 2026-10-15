# 15: Epochs, Random Seeds, and Hyperparameters

## 1. Configuration Inventory

| Hyperparameter | Value in Codebase | Source File | Status / Verification |
| :--- | :--- | :--- | :--- |
| **Random Seeds** | 42, 43, 44 | `run_experiments.sh` | Verified in test logs |
| **Target Epochs (Full)** | 20 epochs | `README.md` | Configured full experiment |
| **Executed Epochs (Fast Run)**| 1 epoch | `run_experiments.sh` line 18 | Verified in `training.log` |
| **Learning Rate** | $1\times 10^{-4}$ ($0.0001$) | `src/train.py` | Verified in code |
| **Batch Size** | 32 | `src/dataset.py` line 105 | Verified in code |
| **Dropout Probability** | 0.5 | `src/train.py` line 37 | Verified in code |
| **PDE Conduction Threshold ($\kappa$)** | 30.0 | `src/denoising.py` line 9 | Verified in code |
| **PDE Iterations ($t$)** | 15 iterations | `src/denoising.py` line 9 | Verified in code |
| **PDE Time-Step ($\lambda$)** | 0.15 | `src/denoising.py` line 10 | Verified in code |
| **Gaussian Kernel Size** | $5\times 5$ | `src/denoising.py` line 72 | Verified in code |
| **Gaussian Sigma ($\sigma$)** | 1.0 | `src/denoising.py` line 72 | Verified in code |

---

## 2. Reproduction Commands
To run the fast verification run:
```bash
./run_experiments.sh
```
To run full 20-epoch multi-seed training:
```bash
python3 src/train.py --data_dir data/pde_denoised --metadata_csv data/metadata.csv --seed 42 --epochs 20
```
