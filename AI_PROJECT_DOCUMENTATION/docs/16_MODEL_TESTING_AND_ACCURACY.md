# 16: Model Testing & Verified Accuracy

## 1. Verified Holdout Test Results
Evaluated on unseen patient videos (`results/test_metrics_all_runs.csv`):

| Condition | Seed | Accuracy | Macro Precision | Macro Recall | Macro F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Raw** | 42 | 88.04% | 0.4260 | 0.4332 | **0.4246** |
| **Raw** | 43 | 95.54% | 0.3404 | 0.2887 | **0.2965** |
| **PDE Denoised** | 42 | 88.51% | 0.3531 | 0.3261 | **0.3330** |
| **PDE Denoised** | 43 | 92.15% | 0.2863 | 0.2448 | **0.2552** |
| **Gaussian Denoised**| 42 | 72.66% | 0.3201 | 0.4581 | **0.3223** |

---

## 2. Statistical Aggregation (Mean ± Std)
Aggregated results from `results/test_metrics_aggregated.csv`:
- **Raw Images:** Accuracy = $91.79\% \pm 3.38\%$, Macro F1 = $0.3606 \pm 0.0906$
- **PDE Denoised:** Accuracy = $90.33\% \pm 2.40\%$, Macro F1 = $0.2941 \pm 0.0550$
- **Gaussian Denoised:** Accuracy = $72.66\%$, Macro F1 = $0.3223$

---

## 3. The Accuracy Paradox in Endoscopy
In medical datasets with >90% Normal frames, high accuracy is trivially obtained by predicting Normal. The true measure of diagnostic efficacy is **Macro Recall and Macro F1**, which evaluates whether ulcers, erosions, and hemorrhages are correctly classified.
