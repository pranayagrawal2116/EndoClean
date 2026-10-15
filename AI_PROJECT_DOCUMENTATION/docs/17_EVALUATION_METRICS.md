# 17: Evaluation Metrics Guide

## 1. Mathematical Definitions

### 1. Accuracy
$$\text{Accuracy} = \frac{\sum_{c=1}^C TP_c}{\text{Total Samples}}$$
*Limitation:* Severely distorted by majority class dominance.

### 2. Macro Precision
$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Macro Precision} = \frac{1}{C} \sum_{c=1}^C \text{Precision}_c$$

### 3. Macro Recall (Clinical Sensitivity)
$$\text{Recall}_c = \frac{TP_c}{TP_c + FN_c}, \quad \text{Macro Recall} = \frac{1}{C} \sum_{c=1}^C \text{Recall}_c$$
*Clinical Importance:* Measures the percentage of actual diseases identified. Missing an ulcer ($FN$) has severe clinical consequences.

### 4. Macro F1-Score (Primary Metric)
$$\text{Macro F1} = \frac{1}{C} \sum_{c=1}^C \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$
Equal weighting across all classes regardless of sample size.
