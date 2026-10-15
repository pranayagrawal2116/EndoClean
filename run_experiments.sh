#!/bin/bash

# Exit on error
set -e

echo "Starting EndoClean Experiments..."

SEEDS=(42)
CONDITIONS=("raw" "pde_denoised" "gaussian_denoised")

# Train all 9 models
for seed in "${SEEDS[@]}"; do
    for condition in "${CONDITIONS[@]}"; do
        echo "=========================================================="
        echo "Training Condition: $condition | Seed: $seed"
        echo "=========================================================="
        
        python3 src/train.py --data_dir "data/$condition" --metadata_csv "data/metadata.csv" --seed "$seed" --epochs 1
    done
done

echo "=========================================================="
echo "Evaluating all models and aggregating results..."
echo "=========================================================="
python3 src/evaluate.py --metadata_csv "data/metadata.csv"

echo "=========================================================="
echo "Generating Grad-CAM Explainability Visualizations..."
echo "=========================================================="
python3 src/gradcam.py --metadata_csv "data/metadata.csv"

echo "Experiments completed successfully!"
