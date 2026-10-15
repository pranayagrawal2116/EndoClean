# 24: Complete Reproducibility Guide

## 1. Environment Installation
```bash
cd /Users/pranay/Documents/Development/Adaresh/EndoClean
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Single-Command Execution
To launch the background training pipeline and the web dashboard simultaneously:
```bash
./run.sh --all
```

## 3. Manual Step-by-Step Reproduction
```bash
# 1. Structure Metadata
python3 prepare_metadata.py

# 2. Train Models
python3 src/train.py --data_dir data/raw --seed 42 --epochs 20
python3 src/train.py --data_dir data/pde_denoised --seed 42 --epochs 20
python3 src/train.py --data_dir data/gaussian_denoised --seed 42 --epochs 20

# 3. Evaluate Metrics & Export Visualizations
python3 src/evaluate.py --metadata_csv data/metadata.csv

# 4. Generate Grad-CAM Heatmaps
python3 src/gradcam.py --metadata_csv data/metadata.csv

# 5. Launch Interactive Dashboard
streamlit run demo/app.py
```
