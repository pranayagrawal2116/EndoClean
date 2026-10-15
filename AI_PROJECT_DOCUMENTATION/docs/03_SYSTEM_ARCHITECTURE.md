# 03: System Architecture

## 1. High-Level Modular Architecture
EndoClean is structured into five decoupled subsystems:
1. **Data Pipeline:** Metadata extraction, patient-wise stratified partitioning, and augmentation.
2. **Preprocessing Engine:** Vectorized Perona-Malik PDE solver and Gaussian blur baseline.
3. **Deep Learning Core:** Pre-trained ResNet50 feature extractor with fine-tuned residual blocks and class-weighted loss.
4. **Explainability & Evaluation:** Grad-CAM saliency mapping and multi-metric aggregation across experimental seeds.
5. **Interactive Console & Orchestrator:** Streamlit web interface with asynchronous background process management.

```mermaid
flowchart TD
    subgraph Data_Pipeline [1. Data Pipeline]
        RawData["Kvasir-Capsule Dataset (47k images)"] --> Meta["metadata.csv (Patient IDs & Labels)"]
        Meta --> Split["GroupShuffleSplit (70% Train, 15% Val, 15% Test)"]
    end

    subgraph Preprocessing [2. Preprocessing Engine]
        Split --> RawCond["Condition 1: Raw Images"]
        Split --> GauCond["Condition 2: Gaussian Blur (5x5, σ=1.0)"]
        Split --> PDECond["Condition 3: Perona-Malik PDE (κ=30, iter=15)"]
    end

    subgraph Modeling [3. Deep Learning Core]
        RawCond --> TrainRaw["ResNet50 (Raw Model)"]
        GauCond --> TrainGau["ResNet50 (Gaussian Model)"]
        PDECond --> TrainPDE["ResNet50 (PDE Model)"]
        WLoss["Class-Weighted Cross Entropy"] -.-> TrainRaw & TrainGau & TrainPDE
    end

    subgraph Evaluation [4. Evaluation & Explainability]
        TrainRaw & TrainGau & TrainPDE --> Eval["Holdout Patient Evaluation (Macro F1)"]
        TrainRaw & TrainPDE --> GradCAM["Grad-CAM (layer4[-1] Activation Maps)"]
    end

    subgraph Deployment [5. Interactive Console]
        GradCAM --> WebUI["Streamlit Dashboard (demo/app.py)"]
        WebUI <--> RunSh["Asynchronous Orchestrator (run.sh --all)"]
    end
```

---

## 2. Hardware Compute Routing
All scripts incorporate dynamic hardware abstraction:
```python
if torch.cuda.is_available():
    device = torch.device('cuda')       # NVIDIA GPUs
elif torch.backends.mps.is_available():
    device = torch.device('mps')        # Apple Silicon Metal Performance Shaders
else:
    device = torch.device('cpu')        # Fallback CPU
```
