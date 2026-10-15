# 04: Project Workflow

This document traces the complete execution lifecycle of EndoClean from raw video ingestion to clinical visualization.

```mermaid
sequenceDiagram
    autonumber
    actor Clinician as Gastroenterologist / User
    participant Script as run.sh Orchestrator
    participant Prep as Preprocessing (PDE / Gaussian)
    participant Train as PyTorch ResNet50
    participant Eval as Evaluation & Grad-CAM
    participant UI as Streamlit Console

    Clinician->>Script: Execute ./run.sh --all
    Script->>Prep: Spawn background experiments
    Script->>UI: Launch demo/app.py immediately in foreground
    Prep->>Prep: Denoise images via Perona-Malik PDE (κ=30, iter=15)
    Prep->>Train: Pass augmented tensors (Train / Val / Test)
    Train->>Train: Optimize class-weighted Cross-Entropy on MPS/GPU
    Train->>Eval: Save checkpoints & compute Macro F1 on unseen patients
    Eval->>Eval: Compute Grad-CAM heatmaps on layer4[-1]
    Clinician->>UI: Upload single test endoscopy frame
    UI->>Prep: Apply online perona_malik_diffusion()
    UI->>Train: Run inference on Raw & PDE models
    UI->>Eval: Compute live Grad-CAM heatmaps
    UI-->>Clinician: Render dual metric readouts & comparative analysis cards
```

### Sequential Operational Stages
1. **Metadata Structuring:** `prepare_metadata.py` ingests official Kvasir splits, extracts `video_id`, and formats `data/metadata.csv`.
2. **Offline Preprocessing:** `process_folder()` runs Perona-Malik diffusion across 47,238 frames, populating `data/pde_denoised/`.
3. **Model Training:** `src/train.py` fine-tunes ResNet50 across data regimes and random seeds ($42, 43, 44$).
4. **Performance Verification:** `src/evaluate.py` verifies generalization on held-out patient videos.
5. **Explainability Extraction:** `src/gradcam.py` synthesizes spatial heatmaps.
6. **Clinical Inference:** `demo/app.py` serves the final interactive diagnostic application.
