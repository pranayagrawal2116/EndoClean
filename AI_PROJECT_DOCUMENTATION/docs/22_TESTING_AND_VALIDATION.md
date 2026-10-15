# 22: Software Testing & Code Audit

## 1. Test Verification Matrix

| Test Suite | Component Tested | Method / Assertion | Result |
| :--- | :--- | :--- | :--- |
| **Unit Test 1** | `perona_malik_diffusion` | Checks output shape == input shape, dtype == uint8 | **PASS** |
| **Unit Test 2** | PDE Stability Limit | Asserts $\lambda = 0.15 \le 0.25$, no NaNs or Infs | **PASS** |
| **Unit Test 3** | `create_patient_split` | Verifies train and test video sets have 0 intersection | **PASS** |
| **Unit Test 4** | `build_model` | Verifies `fc.out_features == 5`, early layers frozen | **PASS** |
| **Integration 1**| PyTorch Forward Pass | Feeds $(1, 3, 224, 224)$ tensor, checks output shape $(1, 5)$ | **PASS** |
| **Integration 2**| Grad-CAM Generation | Generates $224\times 224$ heatmap overlay without dimension error | **PASS** |
| **Regression 1** | Background Process | Verifies `run.sh --all` creates PID file and detaches | **PASS** |
