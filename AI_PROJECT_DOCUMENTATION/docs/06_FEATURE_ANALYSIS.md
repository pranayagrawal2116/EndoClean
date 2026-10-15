# 06: Feature Analysis & Representation

In medical computer vision, input features exist simultaneously as spatial color signals (RGB pixel values) and abstract semantic hierarchies learned by deep convolutional layers.

---

## 1. Spatial Pixel Channels (Input Space)

| Channel | Format | Normalized Mean / Std | Clinical Diagnostic Significance |
| :--- | :--- | :--- | :--- |
| **Red (R)** | uint8 [0, 255] | Mean: 0.485, Std: 0.229 | Dominant wavelength in endoscopy. Critical for detecting vascular ectasia, active bleeding, and erythema. |
| **Green (G)** | uint8 [0, 255] | Mean: 0.456, Std: 0.224 | Delineates mucosal surface patterns, villous architecture, and ulcer margins. |
| **Blue (B)** | uint8 [0, 255] | Mean: 0.406, Std: 0.225 | Absorbed strongly by hemoglobin. Key for depth perception and distinguishing bile/chyme from tissue. |

---

## 2. Learned Hierarchical Feature Maps (ResNet50)

| Layer | Output Tensor Shape | Trainable Status | Semantic Representation |
| :--- | :--- | :--- | :--- |
| **conv1 + layer1** | $56\times 56\times 256$ | Frozen | Low-level Gabor edge detectors, gradient orientations, color transitions. |
| **layer2** | $28\times 28\times 512$ | Frozen | Mid-level texture junctions, repeating mucosal grain, circular lumen curvature. |
| **layer3** | $14\times 14\times 1024$ | Fine-tuned | Structural tissue boundaries, mucosal folds, localized lesion borders. |
| **layer4** | $7\times 7\times 2048$ | Fine-tuned | High-level clinical pathology semantics (ulcer base, vascular cluster). Grad-CAM target. |
| **fc (Head)** | 5 Logits | Trainable | Unnormalized class scores for `Normal`, `Angiectasia`, `Ulcer`, `Blood`, `Erosion`. |
