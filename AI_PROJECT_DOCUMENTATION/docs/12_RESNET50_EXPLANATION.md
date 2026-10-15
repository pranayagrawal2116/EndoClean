# 12: ResNet50 Architecture & Transfer Learning

## 1. Model Architecture
ResNet50 (He et al., 2016) is a 50-layer deep Convolutional Neural Network incorporating residual skip connections:

$$y = \mathcal{F}(x, \{W_i\}) + x$$

The identity shortcut $+x$ eliminates the vanishing gradient problem, allowing gradients to backpropagate directly through deep stages.

---

## 2. Layer Freezing & Classification Head
In `src/train.py`, we employ a targeted transfer learning strategy:
1. **Pre-trained Backbone:** `torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)`.
2. **Frozen Stages:** `conv1`, `bn1`, `layer1`, `layer2` (`requires_grad = False`). Preserves general edge and texture detectors.
3. **Fine-Tuned Stages:** `layer3` (1024 channels) and `layer4` (2048 channels) are fine-tuned to adapt high-level feature representations to endoscopic pathology.
4. **Custom Classification Head:**
   ```python
   model.fc = nn.Sequential(
       nn.Dropout(p=0.5),
       nn.Linear(in_features=2048, out_features=5)
   )
   ```
- **Total Parameters:** ~25.56 Million
- **Trainable Parameters:** ~23.51 Million
- **Frozen Parameters:** ~2.05 Million
