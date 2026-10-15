# 08: Missing Values and Outliers Analysis

## 1. Missing Value Audit
A programmatic audit of `EndoClean/data/metadata.csv` (47,161 rows) yields:
- `filename`: 0 missing values (100% complete)
- `class`: 0 missing values (100% complete)
- `video_id`: 0 missing values (100% complete)

Every image file referenced in the target metadata successfully resolves to a valid, non-corrupted image file on disk.

---

## 2. Outliers in Endoscopy Computer Vision
Unlike tabular data where outliers are extreme numerical values, outliers in endoscopic imaging manifest as:
1. **Severe Underexposure (Black Frames):** Distant lumen sections where camera LED light does not penetrate.
2. **Extreme Specular Highlights:** Saturated white reflections ($[255, 255, 255]$) off wet mucosal walls.
3. **Debris & Chyme Occlusion:** Large food residues or opaque bile pools entirely occluding mucosal tissue.

### Handling Strategy:
- Outliers are **retained** in training to ensure the CNN learns real-world clinical resilience.
- **PDE Denoising Treatment:** Perona-Malik diffusion smooths the high-gradient specular glare boundaries into surrounding mucosal background, mitigating false-positive activations.
