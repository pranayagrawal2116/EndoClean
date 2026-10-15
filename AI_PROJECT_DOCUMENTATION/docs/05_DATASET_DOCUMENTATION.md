# 05: Dataset Documentation

## 1. Dataset Origin & Benchmark Specification
- **Dataset Name:** Kvasir-Capsule
- **Publishing Institution:** Simula Research Laboratory & Oslo University Hospital
- **Access Protocol:** Open-access research archive (OSF repository: `https://osf.io/dv2ag/`)
- **Total Images in Archive:** 47,238 labeled frames
- **Unique Patient Videos:** 43 patient video recordings
- **Format:** RGB JPEG images ($336\times 336$ resolution)
- **Metadata CSV:** `EndoClean/data/metadata.csv` (47,161 verified rows, 3 columns: `filename`, `class`, `video_id`)

---

## 2. Target Study Class Distribution

| Class Name | Clinical Definition | Labeled Frames | Proportion (Target Set) |
| :--- | :--- | :--- | :--- |
| **Normal** | Healthy small bowel mucosal tissue | 34,338 | 92.78% |
| **Angiectasia** | Arteriovenous vascular malformations (bleeding risk) | 866 | 2.34% |
| **Ulcer** | Mucosal breakdown with fibrin base | 854 | 2.31% |
| **Erosion** | Superficial epithelial mucosal defects | 506 | 1.37% |
| **Blood** | Active luminal hemorrhage or pooled hematin | 446 | 1.20% |
| **Total Target Set** | **Study Evaluation Population** | **37,010** | **100.00%** |

*Note on Excluded Classes:* Other anatomical categories in the raw Kvasir dataset (e.g. `Ileocecal valve`: 4,189; `Reduced mucosal view`: 2,906; `Pylorus`: 1,529; `Foreign body`: 776; `Lymphangiectasia`: 592; `Erythema`: 159; `Polyp`: 55; `Ampulla of vater`: 10) represent non-pathological anatomical landmarks or poorly sampled classes, and are excluded from the 5-class study scope.

---

## 3. Data Integrity & Leakage Prevention
- **Missing Values:** Exactly 0 missing values across all columns.
- **Duplicates:** Exactly 0 duplicate rows.
- **Split Strategy:** Grouped by `video_id` using `GroupShuffleSplit`. Video frames from the same patient are never shared across Train and Test splits.
