# ASIA 2026: (E-) → ISNCSCI Imputation — 1st Place Solution

**Competition:** [ASIA 2026 Kaggle Challenge](https://www.kaggle.com/competitions/asia-2026-e-isncsci-imputation)  
**Author:** Anuj Saini  
**Final Result:** 🥇 **Rank 1 on Private Leaderboard — Both Task 1 and Task 2**

---

## Results

| Task | Public LB | Private LB |
|------|-----------|------------|
| Task 1 — ISNCSCI Sensory Imputation | 0.40455 | Rank 1 |
| Task 2 — ISNCSCI Sensory Imputation (w/ baseline) | 0.38327 | Rank 1 |

---

## Repository Structure

```
├── task1/
│   └── pipeline_ordinal_v3_advanced.py   ← Task 1 winning solution
├── task2/
│   ├── pipeline_task2_v9_winning.py      ← Task 2 best (v6 + prob-space avg)
│   └── pipeline_task2_v6.py              ← Task 2 runner-up (simpler, robust)
├── requirements.txt
└── README.md
```

---

## Problem Description

The challenge requires predicting all 112 ISNCSCI sensory scores (0/1/2 ordinal scale) at future clinical time points (4, 8, 16 weeks post-injury) for spinal cord injury patients. Scores represent 28 dermatome levels × 2 modalities (light touch + pin prick) × 2 sides (left + right), plus the anyana binary target.

**Evaluation metric:** RMSE across all predictions.

---

## Task 1 — Winning Solution

**File:** `task1/pipeline_ordinal_v3_advanced.py`

### Key Innovations

**1. Ordinal Regression via Binary Decomposition**

Instead of treating the 0/1/2 scale as continuous, we decompose each target into two binary classifiers:
- Model A: P(score ≥ 1)
- Model B: P(score ≥ 2)
- Final prediction: P(≥1) + P(≥2) → range [0, 2]

This is mathematically equivalent to E[Y] = 0×P(0) + 1×P(1) + 2×P(2), the proper ordinal expectation.

**2. Probability-Space Seed Averaging**

We train with 3 random seeds [123, 777, 999] and average P(≥1) and P(≥2) **separately** across seeds before summing. This preserves ordinal calibration better than averaging final predictions.

**3. Anatomical Spatial Smoothing**

Sensory scores follow a physiological gradient along the spinal cord. We apply post-hoc smoothing to test predictions:
```
final_C6 = 0.25 × C5 + 0.50 × C6 + 0.25 × C7
```
Applied separately for LT-left, LT-right, PP-left, PP-right groups. Only applied to test predictions, not training labels.

**4. Probability Clipping**

Clip all probabilities to [0.001, 0.999] before averaging to prevent numerically degenerate predictions from dominating.

### Model Configuration

| Parameter | CatBoost | LightGBM | XGBoost |
|-----------|----------|----------|---------|
| Type | Classifier | Classifier | Classifier |
| Loss | Logloss | Binary | Logloss |
| Iterations | 1000 | 1000 | 1000 |
| Learning rate | 0.02 | 0.02 | 0.02 |
| Max depth | 7 | 7 | 6 |
| Early stopping | 60 | 60 | 60 |

### Feature Engineering (184 features)

- Motor aggregates: UEMS, LEMS, total_motor, motor_ratio
- Motor level proxies: mlevel_l, mlevel_r, injury_level_est
- Sensory summaries: n_intact_lt, n_zero_lt, lt_mean, pp_mean
- Time interactions: log_time, t×motor, t×UEMS, t×severity
- Demographic metadata (label encoded)

### Hard Baseline Copy

13.4% of target values are known directly from sparse current sensory features. These are copied exactly without model inference.

---

## Task 2 — Winning Solution

**File:** `task2/pipeline_task2_v9_winning.py`

### Key Innovations

**1. Training Data Augmentation (931 → 2114 rows)**

The critical discovery: Task 1 and Task 2 share the same 931 patients. Task 1 contains additional rows for these patients at other timepoints, including:

| Source | Rows | Description |
|--------|------|-------------|
| A | 676 | T2 train patients, time > 1 (original) |
| B | 186 | T2 test patients, time > 1 (NEW) |
| C | 255 | T2 train patients, time = 1 (NEW) |
| D | 66 | T2 test patients, time = 1 (NEW) |

All sources have 100% label fill rate and 100% w1_ baseline feature fill rate after join.

**KFold strategy:** Split on original 931 rows only. Extra rows always in training fold, never in validation. No data leakage.

**2. w1_ Baseline Features**

Task 2 provides 134 w1_ (Week 1) full ISNCSCI baseline features with 0% NaN. These are joined onto all augmented rows via patient ID, making them the most predictive features.

**3. Ordinal Regression**

Same binary decomposition as Task 1.

**4. Probability-Space Reconstruction**

Store P(≥1) and P(≥2) separately across folds, clip to [0.001, 0.999], then reconstruct as their sum. This was the final improvement that pushed to rank 1.

### Feature Engineering (223 features)

- All 134 w1_ raw sensory features (0% NaN) — most important
- w1_ motor aggregates and recovery deltas
- w1_ injury level estimates
- Current motor and time interaction features

---

## How to Run

### Setup
```bash
pip install -r requirements.txt
```

### Task 1
```bash
# Place data files in working directory or update paths at top of script
python task1/pipeline_ordinal_v3_advanced.py
# Output: submission_ordinal_v3_advanced.csv
```

### Task 2
```bash
# Update Kaggle paths at top of script to match your dataset location
python task2/pipeline_task2_v9_winning.py
# Output: submission_task2_v9.csv
```

### Data Paths (Kaggle)
```python
# Task 2 script uses these paths — update if needed:
"/kaggle/input/datasets/anujsaini1231/shared-task2/..."
"/kaggle/input/datasets/anujsaini1231/shared-task1/..."
```

---

## What Worked vs What Didn't

### Worked ✅
| Idea | LB Gain |
|------|---------|
| Ordinal regression (Task 1) | -0.026 |
| Augmentation 931→2114 (Task 2) | -0.022 |
| Ordinal regression (Task 2) | -0.004 |
| Probability-space averaging | -0.002 |
| Spatial smoothing (Task 1 only) | -0.002 |
| Test patient augmentation (src_B, src_D) | -0.002 |

### Didn't Work ❌
- Label smoothing with regressors
- Residual stacking (overfit)
- Isotonic calibration (overfit)
- Spatial smoothing on Task 2 (diverse patients)
- Group-wise models (too few rows per group)
- Neural networks (too few rows)
- Prediction rounding
- Seed averaging beyond 3 seeds (diminishing returns)

---

## Key Insights

1. **Ordinal structure matters** — treating 0/1/2 as ordinal (not continuous) is the single most important modeling decision

2. **Patient overlap is gold** — discovering Task 1 and Task 2 share 931 patients enabled +73% training data augmentation

3. **CV ≠ LB for Task 2** — strong distribution shift means CV improvements don't reliably translate to LB. More data consistently helped LB even when it hurt CV

4. **Probability-space averaging beats prediction-space** — averaging P(≥1) and P(≥2) separately before summing preserves ordinal calibration

5. **Spatial anatomy helps Task 1** — single test patient follows smooth anatomical gradient; diverse Task 2 test patients do not

---

## Citation

If you use this code, please cite:

```
@misc{saini2026isncsci,
  author = {Anuj Saini},
  title  = {First Place Solution: ASIA 2026 (E-) → ISNCSCI Imputation Challenge},
  year   = {2026},
  url    = {https://github.com/anujsaini/asia-2026-isncsci}
}
```

---

## Contact

Anuj Saini  
Kaggle: anujsaini1231  
Competition: ASIA 2026 Kaggle Challenge
