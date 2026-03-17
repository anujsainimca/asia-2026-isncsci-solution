# Task 2 — Winning Pipeline

**Score: 0.38327 public LB | Rank 1 private LB**

## Files
- `pipeline_task2_v9_winning.py` — Final winning submission (v6 + prob-space averaging)
- `pipeline_task2_v6.py` — Runner-up, simpler and robust (0.38327 public LB)

## Quick Start
```bash
# Ensure Kaggle dataset paths are correct at top of script
python task2/pipeline_task2_v9_winning.py
```

## What it does
1. Loads Task 2 data + Task 1 data (shared patients)
2. Builds augmented training set: 931 → 2114 rows from 4 sources:
   - A: Task1 time>1 rows for Task2 train patients (676)
   - B: Task1 time>1 rows for Task2 TEST patients (186) ← key insight
   - C: Task1 time=1 rows for Task2 train patients (255)
   - D: Task1 time=1 rows for Task2 TEST patients (66)
3. Engineers 223 features including all 134 w1_ baseline sensory cols
4. Trains ordinal binary classifiers per target on augmented data
5. Stores P(≥1) and P(≥2) separately, averages in probability space
6. Saves `submission_task2_v9.csv`

## Runtime
~90 minutes on CPU (single seed)

## Data Paths
Update these at the top of the script:
```python
"/kaggle/input/datasets/anujsaini1231/shared-task2/..."
"/kaggle/input/datasets/anujsaini1231/shared-task1/..."
```
