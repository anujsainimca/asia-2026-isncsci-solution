# Task 1 — Winning Pipeline

**Score: 0.40455 public LB | Rank 1 private LB**

## Files
- `pipeline_ordinal_v3_advanced.py` — Final winning submission

## Quick Start
```bash
# Update file paths at top of script if not using Kaggle
python pipeline_ordinal_v3_advanced.py
```

## What it does
1. Builds 184 features from motor, sensory, demographic and time data
2. Copies known sensory values directly (13.4% of targets)
3. For each of 112 targets trains ordinal binary classifiers
4. Averages P(≥1) and P(≥2) in probability space across 3 seeds
5. Applies anatomical spatial smoothing (w=0.25) to test predictions
6. Saves `submission_ordinal_v3_advanced.csv`

## Runtime
~4.5 hours on CPU (3 seeds × 90 mins each)
