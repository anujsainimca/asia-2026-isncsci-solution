"""
Smart blending of Task 1 submissions.
Supports:
  - Simple average
  - Weighted average (by inverse LB score — better score = higher weight)
  - Rank averaging (average the ranks instead of raw predictions)

Usage: place all submission CSVs in same folder and run.
Edit the SUBMISSIONS dict with your actual filenames and LB scores.
"""
import pandas as pd
import numpy as np

# =========================================
# CONFIG — add/edit your submissions here
# =========================================
SUBMISSIONS = {
    "submission_ordinal.csv":         0.40683,  # ordinal v1 seed42
    "submission_seed123_only.csv":    0.40661,  # seed123 only
    "submission_ordinal_v2_final.csv": None,    # overnight run — fill LB score when done
}

# Load all submissions
dfs = {}
for fname, lb in SUBMISSIONS.items():
    try:
        dfs[fname] = pd.read_csv(fname)
        print(f"Loaded {fname}  LB={lb}")
    except FileNotFoundError:
        print(f"SKIPPED (not found): {fname}")

target_cols = [c for c in list(dfs.values())[0].columns if c != "ID"]
available = {k: v for k, v in SUBMISSIONS.items() if k in dfs and v is not None}
print(f"\nAvailable for blending: {len(available)} submissions")

# =========================================
# 1. Simple average (equal weight)
# =========================================
preds_stack = np.stack([dfs[f][target_cols].values for f in dfs], axis=0)
simple_avg = preds_stack.mean(axis=0)

blend_simple = dfs[list(dfs.keys())[0]].copy()
blend_simple[target_cols] = simple_avg
blend_simple.to_csv("submission_task1_blend_simple.csv", index=False)
print("\nSaved: submission_task1_blend_simple.csv (equal weight)")

# =========================================
# 2. Weighted average (inverse LB — lower score = higher weight)
# =========================================
if len(available) >= 2:
    weights = {}
    for fname in dfs:
        if fname in available:
            # inverse of LB score, normalized
            weights[fname] = 1.0 / available[fname]
        else:
            # no LB score known — give it average weight
            avg_inv = np.mean([1.0/v for v in available.values()])
            weights[fname] = avg_inv

    total_w = sum(weights.values())
    weights = {k: v/total_w for k, v in weights.items()}
    print("\nWeights (inverse LB):")
    for f, w in weights.items():
        print(f"  {f}: {w:.4f}")

    weighted_preds = sum(
        weights[f] * dfs[f][target_cols].values
        for f in dfs
    )
    blend_weighted = dfs[list(dfs.keys())[0]].copy()
    blend_weighted[target_cols] = weighted_preds
    blend_weighted.to_csv("submission_task1_blend_weighted.csv", index=False)
    print("Saved: submission_task1_blend_weighted.csv (inverse-LB weighted)")

# =========================================
# 3. Rank averaging (most robust to outliers)
#    Convert predictions to ranks, average ranks, convert back
# =========================================
ranked_stack = np.zeros_like(preds_stack)
for i in range(preds_stack.shape[0]):
    for j in range(preds_stack.shape[2]):
        col_preds = preds_stack[i, :, j]
        # rank within column (argsort of argsort gives ranks)
        ranked_stack[i, :, j] = col_preds.argsort().argsort().astype(float)

# Average ranks, then map back to average prediction values
rank_avg = ranked_stack.mean(axis=0)
# Normalize back to [0,2] range using min/max of original predictions
pred_min = preds_stack.min(axis=0)
pred_max = preds_stack.max(axis=0)
rank_min = ranked_stack.min(axis=0)
rank_max = ranked_stack.max(axis=0)
rank_range = np.where(rank_max > rank_min, rank_max - rank_min, 1)
pred_range = pred_max - pred_min
rank_blend = pred_min + (rank_avg - rank_min) / rank_range * pred_range
rank_blend = np.clip(rank_blend, 0, 2)

blend_rank = dfs[list(dfs.keys())[0]].copy()
blend_rank[target_cols] = rank_blend
blend_rank.to_csv("submission_task1_blend_rank.csv", index=False)
print("Saved: submission_task1_blend_rank.csv (rank averaging)")

# =========================================
# 4. Sanity check
# =========================================
print("\n=== Sanity checks ===")
for name, df in [
    ("simple", blend_simple),
    ("weighted", blend_weighted if len(available) >= 2 else None),
    ("rank", blend_rank)
]:
    if df is None: continue
    vals = df[target_cols].values
    print(f"{name:10s}: min={vals.min():.4f}  max={vals.max():.4f}  "
          f"mean={vals.mean():.4f}  out_of_range={((vals<0)|(vals>2)).sum()}")

print("\nDone! Submit in this order:")
print("  1. submission_task1_blend_weighted.csv (best theory)")
print("  2. submission_task1_blend_rank.csv     (most robust)")
print("  3. submission_task1_blend_simple.csv   (fallback)")
