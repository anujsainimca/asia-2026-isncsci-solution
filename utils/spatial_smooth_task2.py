"""
Post-processing: Spatial smoothing across dermatomes for Task 2.
Apply to existing best submission — no retraining needed.
Runs in seconds.

Smoothing: final_C6 = w*C5 + (1-2w)*C6 + w*C7
Optimal w=0.20 based on simulation.
Applied separately for ltl, ltr, ppl, ppr groups.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# =========================================
# Load best submission
# =========================================
sub = pd.read_csv("submission_task2_v5.csv")  # your best submission
target_cols = [c for c in sub.columns if c != "ID"]
preds = sub[target_cols].values.copy()

print(f"Loaded submission: {sub.shape}")
print(f"Prediction range: [{preds.min():.4f}, {preds.max():.4f}]")

# =========================================
# Spinal level ordering
# =========================================
region_order = ["c2","c3","c4","c5","c6","c7","c8",
                "t1","t2","t3","t4","t5","t6","t7","t8","t9","t10","t11","t12",
                "l1","l2","l3","l4","l5","s1","s2","s3","s45"]

groups = {
    "ltl": [f"{r}ltl" for r in region_order if f"{r}ltl" in target_cols],
    "ltr": [f"{r}ltr" for r in region_order if f"{r}ltr" in target_cols],
    "ppl": [f"{r}ppl" for r in region_order if f"{r}ppl" in target_cols],
    "ppr": [f"{r}ppr" for r in region_order if f"{r}ppr" in target_cols],
}
target_idx = {t: i for i, t in enumerate(target_cols)}

print(f"\nGroups: { {k: len(v) for k,v in groups.items()} }")

# =========================================
# Apply spatial smoothing at multiple weights
# Save each as separate submission
# =========================================
for W in [0.15, 0.20, 0.25]:
    smoothed = preds.copy()

    for group_targets in groups.values():
        for i, target in enumerate(group_targets):
            t_idx = target_idx[target]
            neighbors = []
            if i > 0: neighbors.append(group_targets[i-1])
            if i < len(group_targets)-1: neighbors.append(group_targets[i+1])
            if not neighbors: continue

            center_w = 1.0 - len(neighbors) * W
            new_val = center_w * preds[:, t_idx]
            for nb in neighbors:
                new_val += W * preds[:, target_idx[nb]]
            smoothed[:, t_idx] = new_val

    smoothed = np.clip(smoothed, 0, 2)

    out = sub.copy()
    out[target_cols] = smoothed
    fname = f"submission_task2_smooth_w{int(W*100):02d}.csv"
    out.to_csv(fname, index=False)
    print(f"w={W:.2f}: saved {fname}  range=[{smoothed.min():.4f}, {smoothed.max():.4f}]")

print("\nSubmit in order:")
print("  1. submission_task2_smooth_w20.csv  (optimal from simulation)")
print("  2. submission_task2_smooth_w15.csv  (conservative)")
print("  3. submission_task2_smooth_w25.csv  (aggressive)")
