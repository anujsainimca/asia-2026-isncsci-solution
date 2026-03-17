import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb
import subprocess

# =========================================
# 1  Load
# =========================================
meta          = pd.read_csv("metadata_train_1.csv")
features      = pd.read_csv("features_train_1.csv")
labels        = pd.read_csv("labels_train_1.csv")
test_meta     = pd.read_csv("metadata_test_1.csv")
test_features = pd.read_csv("features_test_1.csv")
submission    = pd.read_csv("labels_test_1_dummy.csv")

target_cols = [c for c in labels.columns if c != "ID"]
y = labels[target_cols].copy()

# GPU detection
try:
    subprocess.check_output("nvidia-smi", stderr=subprocess.DEVNULL)
    GPU_AVAILABLE = True
    print("GPU detected")
except Exception:
    GPU_AVAILABLE = False
    print("No GPU — running on CPU")

# XGBoost version check
XGB_VERSION = tuple(int(x) for x in xgb.__version__.split(".")[:2])
XGB_NEW_API = XGB_VERSION >= (2, 0)
print(f"XGBoost {xgb.__version__}")


# =========================================
# 2  Features (identical to best pipeline)
# =========================================
def build_features(feat_df, meta_df):
    X = feat_df.merge(meta_df, on="ID", how="left").copy()
    la = ["elbfll","wrextl","elbexl","finfll","finabl"]
    ra = ["elbflr","wrextr","elbexr","finflr","finabr"]
    ll = ["hipfll","kneexl","ankdol","ankpll"]
    rl = ["hipflr","kneetr","ankdor","ankplr"]
    all_m = la + ra + ll + rl
    X["UEMS"]        = X[la + ra].sum(axis=1)
    X["LEMS"]        = X[ll + rl].sum(axis=1)
    X["total_motor"] = X[all_m].sum(axis=1)
    X["motor_ratio"] = X["UEMS"] / (X["LEMS"] + 1)
    X["arm_diff"]    = (X[la].sum(axis=1) - X[ra].sum(axis=1)).abs()
    X["leg_diff"]    = (X[ll].sum(axis=1) - X[rl].sum(axis=1)).abs()
    X["hand_sum"]    = X[["finfll","finabl","finflr","finabr"]].sum(axis=1)
    X["n_zero_m"]    = (X[all_m] == 0).sum(axis=1)
    X["n_full_m"]    = (X[all_m] == 5).sum(axis=1)
    X["pct_zero_m"]  = X["n_zero_m"] / len(all_m)
    X["severity"]    = (X["total_motor"] < 10).astype(int)
    X["complete"]    = (X["total_motor"] == 0).astype(int)
    def mlevel(row, cols):
        for i, c in enumerate(cols):
            if row[c] < 3: return i
        return len(cols)
    X["mlevel_l"] = X.apply(mlevel, axis=1, cols=la)
    X["mlevel_r"] = X.apply(mlevel, axis=1, cols=ra)
    X["mlevel"]   = (X["mlevel_l"] + X["mlevel_r"]) / 2
    sens_all = [c for c in feat_df.columns
                if ("ltl" in c or "ltr" in c or "ppl" in c or "ppr" in c
                    or c == "anyana") and c in X.columns]
    lt_c = [c for c in sens_all if "ltl" in c or "ltr" in c]
    pp_c = [c for c in sens_all if "ppl" in c or "ppr" in c]
    X["n_intact_lt"]  = (feat_df[lt_c].fillna(-1) == 2).sum(axis=1).values
    X["n_zero_lt"]    = (feat_df[lt_c].fillna(-1) == 0).sum(axis=1).values
    X["n_partial_lt"] = (feat_df[lt_c].fillna(-1) == 1).sum(axis=1).values
    X["n_intact_pp"]  = (feat_df[pp_c].fillna(-1) == 2).sum(axis=1).values
    X["n_zero_pp"]    = (feat_df[pp_c].fillna(-1) == 0).sum(axis=1).values
    X["n_known_sens"] = feat_df[sens_all].notna().sum(axis=1).values
    X["lt_mean"]      = feat_df[lt_c].mean(axis=1).values
    X["pp_mean"]      = feat_df[pp_c].mean(axis=1).values
    X["sens_cover"]   = feat_df[sens_all].notna().mean(axis=1).values
    region_order = ["c2","c3","c4","c5","c6","c7","c8",
                    "t1","t2","t3","t4","t5","t6","t7","t8","t9","t10","t11","t12",
                    "l1","l2","l3","l4","l5","s1","s2","s3","s45"]
    ltl_ordered = [f"{r}ltl" for r in region_order if f"{r}ltl" in feat_df.columns]
    ltr_ordered = [f"{r}ltr" for r in region_order if f"{r}ltr" in feat_df.columns]
    def first_zero_level(row, cols):
        for i, c in enumerate(cols):
            if c in row.index and row[c] == 0: return i
        return len(cols)
    def last_intact_level(row, cols):
        last = -1
        for i, c in enumerate(cols):
            if c in row.index and row[c] == 2: last = i
        return last
    X["first_zero_ltl"]   = feat_df.apply(first_zero_level,  axis=1, cols=ltl_ordered)
    X["last_intact_ltl"]  = feat_df.apply(last_intact_level, axis=1, cols=ltl_ordered)
    X["first_zero_ltr"]   = feat_df.apply(first_zero_level,  axis=1, cols=ltr_ordered)
    X["last_intact_ltr"]  = feat_df.apply(last_intact_level, axis=1, cols=ltr_ordered)
    X["injury_level_est"] = (X["first_zero_ltl"] + X["first_zero_ltr"]) / 2
    if "time" in X.columns:
        X["log_time"]      = np.log1p(X["time"])
        X["t_x_motor"]     = X["time"] * X["total_motor"]
        X["t_x_UEMS"]      = X["time"] * X["UEMS"]
        X["t_x_LEMS"]      = X["time"] * X["LEMS"]
        X["t_x_severity"]  = X["time"] * X["severity"]
        X["t_x_n_intact"]  = X["time"] * X["n_intact_lt"]
        X["t_x_inj_level"] = X["time"] * X["injury_level_est"]
    for col in X.select_dtypes(include=["object"]).columns:
        if col != "ID":
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    return X.drop(columns=["ID"]).fillna(-1)


X_all      = build_features(features, meta)
X_test_all = build_features(test_features, test_meta)
print(f"Feature matrix: {X_all.shape}")


# =========================================
# 3  Ordinal models
#    For each target train TWO binary classifiers:
#    Model A: P(score >= 1)
#    Model B: P(score >= 2)
#    Prediction = P(>=1) + P(>=2)  →  range [0, 2]
#
#    For anyana (binary 0/1): train ONE classifier P(score >= 1)
# =========================================
def make_cat(seed=42):
    return CatBoostClassifier(
        iterations=1000, learning_rate=0.02, depth=7,
        l2_leaf_reg=3, loss_function="Logloss",
        verbose=False, random_seed=seed,
        task_type="GPU" if GPU_AVAILABLE else "CPU"
    )

def make_lgb(seed=42):
    return lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.02, max_depth=7,
        num_leaves=63, min_child_samples=15,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.05, reg_lambda=0.5,
        random_state=seed, verbose=-1
    )

def make_xgb(seed=42):
    params = dict(
        n_estimators=1000, learning_rate=0.02, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.05, reg_lambda=0.5,
        random_state=seed, verbosity=0,
        early_stopping_rounds=60,
        eval_metric="logloss"
    )
    if GPU_AVAILABLE:
        if XGB_NEW_API:
            params["device"] = "cuda"
        else:
            params["tree_method"] = "gpu_hist"
    else:
        params["tree_method"] = "hist"
    return xgb.XGBClassifier(**params)


def predict_proba_avg(models, X):
    """Average probability predictions from multiple models."""
    probs = []
    for m in models:
        if hasattr(m, "predict_proba"):
            probs.append(m.predict_proba(X)[:, 1])
        else:
            probs.append(m.predict(X))
    return np.mean(probs, axis=0)


# =========================================
# 4  Hard baseline copy
# =========================================
train_known_all    = np.zeros((len(y), len(target_cols)), dtype=bool)
test_known_all     = np.zeros((len(X_test_all), len(target_cols)), dtype=bool)
train_hardcopy     = np.zeros((len(y), len(target_cols)))
test_preds_hardcopy = np.zeros((len(X_test_all), len(target_cols)))

for t, target in enumerate(target_cols):
    if target in features.columns:
        base_tr = pd.to_numeric(features.set_index("ID")[target], errors="coerce").values
        train_known_all[:, t] = ~np.isnan(base_tr)
        train_hardcopy[train_known_all[:, t], t] = base_tr[train_known_all[:, t]]
    if target in test_features.columns:
        base_te = pd.to_numeric(test_features.set_index("ID")[target], errors="coerce").values
        test_known_all[:, t] = ~np.isnan(base_te)
        test_preds_hardcopy[test_known_all[:, t], t] = base_te[test_known_all[:, t]]

print(f"Hard copy rate (train): {train_known_all.mean():.3f}")



# =========================================
# 5  Seed averaging — run 3 seeds, average test predictions
#    OOF from seed=42 used for CV score reporting
# =========================================
SEEDS = [123, 777, 999]

# Accumulators — store P(>=1) and P(>=2) SEPARATELY
# Average in probability space before summing → better ordinal calibration
test_p1_all   = np.zeros((len(X_test_all), len(target_cols)))
test_p2_all   = np.zeros((len(X_test_all), len(target_cols)))
oof_p1_s42    = np.zeros((len(y), len(target_cols)))  # will store seed 123 OOF
oof_p2_s42    = np.zeros((len(y), len(target_cols)))
oof_preds_s42 = np.zeros((len(y), len(target_cols)))

for seed_idx, SEED in enumerate(SEEDS):
    print(f"\n{'='*50}")
    print(f"SEED {SEED}  ({seed_idx+1}/{len(SEEDS)})")
    print(f"{'='*50}")

    kf_seed = KFold(n_splits=5, shuffle=True, random_state=SEED)

    oof_seed  = np.zeros((len(y), len(target_cols)))
    test_seed = np.zeros((len(X_test_all), len(target_cols)))

    # Store p1 and p2 separately for this seed
    test_p1_seed = np.zeros((len(X_test_all), len(target_cols)))
    test_p2_seed = np.zeros((len(X_test_all), len(target_cols)))
    oof_p1_seed  = np.zeros((len(y), len(target_cols)))
    oof_p2_seed  = np.zeros((len(y), len(target_cols)))

    # Re-apply hard baseline copy
    for t, target in enumerate(target_cols):
        oof_seed[train_known_all[:, t], t] = y[target].values[train_known_all[:, t]]
        test_seed[test_known_all[:, t], t] = test_preds_hardcopy[test_known_all[:, t], t]

    for t, target in enumerate(target_cols):
        y_t = y[target].values
        train_unknown = ~train_known_all[:, t]
        test_unknown  = ~test_known_all[:, t]

        is_binary    = (target == "anyana") or (y_t.max() <= 1)
        n_thresholds = 1 if is_binary else 2

        oof_model        = np.zeros(len(y))
        test_fold_preds  = np.zeros((len(X_test_all), kf_seed.n_splits))
        # Per-threshold fold accumulators
        oof_fold_p1   = np.zeros((len(y), kf_seed.n_splits))
        oof_fold_p2   = np.zeros((len(y), kf_seed.n_splits))
        test_fold_p1  = np.zeros((len(X_test_all), kf_seed.n_splits))
        test_fold_p2  = np.zeros((len(X_test_all), kf_seed.n_splits))

        for fold, (tr_idx, val_idx) in enumerate(kf_seed.split(X_all)):
            X_tr  = X_all.iloc[tr_idx]
            X_val = X_all.iloc[val_idx]
            y_tr  = y_t[tr_idx]
            y_val = y_t[val_idx]

            fold_val_pred  = np.zeros(len(val_idx))
            fold_test_pred = np.zeros(len(X_test_all))
            p1_val = np.zeros(len(val_idx))
            p2_val = np.zeros(len(val_idx))
            p1_tst = np.zeros(len(X_test_all))
            p2_tst = np.zeros(len(X_test_all))

            for threshold in range(1, n_thresholds + 1):
                y_tr_bin  = (y_tr  >= threshold).astype(int)
                y_val_bin = (y_val >= threshold).astype(int)

                if len(np.unique(y_tr_bin)) < 2:
                    constant = float(y_tr_bin[0])
                    fold_val_pred  += constant
                    fold_test_pred += constant
                    if threshold == 1:
                        p1_val[:] = constant; p1_tst[:] = constant
                    else:
                        p2_val[:] = constant; p2_tst[:] = constant
                    continue

                cat  = make_cat(seed=SEED)
                lgbm = make_lgb(seed=SEED)
                xgbm = make_xgb(seed=SEED)

                cat.fit(X_tr, y_tr_bin,
                        eval_set=(X_val, y_val_bin),
                        early_stopping_rounds=60)
                lgbm.fit(X_tr, y_tr_bin,
                         eval_set=[(X_val, y_val_bin)],
                         callbacks=[lgb.early_stopping(60, verbose=False),
                                     lgb.log_evaluation(-1)])
                xgbm.fit(X_tr, y_tr_bin,
                         eval_set=[(X_val, y_val_bin)],
                         verbose=False)

                p_val  = predict_proba_avg([cat, lgbm, xgbm], X_val)
                p_test = predict_proba_avg([cat, lgbm, xgbm], X_test_all)

                # Clip probabilities before accumulation
                p_val  = np.clip(p_val,  0.001, 0.999)
                p_test = np.clip(p_test, 0.001, 0.999)

                fold_val_pred  += p_val
                fold_test_pred += p_test

                # Store separately by threshold
                if threshold == 1:
                    p1_val = p_val; p1_tst = p_test
                else:
                    p2_val = p_val; p2_tst = p_test

            oof_model[val_idx]          = fold_val_pred
            test_fold_preds[:, fold]    = fold_test_pred
            oof_fold_p1[val_idx, fold]  = p1_val
            oof_fold_p2[val_idx, fold]  = p2_val
            test_fold_p1[:, fold]       = p1_tst
            test_fold_p2[:, fold]       = p2_tst

        oof_seed[train_unknown, t]   = oof_model[train_unknown]
        test_seed[test_unknown, t]   = test_fold_preds[test_unknown].mean(axis=1)
        # FIX: each OOF row is filled in exactly 1 fold — use sum not mean
        # (mean divides by n_splits=5 but only 1 fold is nonzero per row)
        oof_p1_seed[train_unknown, t] = oof_fold_p1[train_unknown].sum(axis=1)
        oof_p2_seed[train_unknown, t] = oof_fold_p2[train_unknown].sum(axis=1)
        # Test is correct — all 5 folds contribute, mean is right
        test_p1_seed[test_unknown, t] = test_fold_p1[test_unknown].mean(axis=1)
        test_p2_seed[test_unknown, t] = test_fold_p2[test_unknown].mean(axis=1)

        if (t + 1) % 20 == 0 or t < 3:
            rmse_t = np.sqrt(mean_squared_error(y_t, np.clip(oof_seed[:, t], 0, y_t.max())))
            print(f"  [{t+1:3d}/{len(target_cols)}] {target:10s}  RMSE={rmse_t:.4f}")

    # Accumulate p1 and p2 separately across seeds
    test_p1_all += test_p1_seed
    test_p2_all += test_p2_seed

    if SEED == 123:
        oof_preds_s42 = oof_seed.copy()
        oof_p1_s42    = oof_p1_seed.copy()
        oof_p2_s42    = oof_p2_seed.copy()

    seed_rmse = np.sqrt(mean_squared_error(y.values, np.clip(oof_seed, y.min().values, y.max().values)))
    print(f"Seed {SEED} CV RMSE: {seed_rmse:.5f}")


# =========================================
# 6  Reconstruct predictions from averaged probabilities
#    Average P(>=1) and P(>=2) across seeds FIRST
#    then sum → better ordinal calibration
# =========================================
test_p1_avg = test_p1_all / len(SEEDS)
test_p2_avg = test_p2_all / len(SEEDS)

# Clip averaged probabilities
test_p1_avg = np.clip(test_p1_avg, 0.001, 0.999)
test_p2_avg = np.clip(test_p2_avg, 0.001, 0.999)

# Reconstruct: prediction = P(>=1) + P(>=2)
test_preds_prob_avg = test_p1_avg + test_p2_avg

# For hard-copy positions, restore exact values
for t, target in enumerate(target_cols):
    test_preds_prob_avg[test_known_all[:, t], t] = test_preds_hardcopy[test_known_all[:, t], t]

# OOF from seed 42 (same as before for CV reporting)
oof_p1_s42_clipped = np.clip(oof_p1_s42, 0.001, 0.999)
oof_p2_s42_clipped = np.clip(oof_p2_s42, 0.001, 0.999)
oof_prob_avg = oof_p1_s42_clipped + oof_p2_s42_clipped
for t, target in enumerate(target_cols):
    oof_prob_avg[train_known_all[:, t], t] = y[target].values[train_known_all[:, t]]


# =========================================
# 7  Spatial smoothing across spinal segments
#    final_C6 = 0.25*C5 + 0.5*C6 + 0.25*C7
#    Applied to test predictions only, not training
#    Skip hard-copy positions
# =========================================
region_order = ["c2","c3","c4","c5","c6","c7","c8",
                "t1","t2","t3","t4","t5","t6","t7","t8","t9","t10","t11","t12",
                "l1","l2","l3","l4","l5","s1","s2","s3","s45"]

# Build neighbor map per anatomical group
groups = {
    "ltl": [f"{r}ltl" for r in region_order if f"{r}ltl" in target_cols],
    "ltr": [f"{r}ltr" for r in region_order if f"{r}ltr" in target_cols],
    "ppl": [f"{r}ppl" for r in region_order if f"{r}ppl" in target_cols],
    "ppr": [f"{r}ppr" for r in region_order if f"{r}ppr" in target_cols],
}

target_idx = {t: i for i, t in enumerate(target_cols)}
SMOOTH_W = 0.25  # neighbor weight (center gets 1 - 2*W = 0.5)

test_smoothed = test_preds_prob_avg.copy()

for group_targets in groups.values():
    for i, target in enumerate(group_targets):
        t_idx = target_idx[target]
        neighbors = []
        if i > 0:
            neighbors.append(group_targets[i-1])
        if i < len(group_targets) - 1:
            neighbors.append(group_targets[i+1])

        if not neighbors:
            continue

        # Only smooth non-hard-copy positions
        unknown_mask = ~test_known_all[:, t_idx]
        if unknown_mask.sum() == 0:
            continue

        center = test_preds_prob_avg[unknown_mask, t_idx].copy()
        n_neighbors = len(neighbors)
        center_w = 1.0 - n_neighbors * SMOOTH_W

        smoothed = center_w * center
        for nb in neighbors:
            nb_idx = target_idx[nb]
            smoothed += SMOOTH_W * test_preds_prob_avg[unknown_mask, nb_idx]

        test_smoothed[unknown_mask, t_idx] = smoothed

print(f"Spatial smoothing applied with neighbor weight={SMOOTH_W}")


# =========================================
# 8  Score + submit
# =========================================
label_min = y.min().values
label_max = y.max().values

# OOF CV — from prob-avg (seed 42)
oof_final  = np.clip(oof_prob_avg,   label_min, label_max)
# Test — smoothed prob-avg
test_final = np.clip(test_smoothed,  label_min, label_max)

rmse_final = np.sqrt(mean_squared_error(y.values, oof_final))
print(f"\nFinal CV RMSE (seed=123, prob-avg, no spatial smooth on OOF): {rmse_final:.5f}")

worst5 = sorted(
    {col: np.sqrt(mean_squared_error(y[col], oof_final[:, t]))
     for t, col in enumerate(target_cols)}.items(),
    key=lambda x: x[1], reverse=True
)[:5]
print("Hardest 5:", worst5)

submission[target_cols] = test_final
submission.to_csv("submission_ordinal_v3_advanced.csv", index=False)
print("Submission saved as submission_ordinal_v3_advanced.csv!")
