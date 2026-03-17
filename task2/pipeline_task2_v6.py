"""
Task 2 v5: Augmented + ORDINAL Cat + LGB + XGB ensemble
Combines two proven improvements:
  1. Augmentation: 931 -> 1607 rows (Task1 rows for same patients)
  2. Ordinal regression: P(>=1) + P(>=2) instead of direct regression
     Showed +0.004 CV improvement in ET tests on Task 2.
     Gave LB 0.40683 vs 0.40880 on Task 1 ordinal vs regression.
OOF always evaluated on original Task 2 rows only.
"""
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
# 1  Load all files
# =========================================
meta          = pd.read_csv("/kaggle/input/datasets/anujsaini1231/shared-task2/metadata_train_2.csv")
features      = pd.read_csv("/kaggle/input/datasets/anujsaini1231/shared-task2/features_train_2.csv")
labels        = pd.read_csv("/kaggle/input/datasets/anujsaini1231/shared-task2/labels_train_2.csv")
test_meta     = pd.read_csv("/kaggle/input/datasets/anujsaini1231/shared-task2/metadata_test_2.csv")
test_features = pd.read_csv("/kaggle/input/datasets/anujsaini1231/shared-task2/features_test_2.csv")
submission    = pd.read_csv("/kaggle/input/datasets/anujsaini1231/shared-task2/labels_test_2_dummy.csv")

# Task 1 data — same patients, extra timepoints
features1 = pd.read_csv("/kaggle/input/datasets/anujsaini1231/shared-task1/features_train_1.csv")
labels1   = pd.read_csv("/kaggle/input/datasets/anujsaini1231/shared-task1/labels_train_1.csv")
meta1     = pd.read_csv("/kaggle/input/datasets/anujsaini1231/shared-task1/metadata_train_1.csv")

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

XGB_VERSION = tuple(int(x) for x in xgb.__version__.split(".")[:2])
XGB_NEW_API = XGB_VERSION >= (2, 0)
print(f"XGBoost {xgb.__version__}")


# =========================================
# 2  Build augmented extra rows from Task 1
#    EXPANDED augmentation — 4 sources:
#    A (current): T1 time>1 for T2 TRAIN patients (676 rows)
#    B (new):     T1 time>1 for T2 TEST  patients (186 rows)
#    C (new):     T1 time=1 for T2 TRAIN patients (255 rows)
#    D (new):     T1 time=1 for T2 TEST  patients  (66 rows)
#    All have 100% label fill and 100% w1_ fill after join
# =========================================
features['pid']      = features['ID'].str.extract(r'(id_\d+)')[0]
features1['pid']     = features1['ID'].str.extract(r'(id_\d+)')[0]
labels1['pid']       = labels1['ID'].str.extract(r'(id_\d+)')[0]
test_features['pid'] = test_features['ID'].str.extract(r'(id_\d+)')[0]

t2_train_pids = set(features['pid'].unique())
t2_test_pids  = set(test_features['pid'].unique())
all_t2_pids   = t2_train_pids | t2_test_pids

# All T1 rows for ANY Task2 patient (train or test)
t1_all = features1[features1['pid'].isin(all_t2_pids)].copy()

# Source A: T2 train patients, time>1 (current)
src_a = t1_all[t1_all['pid'].isin(t2_train_pids) & (t1_all['time'] > 1)].copy()

# Source B: T2 TEST patients, time>1 (NEW — same logic, different patients)
src_b = t1_all[t1_all['pid'].isin(t2_test_pids) & (t1_all['time'] > 1)].copy()

# Source C: T2 train patients, time=1 (NEW — baseline timepoint)
src_c = t1_all[t1_all['pid'].isin(t2_train_pids) & (t1_all['time'] == 1)].copy()

# Source D: T2 TEST patients, time=1 (NEW)
src_d = t1_all[t1_all['pid'].isin(t2_test_pids) & (t1_all['time'] == 1)].copy()

print(f"Augmentation sources:")
print(f"  A (T2-train, time>1): {len(src_a)} rows ← original")
print(f"  B (T2-test,  time>1): {len(src_b)} rows ← NEW")
print(f"  C (T2-train, time=1): {len(src_c)} rows ← NEW")
print(f"  D (T2-test,  time=1): {len(src_d)} rows ← NEW")

# Combine all extra rows
t1_extra = pd.concat([src_a, src_b, src_c, src_d], axis=0).reset_index(drop=True)
print(f"Total extra rows: {len(t1_extra)}")

# w1_ lookup — use T2 train patients where available, T2 test otherwise
w1_cols = [c for c in features.columns if c.startswith('w1_')]
t2_train_w1 = features.drop_duplicates('pid').set_index('pid')[w1_cols]
t2_test_w1  = test_features.drop_duplicates('pid').set_index('pid')[w1_cols]
t2_w1_lookup = pd.concat([t2_train_w1, t2_test_w1[~t2_test_w1.index.isin(t2_train_w1.index)]])
t1_extra = t1_extra.join(t2_w1_lookup, on='pid', how='left')
print(f"w1_ fill rate after join: {t1_extra[w1_cols].notna().mean().mean():.3f}")

# meta lookup — combine T2 train + T2 test meta
meta_cols_all = [c for c in meta.columns if c != 'ID']
meta['pid'] = meta['ID'].str.extract(r'(id_\d+)')[0]
test_meta['pid'] = test_meta['ID'].str.extract(r'(id_\d+)')[0]

# Build single meta lookup covering ALL T2 patients (train + test)
meta_train_lu = meta.drop_duplicates('pid').set_index('pid')[meta_cols_all]
meta_test_lu  = test_meta.drop_duplicates('pid').set_index('pid')[meta_cols_all]
t2_meta_lookup = pd.concat([
    meta_train_lu,
    meta_test_lu[~meta_test_lu.index.isin(meta_train_lu.index)]
])

t1_extra = t1_extra.join(t2_meta_lookup, on='pid', how='left')

# Fake ID column
t1_extra['_orig_ID'] = t1_extra['ID']
t1_extra['ID'] = t1_extra['pid']

# Labels for ALL extra rows
lab_extra = (labels1[labels1['ID'].isin(t1_extra['_orig_ID'].values)]
             [['ID'] + target_cols]
             .set_index('ID'))
y_extra = lab_extra.loc[t1_extra['_orig_ID'].values][target_cols].values
print(f"Extra labels: {y_extra.shape}, fill rate: {(~np.isnan(y_extra)).mean():.3f}")


# =========================================
# 3  Feature engineering
# =========================================
def build_features(feat_df, meta_df, is_extra=False):
    """
    feat_df: features dataframe (must have w1_ cols and motor cols)
    meta_df: metadata (joined externally for extra rows, passed normally otherwise)
    is_extra: if True, meta is already joined into feat_df
    """
    if is_extra:
        X = feat_df.copy()
    else:
        X = feat_df.merge(meta_df, on="ID", how="left").copy()

    # --- Current motor ---
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

    # --- w1_ baseline motor ---
    w1_la = ["w1_elbfll","w1_wrextl","w1_elbexl","w1_finfll","w1_finabl"]
    w1_ra = ["w1_elbflr","w1_wrextr","w1_elbexr","w1_finflr","w1_finabr"]
    w1_ll = ["w1_hipfll","w1_kneexl","w1_ankdol","w1_ankpll"]
    w1_rl = ["w1_hipflr","w1_kneetr","w1_ankdor","w1_ankplr"]
    w1_all_m = w1_la + w1_ra + w1_ll + w1_rl

    X["w1_UEMS"]        = X[w1_la + w1_ra].sum(axis=1)
    X["w1_LEMS"]        = X[w1_ll + w1_rl].sum(axis=1)
    X["w1_total_motor"] = X[w1_all_m].sum(axis=1)
    X["w1_motor_ratio"] = X["w1_UEMS"] / (X["w1_LEMS"] + 1)
    X["w1_n_zero_m"]    = (X[w1_all_m] == 0).sum(axis=1)
    X["w1_severity"]    = (X["w1_total_motor"] < 10).astype(int)
    X["w1_complete"]    = (X["w1_total_motor"] == 0).astype(int)
    X["w1_mlevel_l"]    = X.apply(mlevel, axis=1, cols=w1_la)
    X["w1_mlevel_r"]    = X.apply(mlevel, axis=1, cols=w1_ra)
    X["w1_mlevel"]      = (X["w1_mlevel_l"] + X["w1_mlevel_r"]) / 2

    # --- Motor recovery delta ---
    X["d_UEMS"]               = X["UEMS"]        - X["w1_UEMS"]
    X["d_LEMS"]               = X["LEMS"]        - X["w1_LEMS"]
    X["d_total_motor"]        = X["total_motor"] - X["w1_total_motor"]
    X["motor_recovery_ratio"] = X["d_total_motor"] / (X["w1_total_motor"] + 1)

    # --- w1_ sensory summaries ---
    region_order = ["c2","c3","c4","c5","c6","c7","c8",
                    "t1","t2","t3","t4","t5","t6","t7","t8","t9","t10","t11","t12",
                    "l1","l2","l3","l4","l5","s1","s2","s3","s45"]
    w1_lt_cols = [f"w1_{r}ltl" for r in region_order if f"w1_{r}ltl" in X.columns] + \
                 [f"w1_{r}ltr" for r in region_order if f"w1_{r}ltr" in X.columns]
    w1_pp_cols = [f"w1_{r}ppl" for r in region_order if f"w1_{r}ppl" in X.columns] + \
                 [f"w1_{r}ppr" for r in region_order if f"w1_{r}ppr" in X.columns]

    X["w1_n_intact_lt"]  = (X[w1_lt_cols].fillna(-1) == 2).sum(axis=1)
    X["w1_n_zero_lt"]    = (X[w1_lt_cols].fillna(-1) == 0).sum(axis=1)
    X["w1_n_partial_lt"] = (X[w1_lt_cols].fillna(-1) == 1).sum(axis=1)
    X["w1_n_intact_pp"]  = (X[w1_pp_cols].fillna(-1) == 2).sum(axis=1)
    X["w1_n_zero_pp"]    = (X[w1_pp_cols].fillna(-1) == 0).sum(axis=1)
    X["w1_lt_mean"]      = X[w1_lt_cols].mean(axis=1)
    X["w1_pp_mean"]      = X[w1_pp_cols].mean(axis=1)

    # w1_ injury level (first zero in LT)
    w1_ltl_ord = [f"w1_{r}ltl" for r in region_order if f"w1_{r}ltl" in X.columns]
    w1_ltr_ord = [f"w1_{r}ltr" for r in region_order if f"w1_{r}ltr" in X.columns]

    def first_zero_level(row, cols):
        for i, c in enumerate(cols):
            if c in row.index and row[c] == 0: return i
        return len(cols)

    def last_intact_level(row, cols):
        last = -1
        for i, c in enumerate(cols):
            if c in row.index and row[c] == 2: last = i
        return last

    X["w1_first_zero_ltl"]  = X.apply(first_zero_level,  axis=1, cols=w1_ltl_ord)
    X["w1_last_intact_ltl"] = X.apply(last_intact_level, axis=1, cols=w1_ltl_ord)
    X["w1_first_zero_ltr"]  = X.apply(first_zero_level,  axis=1, cols=w1_ltr_ord)
    X["w1_last_intact_ltr"] = X.apply(last_intact_level, axis=1, cols=w1_ltr_ord)
    X["w1_injury_level"]    = (X["w1_first_zero_ltl"] + X["w1_first_zero_ltr"]) / 2

    # --- Current sensory summaries (sparse) ---
    curr_lt = [c for c in X.columns if ("ltl" in c or "ltr" in c) and not c.startswith("w1_") and c != "ID" and c != "_orig_ID"]
    curr_pp = [c for c in X.columns if ("ppl" in c or "ppr" in c) and not c.startswith("w1_") and c != "ID" and c != "_orig_ID"]
    X["n_intact_lt"]  = (X[curr_lt].fillna(-1) == 2).sum(axis=1) if curr_lt else 0
    X["n_zero_lt"]    = (X[curr_lt].fillna(-1) == 0).sum(axis=1) if curr_lt else 0
    X["n_intact_pp"]  = (X[curr_pp].fillna(-1) == 2).sum(axis=1) if curr_pp else 0
    X["n_known_sens"] = X[curr_lt + curr_pp].notna().sum(axis=1) if curr_lt + curr_pp else 0

    # --- Time features ---
    if "time" in X.columns:
        X["log_time"]      = np.log1p(X["time"])
        X["t_x_motor"]     = X["time"] * X["total_motor"]
        X["t_x_UEMS"]      = X["time"] * X["UEMS"]
        X["t_x_LEMS"]      = X["time"] * X["LEMS"]
        X["t_x_severity"]  = X["time"] * X["severity"]
        X["t_x_inj_level"] = X["time"] * X["w1_injury_level"]
        X["t_x_d_motor"]   = X["time"] * X["d_total_motor"]
        X["t_x_w1_intact"] = X["time"] * X["w1_n_intact_lt"]

    # Encode categoricals
    for col in X.select_dtypes(include=["object"]).columns:
        if col not in ["ID", "_orig_ID", "pid"]:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Drop raw current sensory (71.7% NaN)
    drop_cols = [c for c in X.columns
                 if any(x in c for x in ["ltl","ltr","ppl","ppr"])
                 and not c.startswith("w1_")]
    X = X.drop(columns=drop_cols + ["ID", "_orig_ID", "pid"], errors="ignore")

    return X.fillna(-1)


# Build feature matrices
X_orig     = build_features(features, meta, is_extra=False)
X_extra    = build_features(t1_extra, None, is_extra=True)
X_test_all = build_features(test_features, test_meta, is_extra=False)

print(f"Original Task2 features: {X_orig.shape}")
print(f"Extra Task1 features:    {X_extra.shape}")
print(f"Test features:           {X_test_all.shape}")

# Align columns (extra rows built from Task1 may differ slightly)
common_cols = [c for c in X_orig.columns if c in X_extra.columns]
X_orig  = X_orig[common_cols]
X_extra = X_extra[common_cols]
X_test_all = X_test_all[[c for c in common_cols if c in X_test_all.columns]]
# Fill any missing test cols with -1
for c in common_cols:
    if c not in X_test_all.columns:
        X_test_all[c] = -1
X_test_all = X_test_all[common_cols]

print(f"Aligned feature count: {len(common_cols)}")

# Full augmented training set
X_aug = pd.concat([X_orig, X_extra], axis=0).reset_index(drop=True)
y_aug = np.vstack([y.values, y_extra])
n_orig  = len(X_orig)
n_aug   = len(X_aug)
print(f"Augmented training rows: {n_aug} (original={n_orig}, extra={n_aug - n_orig})")


# =========================================
# 4  Models
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
    """Average P(class=1) across models."""
    probs = []
    for m in models:
        probs.append(m.predict_proba(X)[:, 1])
    return np.mean(probs, axis=0)


# =========================================
# 5  Hard baseline copy (current sparse sensory)
# =========================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds  = np.zeros((n_orig, len(target_cols)))
test_preds = np.zeros((len(X_test_all), len(target_cols)))
train_known_all = np.zeros((n_orig, len(target_cols)), dtype=bool)
test_known_all  = np.zeros((len(X_test_all), len(target_cols)), dtype=bool)

for t, target in enumerate(target_cols):
    if target in features.columns:
        base_tr = pd.to_numeric(features.set_index("ID")[target], errors="coerce").values
        train_known_all[:, t] = ~np.isnan(base_tr)
        oof_preds[train_known_all[:, t], t] = base_tr[train_known_all[:, t]]
    if target in test_features.columns:
        base_te = pd.to_numeric(test_features.set_index("ID")[target], errors="coerce").values
        test_known_all[:, t] = ~np.isnan(base_te)
        test_preds[test_known_all[:, t], t] = base_te[test_known_all[:, t]]

print(f"Hard copy rate (train): {train_known_all.mean():.3f}")


# =========================================
# 6  Train — augmented ORDINAL Cat + LGB + XGB
#    For each target:
#      binary A: P(score >= 1)
#      binary B: P(score >= 2)  [skip if target is binary 0/1]
#    Prediction = P(>=1) + P(>=2)  ->  range [0, 2]
#    KFold on ORIGINAL rows only.
#    Extra rows always in training set (never in val).
# =========================================
extra_idx = np.arange(n_orig, n_aug)

for t, target in enumerate(target_cols):
    y_t_orig = y.values[:, t]
    y_t_aug  = y_aug[:, t]

    train_unknown = ~train_known_all[:, t]
    test_unknown  = ~test_known_all[:, t]

    is_binary  = (y_t_orig.max() <= 1)
    n_thresh   = 1 if is_binary else 2

    oof_model       = np.zeros(n_orig)
    test_fold_preds = np.zeros((len(X_test_all), kf.n_splits))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_orig)):
        tr_aug_idx = np.concatenate([tr_idx, extra_idx])
        X_tr  = X_aug.iloc[tr_aug_idx]
        y_tr  = y_t_aug[tr_aug_idx]
        X_val = X_orig.iloc[val_idx]
        y_val = y_t_orig[val_idx]

        fold_val_pred  = np.zeros(len(val_idx))
        fold_test_pred = np.zeros(len(X_test_all))

        for threshold in range(1, n_thresh + 1):
            y_tr_bin  = (y_tr  >= threshold).astype(int)
            y_val_bin = (y_val >= threshold).astype(int)

            # Skip if only one class in training fold
            if len(np.unique(y_tr_bin)) < 2:
                fold_val_pred  += float(y_tr_bin[0])
                fold_test_pred += float(y_tr_bin[0])
                continue

            cat  = make_cat()
            lgbm = make_lgb()
            xgbm = make_xgb()

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

            fold_val_pred  += p_val
            fold_test_pred += p_test

        oof_model[val_idx]       = fold_val_pred
        test_fold_preds[:, fold] = fold_test_pred

    oof_preds[train_unknown, t]  = oof_model[train_unknown]
    test_preds[test_unknown, t]  = test_fold_preds[test_unknown].mean(axis=1)

    if (t + 1) % 20 == 0 or t < 3:
        rmse_t = np.sqrt(mean_squared_error(
            y_t_orig, np.clip(oof_preds[:, t], y[target].min(), y[target].max())
        ))
        print(f"[{t+1:3d}/{len(target_cols)}] {target:10s}  RMSE={rmse_t:.4f}  "
              f"(binary={is_binary}, model n={train_unknown.sum()})")


# =========================================
# 7  Score + submit
# =========================================
label_min = y.min().values
label_max = y.max().values
oof_preds  = np.clip(oof_preds,  label_min, label_max)
test_preds = np.clip(test_preds, label_min, label_max)

rmse_final = np.sqrt(mean_squared_error(y.values, oof_preds))
print(f"\nFinal CV RMSE: {rmse_final:.5f}")

worst5 = sorted(
    {col: np.sqrt(mean_squared_error(y[col], oof_preds[:, t]))
     for t, col in enumerate(target_cols)}.items(),
    key=lambda x: x[1], reverse=True
)[:5]
print("Hardest 5:", worst5)

submission[target_cols] = test_preds
submission.to_csv("/kaggle/working/submission_task2_v6.csv", index=False)
print("Submission saved as submission_task2_v6.csv!")
