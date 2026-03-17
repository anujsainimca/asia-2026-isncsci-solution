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
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds  = np.zeros((len(y), len(target_cols)))
test_preds = np.zeros((len(X_test_all), len(target_cols)))
train_known_all = np.zeros((len(y), len(target_cols)), dtype=bool)
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


# =========================================
# 5  Train ordinal model per target
# =========================================
for t, target in enumerate(target_cols):
    y_t = y[target].values
    train_unknown = ~train_known_all[:, t]
    test_unknown  = ~test_known_all[:, t]

    is_binary = (target == "anyana") or (y_t.max() <= 1)
    n_thresholds = 1 if is_binary else 2  # thresholds: >=1 (and >=2 if 0/1/2)

    oof_model       = np.zeros(len(y))
    test_fold_preds = np.zeros((len(X_test_all), kf.n_splits))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_all)):
        X_tr  = X_all.iloc[tr_idx]
        X_val = X_all.iloc[val_idx]
        y_tr  = y_t[tr_idx]
        y_val = y_t[val_idx]

        fold_val_pred  = np.zeros(len(val_idx))
        fold_test_pred = np.zeros(len(X_test_all))

        for threshold in range(1, n_thresholds + 1):
            # Binary target: 1 if score >= threshold, else 0
            y_tr_bin  = (y_tr  >= threshold).astype(int)
            y_val_bin = (y_val >= threshold).astype(int)

            # Skip if only one class in training fold
            if len(np.unique(y_tr_bin)) < 2:
                # All same class — predict the constant
                constant = float(y_tr_bin[0])
                fold_val_pred  += constant
                fold_test_pred += constant
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

            # P(score >= threshold) averaged across 3 models
            p_val  = predict_proba_avg([cat, lgbm, xgbm], X_val)
            p_test = predict_proba_avg([cat, lgbm, xgbm], X_test_all)

            fold_val_pred  += p_val
            fold_test_pred += p_test

        oof_model[val_idx]       = fold_val_pred
        test_fold_preds[:, fold] = fold_test_pred

    oof_preds[train_unknown, t]  = oof_model[train_unknown]
    test_preds[test_unknown, t]  = test_fold_preds[test_unknown].mean(axis=1)

    if (t + 1) % 20 == 0 or t < 3:
        rmse_t = np.sqrt(mean_squared_error(y_t, np.clip(oof_preds[:, t], 0, y_t.max())))
        print(f"[{t+1:3d}/{len(target_cols)}] {target:10s}  RMSE={rmse_t:.4f}  (binary={is_binary})")


# =========================================
# 6  Score + submit
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
submission.to_csv("submission_ordinal.csv", index=False)
print("Submission saved as submission_ordinal.csv!")
