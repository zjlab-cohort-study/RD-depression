# -*- coding: utf-8 -*-
import os
import glob
import json
import argparse
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore")


def refine_lasso(X, y):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l1", solver="saga", C=0.1,
            max_iter=5000, random_state=42
        ))
    ])
    pipe.fit(X, y)
    coef = np.abs(pipe.named_steps["clf"].coef_[0])
    return pd.DataFrame({"feature": X.columns, "score": coef}).sort_values(["score", "feature"], ascending=[False, True])


def refine_enet(X, y):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet", solver="saga", C=0.1, l1_ratio=0.5,
            max_iter=5000, random_state=42
        ))
    ])
    pipe.fit(X, y)
    coef = np.abs(pipe.named_steps["clf"].coef_[0])
    return pd.DataFrame({"feature": X.columns, "score": coef}).sort_values(["score", "feature"], ascending=[False, True])


def refine_linear_svm_l1(X, y):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(
            penalty="l1", dual=False, C=0.05,
            max_iter=5000, random_state=42
        ))
    ])
    pipe.fit(X, y)
    coef = np.abs(pipe.named_steps["clf"].coef_[0])
    return pd.DataFrame({"feature": X.columns, "score": coef}).sort_values(["score", "feature"], ascending=[False, True])


def refine_rf(X, y):
    clf = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=5,
        random_state=42, n_jobs=-1, class_weight="balanced_subsample"
    )
    clf.fit(X, y)
    imp = clf.feature_importances_
    return pd.DataFrame({"feature": X.columns, "score": imp}).sort_values(["score", "feature"], ascending=[False, True])


def refine_et(X, y):
    clf = ExtraTreesClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=5,
        random_state=42, n_jobs=-1, class_weight="balanced_subsample"
    )
    clf.fit(X, y)
    imp = clf.feature_importances_
    return pd.DataFrame({"feature": X.columns, "score": imp}).sort_values(["score", "feature"], ascending=[False, True])


def refine_mi(X, y):
    mi = mutual_info_classif(X, y, random_state=42)
    return pd.DataFrame({"feature": X.columns, "score": mi}).sort_values(["score", "feature"], ascending=[False, True])


def refine_xgb_like_gbdt(X, y):
    clf = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.03,
        max_iter=300,
        random_state=42
    )
    clf.fit(X, y)
    
    rows = []
    for c in X.columns:
        try:
            x = X[c].values.astype(float)
            x = np.nan_to_num(x, nan=np.nanmedian(x))
            order = np.argsort(x)
            # 用分位离散化 proxy
            score = np.abs(np.corrcoef(x, y)[0, 1])
            if np.isnan(score):
                score = 0.0
            rows.append((c, float(score)))
        except Exception:
            rows.append((c, 0.0))
    return pd.DataFrame(rows, columns=["feature", "score"]).sort_values(["score", "feature"], ascending=[False, True])


def refine_ridge(X, y):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2", solver="liblinear", C=1.0,
            max_iter=5000, random_state=42
        ))
    ])
    pipe.fit(X, y)
    coef = np.abs(pipe.named_steps["clf"].coef_[0])
    return pd.DataFrame({"feature": X.columns, "score": coef}).sort_values(["score", "feature"], ascending=[False, True])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="earlyRD_clean_for_screening.csv"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="earlyRD_protein_ml"
    )
    parser.add_argument("--coarse_topk", type=int, default=200)
    parser.add_argument("--refine_topk", type=int, default=20)
    args = parser.parse_args()

    step3_dir = os.path.join(args.outdir, "results", "step3_refine")
    feat_dir = os.path.join(args.outdir, "data", "feature_sets")
    os.makedirs(step3_dir, exist_ok=True)
    os.makedirs(feat_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    cols = list(df.columns)
    y_col = cols[2]
    y = df[y_col].astype(int)

    coarse_files = sorted(glob.glob(os.path.join(feat_dir, f"coarse_*_top{args.coarse_topk}.csv")))

    refine_methods = {
        "lasso": refine_lasso,
        "enet": refine_enet,
        "linear_svm_l1": refine_linear_svm_l1,
        "rf": refine_rf,
        "et": refine_et,
        "mutual_info_refine": refine_mi,
        "xgb_like_gbdt": refine_xgb_like_gbdt,
        "ridge": refine_ridge
    }

    summary_rows = []

    for cf in coarse_files:
        coarse_name = os.path.basename(cf).replace("coarse_", "").replace(f"_top{args.coarse_topk}.csv", "")
        coarse_feats = pd.read_csv(cf)["feature"].tolist()
        X = df[coarse_feats].copy()

        for refine_name, func in refine_methods.items():
            print(f"[INFO] coarse={coarse_name} refine={refine_name}")
            rank_df = func(X, y)

            rank_file = os.path.join(step3_dir, f"refine_rank__coarse-{coarse_name}__refine-{refine_name}.csv")
            rank_df.to_csv(rank_file, index=False)

            top_feats = rank_df["feature"].head(args.refine_topk).tolist()
            feat_file = os.path.join(
                feat_dir,
                f"feature_set__coarse-{coarse_name}__refine-{refine_name}__top{args.refine_topk}.csv"
            )
            pd.DataFrame({"feature": top_feats}).to_csv(feat_file, index=False)

            summary_rows.append({
                "coarse_method": coarse_name,
                "refine_method": refine_name,
                "coarse_topk": args.coarse_topk,
                "refine_topk": args.refine_topk,
                "rank_file": rank_file,
                "feature_file": feat_file,
                "n_features": len(top_feats)
            })

    pd.DataFrame(summary_rows).to_csv(os.path.join(step3_dir, "refine_summary.csv"), index=False)

    with open(os.path.join(step3_dir, "refine_methods.json"), "w", encoding="utf-8") as f:
        json.dump({
            "refine_methods": list(refine_methods.keys()),
            "coarse_topk": args.coarse_topk,
            "refine_topk": args.refine_topk
        }, f, indent=2, ensure_ascii=False)

    print(f"[INFO] feature sets: {len(summary_rows)}")
    print("[INFO] finished Step3")


if __name__ == "__main__":
    main()
