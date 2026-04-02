# -*- coding: utf-8 -*-
import os
import glob
import json
import argparse
import warnings
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, f1_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")


def get_models():
    models = {
        "LR_L2": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2", solver="liblinear", C=1.0,
                class_weight="balanced", max_iter=5000, random_state=42
            ))
        ]),

        "LR_L1": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l1", solver="liblinear", C=0.1,
                class_weight="balanced", max_iter=5000, random_state=42
            ))
        ]),

        "LR_ElasticNet": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="elasticnet", solver="saga", C=0.1, l1_ratio=0.5,
                class_weight="balanced", max_iter=5000, random_state=42
            ))
        ]),

        "LinearSVM_Cal": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                estimator=LinearSVC(C=0.1, class_weight="balanced", max_iter=5000, random_state=42),
                method="sigmoid",
                cv=3
            ))
        ]),

        "SVM_RBF_Cal": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                estimator=SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=False, random_state=42),
                method="sigmoid",
                cv=3
            ))
        ]),

        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=500, min_samples_leaf=5,
                class_weight="balanced_subsample", n_jobs=-1, random_state=42
            ))
        ]),

        "ExtraTrees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", ExtraTreesClassifier(
                n_estimators=500, min_samples_leaf=5,
                class_weight="balanced_subsample", n_jobs=-1, random_state=42
            ))
        ]),

        "HistGBDT": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(
                max_depth=4, learning_rate=0.03, max_iter=200,
                random_state=42
            ))
        ]),

        "KNN": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=15))
        ]),

        "MLP": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                alpha=1e-3, learning_rate_init=1e-3,
                max_iter=500, random_state=42
            ))
        ]),
    }
    return models


def get_scores(y_true, y_prob, y_pred):
    auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "acc": acc,
        "f1": f1,
        "sensitivity": sens,
        "specificity": spec
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="/home/wangzhaoxiang/RD_RD2DEP_ML/RE_DP_NEW/earlyRD_protein_ml/data/processed/earlyRD_clean_for_screening.csv"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="/home/wangzhaoxiang/RD_RD2DEP_ML/RE_DP_NEW/earlyRD_protein_ml"
    )
    parser.add_argument("--feature_topk", type=int, default=20)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_covariates", action="store_true")
    args = parser.parse_args()

    step4_dir = os.path.join(args.outdir, "results", "step4_modeling")
    feat_dir = os.path.join(args.outdir, "data", "feature_sets")
    os.makedirs(step4_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    cols = list(df.columns)

    y_col = cols[2]
    covar_cols = cols[17:27]
    y = df[y_col].astype(int).values

    feature_files = sorted(glob.glob(
        os.path.join(feat_dir, f"feature_set__coarse-*__refine-*__top{args.feature_topk}.csv")
    ))

    models = get_models()
    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    all_rows = []

    for ff in feature_files:
        tag = os.path.basename(ff).replace(".csv", "")
        prot_feats = pd.read_csv(ff)["feature"].tolist()

        if args.use_covariates:
            x_cols = covar_cols + prot_feats
        else:
            x_cols = prot_feats

        X = df[x_cols].copy()

        for model_name, model in models.items():
            print(f"[INFO] feature_set={tag} | model={model_name} | n_features={len(x_cols)}")

            fold_rows = []
            oof_prob = np.zeros(len(df), dtype=float)
            oof_pred = np.zeros(len(df), dtype=int)

            for fold_id, (tr_idx, te_idx) in enumerate(cv.split(X, y), start=1):
                Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
                ytr, yte = y[tr_idx], y[te_idx]

                clf = clone(model)
                clf.fit(Xtr, ytr)

                prob = clf.predict_proba(Xte)[:, 1]
                pred = (prob >= 0.5).astype(int)

                oof_prob[te_idx] = prob
                oof_pred[te_idx] = pred

                res = get_scores(yte, prob, pred)
                res.update({
                    "feature_set": tag,
                    "model": model_name,
                    "fold": fold_id,
                    "n_features": len(x_cols)
                })
                fold_rows.append(res)

            fold_df = pd.DataFrame(fold_rows)
            fold_df.to_csv(
                os.path.join(step4_dir, f"cv_detail__{tag}__{model_name}.csv"),
                index=False
            )

            overall = get_scores(y, oof_prob, oof_pred)
            summary = {
                "feature_set": tag,
                "model": model_name,
                "n_features": len(x_cols),
                "auc_cv_mean": fold_df["auc"].mean(),
                "auc_cv_std": fold_df["auc"].std(ddof=1),
                "pr_auc_cv_mean": fold_df["pr_auc"].mean(),
                "pr_auc_cv_std": fold_df["pr_auc"].std(ddof=1),
                "acc_cv_mean": fold_df["acc"].mean(),
                "f1_cv_mean": fold_df["f1"].mean(),
                "sens_cv_mean": fold_df["sensitivity"].mean(),
                "spec_cv_mean": fold_df["specificity"].mean(),
                "auc_oof": overall["auc"],
                "pr_auc_oof": overall["pr_auc"],
                "acc_oof": overall["acc"],
                "f1_oof": overall["f1"],
                "sens_oof": overall["sensitivity"],
                "spec_oof": overall["specificity"],
            }
            all_rows.append(summary)

    res_df = pd.DataFrame(all_rows).sort_values(
        ["auc_oof", "pr_auc_oof", "auc_cv_mean"],
        ascending=[False, False, False]
    )
    res_df.to_csv(os.path.join(step4_dir, "all_model_results.csv"), index=False)

    best_per_feature_set = res_df.sort_values(
        ["feature_set", "auc_oof", "pr_auc_oof"],
        ascending=[True, False, False]
    ).groupby("feature_set", as_index=False).head(1)
    best_per_feature_set.to_csv(os.path.join(step4_dir, "best_model_per_feature_set.csv"), index=False)

    best_overall = res_df.head(50)
    best_overall.to_csv(os.path.join(step4_dir, "top50_overall_results.csv"), index=False)

    with open(os.path.join(step4_dir, "modeling_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "n_feature_sets": len(feature_files),
            "n_models": len(models),
            "n_total_runs": len(feature_files) * len(models),
            "use_covariates": args.use_covariates,
            "n_splits": args.n_splits,
            "seed": args.seed
        }, f, indent=2, ensure_ascii=False)

    print("[INFO] finished Step4")
    print(os.path.join(step4_dir, "all_model_results.csv"))


if __name__ == "__main__":
    main()
