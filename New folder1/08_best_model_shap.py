# -*- coding: utf-8 -*-
import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import shap


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
                method="sigmoid", cv=3
            ))
        ]),
        "SVM_RBF_Cal": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                estimator=SVC(C=1.0, kernel="rbf", gamma="scale",
                              class_weight="balanced", probability=False, random_state=42),
                method="sigmoid", cv=3
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
                max_depth=4, learning_rate=0.03, max_iter=200, random_state=42
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


def get_best_run(results_csv):
    df = pd.read_csv(results_csv)
    df = df.sort_values(["auc_oof", "pr_auc_oof", "auc_cv_mean"], ascending=[False, False, False]).reset_index(drop=True)
    return df.iloc[0].to_dict()


def unwrap_model_for_shap(fitted_model):
    if isinstance(fitted_model, Pipeline):
        return fitted_model
    return fitted_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="earlyRD_clean_for_screening.csv"
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        default="feature_sets"
    )
    parser.add_argument(
        "--results_csv",
        type=str,
        default="all_model_results.csv"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="step7_best_model_shap"
    )
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--background_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    best = get_best_run(args.results_csv)
    best_feature_set = best["feature_set"]
    best_model_name = best["model"]

    feature_file = os.path.join(args.feature_dir, f"{best_feature_set}.csv")
    feat_df = pd.read_csv(feature_file)
    features = feat_df["feature"].astype(str).tolist()[:20]

    df = pd.read_csv(args.input)
    X = df[features].copy()
    y = df.iloc[:, 2].astype(int).values

    if len(X) > args.max_samples:
        idx = rng.choice(np.arange(len(X)), size=args.max_samples, replace=False)
        X_use = X.iloc[idx].copy()
        y_use = y[idx]
    else:
        X_use = X.copy()
        y_use = y.copy()

    model = get_models()[best_model_name]
    model.fit(X_use, y_use)

    if len(X_use) > args.background_samples:
        bg_idx = rng.choice(np.arange(len(X_use)), size=args.background_samples, replace=False)
        X_bg = X_use.iloc[bg_idx].copy()
    else:
        X_bg = X_use.copy()

    def predict_proba_fn(data):
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=features)
        return model.predict_proba(data)[:, 1]

    try:
        explainer = shap.Explainer(predict_proba_fn, X_bg)
        shap_values = explainer(X_use)
        shap_mat = shap_values.values
        if shap_mat.ndim == 3:
            shap_mat = shap_mat[:, :, 0]
    except Exception:
        # fallback
        explainer = shap.KernelExplainer(predict_proba_fn, X_bg)
        shap_mat = explainer.shap_values(X_use, nsamples=100)
        if isinstance(shap_mat, list):
            shap_mat = shap_mat[0]
        shap_values = shap.Explanation(
            values=shap_mat,
            base_values=np.repeat(np.mean(predict_proba_fn(X_bg)), X_use.shape[0]),
            data=X_use.values,
            feature_names=features
        )

    shap_df = pd.DataFrame(shap_mat, columns=features)
    shap_df.to_csv(os.path.join(args.outdir, "best_model_shap_values.csv"), index=False)

    mean_abs_shap = np.mean(np.abs(shap_mat), axis=0)
    shap_imp = pd.DataFrame({
        "feature": features,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(["mean_abs_shap", "feature"], ascending=[False, True])
    shap_imp.to_csv(os.path.join(args.outdir, "best_model_shap_importance.csv"), index=False)

    plt.figure(figsize=(8, 6), dpi=160)
    shap.summary_plot(shap_mat, X_use, feature_names=features, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "shap_summary_beeswarm.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(args.outdir, "shap_summary_beeswarm.pdf"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6), dpi=160)
    shap.summary_plot(shap_mat, X_use, feature_names=features, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "shap_summary_bar.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(args.outdir, "shap_summary_bar.pdf"), bbox_inches="tight")
    plt.close()

    summary = {
        "best_feature_set": best_feature_set,
        "best_model": best_model_name,
        "n_features": len(features),
        "features": features,
        "n_samples_used_for_shap": int(len(X_use)),
        "background_samples": int(len(X_bg))
    }
    with open(os.path.join(args.outdir, "best_model_shap_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[INFO] best feature set:", best_feature_set)
    print("[INFO] best model:", best_model_name)
    print("[INFO] SHAP output dir:", args.outdir)


if __name__ == "__main__":
    main()
