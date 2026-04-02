# -*- coding: utf-8 -*-
import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.stats import ranksums
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


def rank_by_mutual_info(X, y):
    mi = mutual_info_classif(X, y, random_state=42)
    out = pd.DataFrame({"feature": X.columns, "score": mi}).sort_values(["score", "feature"], ascending=[False, True])
    return out


def rank_by_anova_f(X, y):
    f_vals, p_vals = f_classif(X, y)
    out = pd.DataFrame({"feature": X.columns, "score": f_vals, "pvalue": p_vals}).sort_values(
        ["score", "feature"], ascending=[False, True]
    )
    return out


def rank_by_logit_p(X, y):
    rows = []
    for c in X.columns:
        try:
            xx = sm.add_constant(X[[c]])
            model = sm.Logit(y, xx).fit(disp=0)
            pval = float(model.pvalues[c])
            coef = float(model.params[c])
            score = -np.log10(max(pval, 1e-300))
            rows.append((c, score, pval, coef))
        except Exception:
            rows.append((c, -np.log10(1.0), 1.0, np.nan))
    out = pd.DataFrame(rows, columns=["feature", "score", "pvalue", "coef"]).sort_values(
        ["score", "feature"], ascending=[False, True]
    )
    return out


def rank_by_auc(X, y):
    rows = []
    yv = y.values
    for c in X.columns:
        x = X[c].values
        try:
            auc = roc_auc_score(yv, x)
            score = abs(auc - 0.5)
            rows.append((c, score, auc))
        except Exception:
            rows.append((c, 0.0, np.nan))
    out = pd.DataFrame(rows, columns=["feature", "score", "auc"]).sort_values(
        ["score", "feature"], ascending=[False, True]
    )
    return out


def rank_by_wilcoxon(X, y):
    rows = []
    idx0 = (y == 0).values
    idx1 = (y == 1).values
    for c in X.columns:
        try:
            x0 = X.loc[idx0, c].values
            x1 = X.loc[idx1, c].values
            stat, pval = ranksums(x1, x0)
            score = -np.log10(max(float(pval), 1e-300))
            rows.append((c, score, stat, pval))
        except Exception:
            rows.append((c, 0.0, np.nan, 1.0))
    out = pd.DataFrame(rows, columns=["feature", "score", "stat", "pvalue"]).sort_values(
        ["score", "feature"], ascending=[False, True]
    )
    return out


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
    parser.add_argument("--topk", type=int, default=200)
    args = parser.parse_args()

    step2_dir = os.path.join(args.outdir, "results", "step2_coarse")
    feat_dir = os.path.join(args.outdir, "data", "feature_sets")
    os.makedirs(step2_dir, exist_ok=True)
    os.makedirs(feat_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    cols = list(df.columns)
    y_col = cols[2]
    protein_cols = cols[27:]

    X = df[protein_cols].copy()
    y = df[y_col].astype(int).copy()

    methods = {
        "mutual_info": rank_by_mutual_info,
        "anova_f": rank_by_anova_f,
        "logit_p": rank_by_logit_p,
        "auc_rank": rank_by_auc,
        "wilcoxon_rank": rank_by_wilcoxon,
    }

    summary_rows = []

    for name, func in methods.items():
        print(f"[INFO] coarse method: {name}")
        rank_df = func(X, y)
        rank_path = os.path.join(step2_dir, f"coarse_rank_{name}.csv")
        rank_df.to_csv(rank_path, index=False)

        top_feats = rank_df["feature"].head(args.topk).tolist()
        feature_path = os.path.join(feat_dir, f"coarse_{name}_top{args.topk}.csv")
        pd.DataFrame({"feature": top_feats}).to_csv(feature_path, index=False)

        summary_rows.append({
            "method": name,
            "topk": args.topk,
            "rank_file": rank_path,
            "feature_file": feature_path,
            "n_features": len(top_feats)
        })

    pd.DataFrame(summary_rows).to_csv(os.path.join(step2_dir, "coarse_summary.csv"), index=False)

    with open(os.path.join(step2_dir, "coarse_methods.json"), "w", encoding="utf-8") as f:
        json.dump({"methods": list(methods.keys()), "topk": args.topk}, f, indent=2, ensure_ascii=False)

    print("[INFO] finished Step2")


if __name__ == "__main__":
    main()
