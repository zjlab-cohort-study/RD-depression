# -*- coding: utf-8 -*-
import os
import re
import json
import math
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


# =========================
# Models
# =========================
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
                estimator=SVC(C=1.0, kernel="rbf", gamma="scale",
                              class_weight="balanced", probability=False, random_state=42),
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


# =========================
# Utils
# =========================
def parse_feature_set_name(tag):
    m = re.match(r"feature_set__coarse-(.*?)__refine-(.*?)__top(\d+)", tag)
    if m is None:
        return {"coarse": "unknown", "refine": "unknown", "topk": None}
    return {"coarse": m.group(1), "refine": m.group(2), "topk": int(m.group(3))}


def auc_delong_approx_test(y_true, pred1, pred2):
    
    auc1 = roc_auc_score(y_true, pred1)
    auc2 = roc_auc_score(y_true, pred2)
    diff = auc2 - auc1

    rng = np.random.RandomState(42)
    diffs = []
    n = len(y_true)

    for _ in range(300):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        yb = y_true[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            a1 = roc_auc_score(yb, pred1[idx])
            a2 = roc_auc_score(yb, pred2[idx])
            diffs.append(a2 - a1)
        except Exception:
            continue

    if len(diffs) < 30:
        return {"pvalue": 1.0, "auc1": auc1, "auc2": auc2, "diff": diff}

    diffs = np.array(diffs, dtype=float)
    mean_diff = np.mean(diffs)
    sd_diff = np.std(diffs, ddof=1)

    if sd_diff <= 1e-12:
        p = 1.0
    else:
        z = abs(mean_diff / sd_diff)
        # 双侧正态近似
        p = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))

    return {"pvalue": float(p), "auc1": float(auc1), "auc2": float(auc2), "diff": float(diff)}


def get_oof_auc(model, X, y, n_splits=10, seed=42):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_prob = np.zeros(len(y), dtype=float)

    for tr_idx, te_idx in cv.split(X, y):
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr = y[tr_idx]
        clf = clone(model)
        clf.fit(Xtr, ytr)
        prob = clf.predict_proba(Xte)[:, 1]
        oof_prob[te_idx] = prob

    auc = roc_auc_score(y, oof_prob)
    return auc, oof_prob


def bootstrap_auc_ci(y_true, y_prob, n_boot=500, seed=42):
    rng = np.random.RandomState(seed)
    vals = []
    n = len(y_true)

    for _ in range(n_boot):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        yb = y_true[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            vals.append(roc_auc_score(yb, y_prob[idx]))
        except Exception:
            continue

    if len(vals) < 30:
        auc = roc_auc_score(y_true, y_prob)
        return auc, auc, auc

    vals = np.array(vals, dtype=float)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 50)), float(np.percentile(vals, 97.5))


def get_best_run(results_csv):
    df = pd.read_csv(results_csv)
    df = df.sort_values(["auc_oof", "pr_auc_oof", "auc_cv_mean"], ascending=[False, False, False]).reset_index(drop=True)
    best = df.iloc[0].to_dict()
    return best


def get_ranked_features_by_permutation(model, X, y, seed=42):
    clf = clone(model)
    clf.fit(X, y)
    r = permutation_importance(
        clf, X, y,
        n_repeats=20,
        random_state=seed,
        scoring="roc_auc",
        n_jobs=1
    )
    imp = pd.DataFrame({
        "feature": X.columns,
        "importance": r.importances_mean
    }).sort_values(["importance", "feature"], ascending=[False, True]).reset_index(drop=True)
    return imp


def determine_stop_k_auc_delta(auc_list, delta=0.001, extra_steps=5, max_k=20):
   
    stop_k = None
    for k in range(1, len(auc_list) - 1):
        d1 = auc_list[k] - auc_list[k - 1]
        d2 = auc_list[k + 1] - auc_list[k]
        if (d1 < delta) and (d2 < delta):
            stop_k = k
            break

    if stop_k is None:
        stop_k = len(auc_list) - 1

    final_k = min(stop_k + extra_steps, max_k, len(auc_list))
    return stop_k + 1, final_k  # 返回1-based


def determine_stop_k_pvalue(y, pred_list, p_thresh=0.05, extra_steps=5, max_k=20):
   
    stop_k = None
    test_rows = []

    for k in range(0, len(pred_list) - 2):
        t1 = auc_delong_approx_test(y, pred_list[k], pred_list[k + 1])
        t2 = auc_delong_approx_test(y, pred_list[k + 1], pred_list[k + 2])

        test_rows.append({
            "k": k + 1,
            "compare": f"{k+1}vs{k+2}",
            "auc_prev": t1["auc1"],
            "auc_next": t1["auc2"],
            "auc_diff": t1["diff"],
            "pvalue": t1["pvalue"]
        })
        test_rows.append({
            "k": k + 1,
            "compare": f"{k+2}vs{k+3}",
            "auc_prev": t2["auc1"],
            "auc_next": t2["auc2"],
            "auc_diff": t2["diff"],
            "pvalue": t2["pvalue"]
        })

        if (t1["pvalue"] > p_thresh) and (t2["pvalue"] > p_thresh):
            stop_k = k
            break

    if stop_k is None:
        stop_k = len(pred_list) - 1

    final_k = min(stop_k + 1 + extra_steps, max_k, len(pred_list))
    return stop_k + 1, final_k, pd.DataFrame(test_rows)  # 1-based


def plot_forward_selection(
    out_png, out_pdf, eval_df, selected_k, final_k, title_note=""
):
    
    plot_df = eval_df.iloc[:final_k].copy().reset_index(drop=True)

    x = np.arange(len(plot_df))
    names = plot_df["feature"].tolist()
    imp = plot_df["importance"].values
    auc = plot_df["auc"].values
    ci_low = plot_df["auc_ci_low"].values
    ci_high = plot_df["auc_ci_high"].values

    plt.figure(figsize=(16, 6), dpi=160)
    ax1 = plt.gca()

    bar_colors = []
    label_colors = []
    for i in range(len(plot_df)):
        if i < selected_k:
            bar_colors.append("#355f8a")
            label_colors.append("red")
        else:
            bar_colors.append("#9bb6c9")
            label_colors.append("black")

    ax1.bar(x, imp, color=bar_colors, width=0.82)
    ax1.set_ylabel("Protein importance", fontsize=12)
    ax1.set_xlim(-0.6, len(plot_df) - 0.4)

    ax2 = ax1.twinx()
    ax2.plot(x, auc, color="black", linewidth=1.6, zorder=5)
    ax2.scatter(x[:selected_k], auc[:selected_k], s=95, color="red", zorder=6)
    if selected_k < len(plot_df):
        ax2.scatter(x[selected_k:], auc[selected_k:], s=32, facecolors="none", edgecolors="black", linewidth=1.2, zorder=6)

    ax2.fill_between(x, ci_low, ci_high, color="#e6a89a", alpha=0.28, zorder=1)
    ax2.set_ylabel("Cumulative AUC", fontsize=12, rotation=270, labelpad=25)

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=50, ha="right", fontsize=9)

    for tick, c in zip(ax1.get_xticklabels(), label_colors):
        tick.set_color(c)

    ax1.grid(True, axis="both", linestyle="--", alpha=0.35)
    ax1.set_axisbelow(True)

    ttl = "a"
    if title_note:
        ttl += f"   {title_note}"
    ax1.set_title(ttl, loc="left", fontsize=16, fontweight="bold", pad=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


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
        default="step5_forward_selection"
    )
    parser.add_argument("--max_features", type=int, default=20)
    parser.add_argument("--delta_auc", type=float, default=0.001)
    parser.add_argument("--p_thresh", type=float, default=0.05)
    parser.add_argument("--extra_after_stop", type=int, default=5)
    parser.add_argument("--cv_folds", type=int, default=10)
    parser.add_argument("--bootstrap_n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1. best run
    best = get_best_run(args.results_csv)
    best_feature_set = best["feature_set"]
    best_model_name = best["model"]

    info = parse_feature_set_name(best_feature_set)
    feature_file = os.path.join(args.feature_dir, f"{best_feature_set}.csv")
    if not os.path.exists(feature_file):
        raise FileNotFoundError(feature_file)

    feat_df = pd.read_csv(feature_file)
    features = feat_df["feature"].astype(str).tolist()[:args.max_features]

    # 2. data
    df = pd.read_csv(args.input)
    cols = list(df.columns)
    y_col = cols[2]
    y = df[y_col].astype(int).values

    X = df[features].copy()

    # 3. model
    model = get_models()[best_model_name]

    # 4. permutation importance ranking
    imp_df = get_ranked_features_by_permutation(model, X, y, seed=args.seed)
    ranked_features = imp_df["feature"].tolist()[:args.max_features]
    imp_map = dict(zip(imp_df["feature"], imp_df["importance"]))

    # 5. forward evaluation for all k up to max_features
    rows = []
    pred_list = []

    for k in range(1, len(ranked_features) + 1):
        sub_feats = ranked_features[:k]
        Xk = df[sub_feats].copy()

        auc, oof_prob = get_oof_auc(model, Xk, y, n_splits=args.cv_folds, seed=args.seed)
        ci_low, ci_mid, ci_high = bootstrap_auc_ci(y, oof_prob, n_boot=args.bootstrap_n, seed=args.seed + k)

        rows.append({
            "k": k,
            "feature": ranked_features[k - 1],
            "importance": float(imp_map[ranked_features[k - 1]]),
            "auc": float(auc),
            "auc_ci_low": float(ci_low),
            "auc_ci_mid": float(ci_mid),
            "auc_ci_high": float(ci_high),
            "feature_list": ";".join(sub_feats)
        })
        pred_list.append(oof_prob.copy())

        print(f"[INFO] k={k:02d} auc={auc:.4f} feature={ranked_features[k-1]}")

    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(os.path.join(args.outdir, "forward_selection_full_evaluation.csv"), index=False)

    # 6A. rule by auc delta
    stop_k_auc, final_k_auc = determine_stop_k_auc_delta(
        eval_df["auc"].tolist(),
        delta=args.delta_auc,
        extra_steps=args.extra_after_stop,
        max_k=args.max_features
    )
    eval_df["selected_by_auc_delta"] = (eval_df["k"] <= stop_k_auc).astype(int)
    eval_df["plotted_by_auc_delta"] = (eval_df["k"] <= final_k_auc).astype(int)
    eval_df.to_csv(os.path.join(args.outdir, "forward_selection_auc_delta_rule.csv"), index=False)

    # 6B. rule by pvalue
    stop_k_p, final_k_p, ptest_df = determine_stop_k_pvalue(
        y=y,
        pred_list=pred_list,
        p_thresh=args.p_thresh,
        extra_steps=args.extra_after_stop,
        max_k=args.max_features
    )
    ptest_df.to_csv(os.path.join(args.outdir, "forward_selection_pvalue_tests.csv"), index=False)

    eval_df["selected_by_pvalue"] = (eval_df["k"] <= stop_k_p).astype(int)
    eval_df["plotted_by_pvalue"] = (eval_df["k"] <= final_k_p).astype(int)
    eval_df.to_csv(os.path.join(args.outdir, "forward_selection_pvalue_rule.csv"), index=False)

    # 7. plots
    plot_forward_selection(
        out_png=os.path.join(args.outdir, "forward_selection_plot_auc_delta001.png"),
        out_pdf=os.path.join(args.outdir, "forward_selection_plot_auc_delta001.pdf"),
        eval_df=eval_df,
        selected_k=stop_k_auc,
        final_k=final_k_auc,
        title_note=f"Best set: {info['coarse']} + {info['refine']} | Best model: {best_model_name} | rule: ΔAUC<{args.delta_auc} then +{args.extra_after_stop}"
    )

    plot_forward_selection(
        out_png=os.path.join(args.outdir, "forward_selection_plot_pvalue005.png"),
        out_pdf=os.path.join(args.outdir, "forward_selection_plot_pvalue005.pdf"),
        eval_df=eval_df,
        selected_k=stop_k_p,
        final_k=final_k_p,
        title_note=f"Best set: {info['coarse']} + {info['refine']} | Best model: {best_model_name} | rule: P>{args.p_thresh} twice then +{args.extra_after_stop}"
    )

    # 8. summary
    summary = {
        "best_feature_set": best_feature_set,
        "best_model": best_model_name,
        "best_auc_oof_from_step4": float(best["auc_oof"]),
        "coarse_method": info["coarse"],
        "refine_method": info["refine"],
        "ranked_features": ranked_features,

        "auc_delta_rule": {
            "delta_auc": args.delta_auc,
            "stop_k": int(stop_k_auc),
            "final_k_for_plot": int(final_k_auc),
            "selected_features": ranked_features[:stop_k_auc],
            "plotted_features": ranked_features[:final_k_auc]
        },

        "pvalue_rule": {
            "p_thresh": args.p_thresh,
            "stop_k": int(stop_k_p),
            "final_k_for_plot": int(final_k_p),
            "selected_features": ranked_features[:stop_k_p],
            "plotted_features": ranked_features[:final_k_p]
        }
    }

    with open(os.path.join(args.outdir, "forward_selection_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[INFO] done")
    print("[INFO] best feature set:", best_feature_set)
    print("[INFO] best model:", best_model_name)
    print("[INFO] auc-delta stop_k:", stop_k_auc, " final_k:", final_k_auc)
    print("[INFO] pvalue stop_k:", stop_k_p, " final_k:", final_k_p)
    print("[INFO] output dir:", args.outdir)


if __name__ == "__main__":
    main()
