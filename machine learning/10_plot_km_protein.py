# -*- coding: utf-8 -*-
import os
import json
import math
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import gridspec
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

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

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


def detect_age_sex(covar_cols):
    age_col, sex_col = None, None
    for c in covar_cols:
        cl = c.lower()
        if age_col is None and "age" in cl:
            age_col = c
        if sex_col is None and ("sex" in cl or "gender" in cl):
            sex_col = c
    return age_col, sex_col


def get_oof_predictions(model, X, y, n_splits=5, seed=42):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_prob = np.zeros(len(y), dtype=float)
    fold_id = np.zeros(len(y), dtype=int)

    for i, (tr_idx, te_idx) in enumerate(cv.split(X, y), start=1):
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr = y[tr_idx]
        clf = clone(model)
        clf.fit(Xtr, ytr)
        oof_prob[te_idx] = clf.predict_proba(Xte)[:, 1]
        fold_id[te_idx] = i

    return oof_prob, fold_id


def choose_time_grid(max_time, n_points=8):
    if max_time <= 5:
        step = 1.0
    elif max_time <= 10:
        step = 2.0
    elif max_time <= 20:
        step = 2.5
    else:
        step = 5.0
    end = math.ceil(max_time / step) * step
    times = np.arange(0, end + 1e-8, step)
    if len(times) < n_points:
        times = np.linspace(0, end, n_points)
    return np.round(times, 2)


def get_group_table(df, time_col, event_col, group_col, time_grid):
    rows = []
    for g in [0, 1]:
        dg = df[df[group_col] == g].copy()
        for t in time_grid:
            at_risk = int((dg[time_col] >= t).sum())
            cum_event = int(((dg[event_col] == 1) & (dg[time_col] <= t)).sum())
            rows.append({
                "group": g,
                "time": t,
                "n_at_risk": at_risk,
                "cum_events": cum_event
            })
    return pd.DataFrame(rows)


def format_p_math(p):
    if p < 1e-3:
        exp = int(np.floor(np.log10(p)))
        mant = p / (10 ** exp)
        return rf"$p$ = {mant:.2f} $\times$ 10$^{{{exp}}}$"
    return rf"$p$ = {p:.3f}"


def fit_continuous_cox(df, time_col, event_col, risk_col, age_col=None, sex_col=None):
    use_cols = [time_col, event_col, risk_col]
    if age_col is not None:
        use_cols.append(age_col)
    if sex_col is not None:
        use_cols.append(sex_col)

    dx = df[use_cols].copy()
    for c in use_cols:
        dx[c] = pd.to_numeric(dx[c], errors="coerce")
    dx = dx.dropna().copy()

    dx[risk_col] = (dx[risk_col] - dx[risk_col].mean()) / dx[risk_col].std(ddof=0)

    cph = CoxPHFitter()
    try:
        cph.fit(dx, duration_col=time_col, event_col=event_col)
        adjusted = True if (age_col is not None or sex_col is not None) else False
    except Exception:
        dx2 = dx[[time_col, event_col, risk_col]].copy()
        cph.fit(dx2, duration_col=time_col, event_col=event_col)
        adjusted = False

    hr = float(np.exp(cph.params_[risk_col]))
    ci_low = float(np.exp(cph.confidence_intervals_.loc[risk_col].iloc[0]))
    ci_high = float(np.exp(cph.confidence_intervals_.loc[risk_col].iloc[1]))
    pval = float(cph.summary.loc[risk_col, "p"])

    return {
        "hr": hr,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "pvalue": pval,
        "adjusted_age_sex": adjusted
    }


def get_adaptive_ylim(kmf_low, kmf_high):
    low_min = float(kmf_low.survival_function_.iloc[:, 0].min())
    high_min = float(kmf_high.survival_function_.iloc[:, 0].min())
    ymin = min(low_min, high_min)

    pad = 0.008
    ylow = ymin - pad
    ylow = max(0.0, ylow)

    # 让刻度更好看
    ylow = np.floor(ylow * 1000) / 1000

    if 0.94 <= ylow <= 0.999:
        step = 0.01
        ylow = np.floor(ylow / step) * step
    elif 0.85 <= ylow < 0.94:
        step = 0.02
        ylow = np.floor(ylow / step) * step
    else:
        step = 0.05
        ylow = np.floor(ylow / step) * step

    yhigh = 1.002
    return ylow, yhigh


def plot_km_publication(ax, table_ax, df_panel, time_col, event_col, group_col,
                        title, hr_text, p_text, time_grid):
    low_color = "#1f77b4"
    high_color = "#f0851f"

    d_low = df_panel[df_panel[group_col] == 0].copy()
    d_high = df_panel[df_panel[group_col] == 1].copy()

    kmf_low = KaplanMeierFitter()
    kmf_high = KaplanMeierFitter()

    kmf_low.fit(d_low[time_col], event_observed=d_low[event_col], label="Low-risk group")
    kmf_high.fit(d_high[time_col], event_observed=d_high[event_col], label="High-risk group")

    kmf_low.plot_survival_function(
        ax=ax, ci_show=True, color=low_color, ci_alpha=0.15, linewidth=3.0
    )
    kmf_high.plot_survival_function(
        ax=ax, ci_show=True, color=high_color, ci_alpha=0.22, linewidth=3.0
    )

    ax.set_title(title, fontsize=24, pad=16)
    ax.set_xlabel("Timeline", fontsize=18)
    ax.set_ylabel("Survival probability", fontsize=24)

    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    ax.tick_params(axis="both", labelsize=16, width=1.8, length=5)

    ylow, yhigh = get_adaptive_ylim(kmf_low, kmf_high)
    ax.set_ylim(ylow, yhigh)

    xmax = max(df_panel[time_col].max(), np.max(time_grid))
    ax.set_xlim(-0.05 * xmax, xmax * 1.03)

    leg = ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 0.88),
        frameon=False,
        fontsize=18,
        handlelength=1.6,
        handletextpad=0.6
    )
    handles = getattr(leg, "legendHandles", None)
    if handles is None:
        handles = getattr(leg, "legend_handles", [])
    for lh in handles:
        try:
            lh.set_linewidth(3.0)
        except Exception:
            pass

    ax.text(
        0.05, 0.10,
        hr_text + "\n" + p_text,
        transform=ax.transAxes,
        fontsize=20,
        ha="left",
        va="bottom"
    )

    # risk table
    tab = get_group_table(df_panel, time_col, event_col, group_col, time_grid)

    table_ax.set_xlim(ax.get_xlim())
    table_ax.set_ylim(0, 1)
    table_ax.axis("off")

    x0, x1 = ax.get_xlim()
    xr = x1 - x0
    line_x0 = x0 - 0.11 * xr
    line_x1 = x0 - 0.03 * xr

    y_low_at, y_low_ev = 0.72, 0.54
    y_high_at, y_high_ev = 0.30, 0.12

    table_ax.plot([line_x0, line_x1], [y_low_at, y_low_at], color=low_color, lw=2.4, clip_on=False)
    table_ax.plot([line_x0, line_x1], [y_high_at, y_high_at], color=high_color, lw=2.4, clip_on=False)

    low_tab = tab[tab["group"] == 0].copy()
    high_tab = tab[tab["group"] == 1].copy()

    for t in time_grid:
        row_la = low_tab[low_tab["time"] == t].iloc[0]
        row_ha = high_tab[high_tab["time"] == t].iloc[0]

        table_ax.text(t, y_low_at, f"{int(row_la['n_at_risk']):,}", ha="center", va="center", fontsize=13)
        table_ax.text(t, y_low_ev, f"{int(row_la['cum_events']):,}", ha="center", va="center", fontsize=13)

        table_ax.text(t, y_high_at, f"{int(row_ha['n_at_risk']):,}", ha="center", va="center", fontsize=13)
        table_ax.text(t, y_high_ev, f"{int(row_ha['cum_events']):,}", ha="center", va="center", fontsize=13)


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
        default="step8_survival_protein"
    )
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cutoff_mode", type=str, default="median", choices=["median", "quantile"])
    parser.add_argument("--quantile", type=float, default=0.5)
    parser.add_argument("--title", type=str, default="Protein panel model")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    cols = list(df.columns)

    eid_col = cols[0]
    label_col = cols[2]
    time_col = cols[3]
    covar_cols = cols[17:27]

    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    for c in covar_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df[label_col].isin([0, 1])].copy()
    df = df[df[time_col].notna()].copy()

    y = df[label_col].astype(int).values

    best = get_best_run(args.results_csv)
    best_feature_set = best["feature_set"]
    best_model_name = best["model"]
    model = get_models()[best_model_name]

    feature_file = os.path.join(args.feature_dir, f"{best_feature_set}.csv")
    feat_df = pd.read_csv(feature_file)
    protein20 = feat_df["feature"].astype(str).tolist()[:20]

    X = df[protein20].copy()
    oof_prob, fold_id = get_oof_predictions(model, X, y, n_splits=args.cv_folds, seed=args.seed)

    if args.cutoff_mode == "median":
        cutoff = float(np.median(oof_prob))
    else:
        cutoff = float(np.quantile(oof_prob, args.quantile))

    risk_group = (oof_prob >= cutoff).astype(int)

    age_col, sex_col = detect_age_sex(covar_cols)

    out_pred = df[[eid_col, label_col, time_col]].copy()
    out_pred["pred_prob_oof"] = oof_prob
    out_pred["risk_group"] = risk_group
    out_pred["cv_fold"] = fold_id
    out_pred.to_csv(os.path.join(args.outdir, "protein_only_oof_predictions_and_groups.csv"), index=False)

    cox_df = df[[time_col, label_col]].copy()
    cox_df["risk_score"] = oof_prob
    if age_col is not None:
        cox_df[age_col] = df[age_col].values
    if sex_col is not None:
        cox_df[sex_col] = df[sex_col].values

    cox_res = fit_continuous_cox(
        cox_df,
        time_col=time_col,
        event_col=label_col,
        risk_col="risk_score",
        age_col=age_col,
        sex_col=sex_col
    )

    lr = logrank_test(
        df.loc[risk_group == 0, time_col],
        df.loc[risk_group == 1, time_col],
        event_observed_A=df.loc[risk_group == 0, label_col],
        event_observed_B=df.loc[risk_group == 1, label_col]
    )

    panel_df = df[[time_col, label_col]].copy()
    panel_df["risk_group"] = risk_group

    time_grid = choose_time_grid(df[time_col].max(), n_points=8)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12
    })

    fig = plt.figure(figsize=(9.2, 8.6), dpi=200)
    gs = gridspec.GridSpec(
        2, 1,
        height_ratios=[4.8, 1.35],
        hspace=0.06
    )

    ax = fig.add_subplot(gs[0, 0])
    tab_ax = fig.add_subplot(gs[1, 0], sharex=ax)

    hr_text = f"HR per 1-SD risk = {cox_res['hr']:.2f} ({cox_res['ci_low']:.2f} - {cox_res['ci_high']:.2f})"
    p_text = format_p_math(cox_res["pvalue"])

    plot_km_publication(
        ax=ax,
        table_ax=tab_ax,
        df_panel=panel_df,
        time_col=time_col,
        event_col=label_col,
        group_col="risk_group",
        title=args.title,
        hr_text=hr_text,
        p_text=p_text,
        time_grid=time_grid
    )

    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "protein_only_km_publication.png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(args.outdir, "protein_only_km_publication.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(args.outdir, "protein_only_km_publication.svg"), bbox_inches="tight")
    plt.close(fig)

    risk_table = get_group_table(panel_df, time_col, label_col, "risk_group", time_grid)
    risk_table.to_csv(os.path.join(args.outdir, "protein_only_km_risk_table.csv"), index=False)

    summary = {
        "best_feature_set": best_feature_set,
        "best_model": best_model_name,
        "protein20": protein20,
        "cv_folds": args.cv_folds,
        "cutoff_mode": args.cutoff_mode,
        "cutoff": cutoff,
        "n_low_risk": int((risk_group == 0).sum()),
        "n_high_risk": int((risk_group == 1).sum()),
        "age_col_detected": age_col,
        "sex_col_detected": sex_col,
        "cox_continuous_per_1sd": {
            "hr": cox_res["hr"],
            "ci_low": cox_res["ci_low"],
            "ci_high": cox_res["ci_high"],
            "pvalue": cox_res["pvalue"],
            "adjusted_age_sex": cox_res["adjusted_age_sex"]
        },
        "logrank_pvalue_for_km_groups": float(lr.p_value)
    }

    with open(os.path.join(args.outdir, "protein_only_km_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[INFO] best feature set:", best_feature_set)
    print("[INFO] best model:", best_model_name)
    print("[INFO] cutoff mode:", args.cutoff_mode)
    print("[INFO] cutoff:", cutoff)
    print("[INFO] low/high:", int((risk_group == 0).sum()), int((risk_group == 1).sum()))
    print("[INFO] continuous HR per 1-SD:", cox_res["hr"])
    print("[INFO] 95% CI:", cox_res["ci_low"], cox_res["ci_high"])
    print("[INFO] p:", cox_res["pvalue"])
    print("[INFO] output dir:", args.outdir)


if __name__ == "__main__":
    main()
