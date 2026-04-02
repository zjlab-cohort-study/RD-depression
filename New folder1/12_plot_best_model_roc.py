# -*- coding: utf-8 -*-
import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

warnings.filterwarnings("ignore")


def bootstrap_auc_ci(y, prob, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    n = len(y)

    for _ in range(n_boot):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        yb = y[idx]
        pb = prob[idx]

        if len(np.unique(yb)) < 2:
            continue

        try:
            aucs.append(roc_auc_score(yb, pb))
        except Exception:
            continue

    auc = roc_auc_score(y, prob)
    aucs = np.array(aucs, dtype=float)

    if len(aucs) < 30:
        return auc, auc, auc, aucs

    low = float(np.percentile(aucs, 2.5))
    high = float(np.percentile(aucs, 97.5))
    return auc, low, high, aucs


def bootstrap_roc_ci_band(y, prob, n_boot=2000, seed=42, n_grid=400):
    rng = np.random.RandomState(seed)
    base_fpr = np.linspace(0, 1, n_grid)
    tprs = []
    n = len(y)

    for _ in range(n_boot):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        yb = y[idx]
        pb = prob[idx]

        if len(np.unique(yb)) < 2:
            continue

        try:
            fpr, tpr, _ = roc_curve(yb, pb)
            interp_tpr = np.interp(base_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tpr[-1] = 1.0
            interp_tpr = np.maximum.accumulate(interp_tpr)
            tprs.append(interp_tpr)
        except Exception:
            continue

    if len(tprs) < 30:
        fpr, tpr, _ = roc_curve(y, prob)
        interp_tpr = np.interp(base_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0
        interp_tpr = np.maximum.accumulate(interp_tpr)
        return base_fpr, interp_tpr, interp_tpr, interp_tpr

    tprs = np.array(tprs, dtype=float)

    low_tpr = np.percentile(tprs, 2.5, axis=0)
    high_tpr = np.percentile(tprs, 97.5, axis=0)

    low_tpr[0] = 0.0
    high_tpr[0] = 0.0
    low_tpr[-1] = 1.0
    high_tpr[-1] = 1.0

    low_tpr = np.maximum.accumulate(low_tpr)
    high_tpr = np.maximum.accumulate(high_tpr)
    high_tpr = np.maximum(high_tpr, low_tpr)

    return base_fpr, None, low_tpr, high_tpr


def make_publication_roc_plot(
    y,
    prob,
    auc,
    auc_low,
    auc_high,
    ci_fpr,
    ci_low_tpr,
    ci_high_tpr,
    out_png,
    out_pdf,
    out_svg,
    title="",
    panel_label="",
    curve_color="#11a84b",
    ci_alpha=0.16,
    diag_color="#f4a261",
    seed=42
):
    fpr, tpr, _ = roc_curve(y, prob)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "axes.linewidth": 1.8,
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
    })

    fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=180)

    # CI band
    ax.fill_between(
        ci_fpr,
        ci_low_tpr,
        ci_high_tpr,
        color=curve_color,
        alpha=ci_alpha,
        linewidth=0,
        zorder=1
    )

    # Diagonal line
    ax.plot(
        [-0.02, 1.02],
        [-0.02, 1.02],
        linestyle="--",
        color=diag_color,
        linewidth=2.0,
        dashes=(5, 3),
        zorder=0
    )

    # Main ROC
    ax.plot(
        fpr,
        tpr,
        color=curve_color,
        linewidth=2.8,
        zorder=3
    )

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")

    # ticks
    ticks = np.linspace(0, 1, 6)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # No grid
    ax.grid(False)

    # Spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.8)

    # Title
    if title:
        ax.set_title(title, pad=14)

    # Panel label
    if panel_label:
        ax.text(
            -0.18, 1.08, panel_label,
            transform=ax.transAxes,
            fontsize=28,
            fontweight="bold",
            va="top",
            ha="left"
        )

    # AUC annotation inside panel
    auc_text = f"AUC = {auc:.3f}\n({auc_low:.3f} - {auc_high:.3f})"
    ax.text(
        0.58, 0.13, auc_text,
        transform=ax.transAxes,
        fontsize=17,
        ha="left",
        va="bottom"
    )

    plt.tight_layout()
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oof",
        type=str,
        default="best_model_oof_predictions.csv"
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        default="best_model_empirical_roc_summary.json"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="step6_best_model_roc_publication"
    )
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--panel_label", type=str, default="")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.oof)
    y = df["y_true"].values.astype(int)
    prob = df["oof_prob"].values.astype(float)

    auc, auc_low, auc_high, boot_aucs = bootstrap_auc_ci(
        y=y,
        prob=prob,
        n_boot=args.n_boot,
        seed=args.seed
    )

    ci_fpr, _, ci_low_tpr, ci_high_tpr = bootstrap_roc_ci_band(
        y=y,
        prob=prob,
        n_boot=args.n_boot,
        seed=args.seed
    )

    # save CI band
    band_df = pd.DataFrame({
        "fpr": ci_fpr,
        "tpr_low95": ci_low_tpr,
        "tpr_high95": ci_high_tpr
    })
    band_df.to_csv(os.path.join(args.outdir, "publication_roc_ci_band.csv"), index=False)

    # save auc bootstrap dist
    if len(boot_aucs) > 0:
        pd.DataFrame({"bootstrap_auc": boot_aucs}).to_csv(
            os.path.join(args.outdir, "publication_roc_bootstrap_auc_distribution.csv"),
            index=False
        )

    out_png = os.path.join(args.outdir, "best_model_roc_publication.png")
    out_pdf = os.path.join(args.outdir, "best_model_roc_publication.pdf")
    out_svg = os.path.join(args.outdir, "best_model_roc_publication.svg")

    make_publication_roc_plot(
        y=y,
        prob=prob,
        auc=auc,
        auc_low=auc_low,
        auc_high=auc_high,
        ci_fpr=ci_fpr,
        ci_low_tpr=ci_low_tpr,
        ci_high_tpr=ci_high_tpr,
        out_png=out_png,
        out_pdf=out_pdf,
        out_svg=out_svg,
        title=args.title,
        panel_label=args.panel_label
    )

    meta = {
        "oof_input": args.oof,
        "summary_json": args.summary_json,
        "n_samples": int(len(df)),
        "n_boot": int(args.n_boot),
        "auc": float(auc),
        "auc_ci_low": float(auc_low),
        "auc_ci_high": float(auc_high),
        "outputs": {
            "png": out_png,
            "pdf": out_pdf,
            "svg": out_svg,
            "ci_band_csv": os.path.join(args.outdir, "publication_roc_ci_band.csv"),
            "bootstrap_auc_csv": os.path.join(args.outdir, "publication_roc_bootstrap_auc_distribution.csv")
        }
    }

    with open(os.path.join(args.outdir, "best_model_roc_publication_summary.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[INFO] empirical AUC = {:.4f}".format(auc))
    print("[INFO] 95% CI = ({:.4f}, {:.4f})".format(auc_low, auc_high))
    print("[INFO] output dir:", args.outdir)


if __name__ == "__main__":
    main()
