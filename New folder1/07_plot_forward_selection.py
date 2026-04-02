# -*- coding: utf-8 -*-
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def determine_stop_k_auc_delta(auc_list, delta=0.005):
    """
    连续两步增量都 < delta，则在当前k处停止
    返回 1-based stop_k
    """
    stop_k = None
    for k in range(1, len(auc_list) - 1):
        d1 = auc_list[k] - auc_list[k - 1]
        d2 = auc_list[k + 1] - auc_list[k]
        if (d1 < delta) and (d2 < delta):
            stop_k = k
            break

    if stop_k is None:
        stop_k = len(auc_list) - 1

    return stop_k + 1


def build_gradient_colors(n):
    """
    生成从深蓝到浅灰蓝的渐变，接近参考图
    """
    cmap = LinearSegmentedColormap.from_list(
        "custom_blue_grad",
        ["#2d4373", "#3e6997", "#5d8aad", "#85abc1", "#b9cad8", "#d9dee7"]
    )
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def plot_from_existing(eval_df, stop_k, out_png, out_pdf, title_note=""):
    plot_df = eval_df.copy().reset_index(drop=True)
    n = plot_df.shape[0]

    x = np.arange(n)
    names = plot_df["feature"].tolist()
    imp = plot_df["importance"].values
    auc = plot_df["auc"].values
    ci_low = plot_df["auc_ci_low"].values
    ci_high = plot_df["auc_ci_high"].values

    bar_colors = build_gradient_colors(n)

    plt.figure(figsize=(16, 6), dpi=160)
    ax1 = plt.gca()

    # bar
    ax1.bar(x, imp, color=bar_colors, width=0.82, edgecolor="none")
    ax1.set_ylabel("Protein importance", fontsize=12)
    ax1.set_xlim(-0.6, n - 0.4)

    # grid
    ax1.grid(True, axis="both", linestyle=(0, (5, 3)), linewidth=0.8, alpha=0.35)
    ax1.set_axisbelow(True)

    # auc line
    ax2 = ax1.twinx()
    ax2.fill_between(x, ci_low, ci_high, color="#e7b1a7", alpha=0.28, zorder=1)
    ax2.plot(x, auc, color="#222222", linewidth=1.6, zorder=5)

    # selected part red
    sel_idx = np.arange(stop_k)
    rem_idx = np.arange(stop_k, n)

    if len(sel_idx) > 0:
        ax2.scatter(
            sel_idx, auc[sel_idx],
            s=90, color="#f03228", edgecolors="#f03228", zorder=6
        )
    if len(rem_idx) > 0:
        ax2.scatter(
            rem_idx, auc[rem_idx],
            s=28, facecolors="none", edgecolors="#222222", linewidth=1.2, zorder=6
        )

    ax2.set_ylabel("Cumulative AUC", fontsize=12, rotation=270, labelpad=28)

    # x labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=50, ha="right", fontsize=9)

    for i, tick in enumerate(ax1.get_xticklabels()):
        if i < stop_k:
            tick.set_color("#f03228")
        else:
            tick.set_color("#111111")

    # left-top panel label
    ax1.text(
        -0.07, 1.04, "a",
        transform=ax1.transAxes,
        fontsize=18, fontweight="bold", va="center"
    )

    if title_note:
        ax1.set_title(title_note, fontsize=11, pad=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        default="/home/wangzhaoxiang/RD_RD2DEP_ML/RE_DP_NEW/earlyRD_protein_ml/results/step5_forward_selection/forward_selection_full_evaluation.csv"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="/home/wangzhaoxiang/RD_RD2DEP_ML/RE_DP_NEW/earlyRD_protein_ml/results/step5_forward_selection_replot"
    )
    parser.add_argument("--delta_auc", type=float, default=0.005)
    parser.add_argument("--topn", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    df = df.iloc[:args.topn].copy().reset_index(drop=True)

    stop_k = determine_stop_k_auc_delta(df["auc"].tolist(), delta=args.delta_auc)

    out_png = os.path.join(args.outdir, f"forward_selection_replot_delta{str(args.delta_auc).replace('.', '')}.png")
    out_pdf = os.path.join(args.outdir, f"forward_selection_replot_delta{str(args.delta_auc).replace('.', '')}.pdf")

    title_note = f"Forward selection plot re-drawn from existing results (ΔAUC threshold = {args.delta_auc}, all {len(df)} proteins shown)"
    plot_from_existing(df, stop_k, out_png, out_pdf, title_note=title_note)

    summary = {
        "input_csv": args.input_csv,
        "delta_auc": args.delta_auc,
        "n_proteins_plotted": int(len(df)),
        "stop_k": int(stop_k),
        "selected_features": df["feature"].tolist()[:stop_k],
        "all_features_plotted": df["feature"].tolist(),
        "output_png": out_png,
        "output_pdf": out_pdf
    }

    with open(os.path.join(args.outdir, "replot_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[INFO] input_csv:", args.input_csv)
    print("[INFO] delta_auc:", args.delta_auc)
    print("[INFO] stop_k:", stop_k)
    print("[INFO] output_png:", out_png)
    print("[INFO] output_pdf:", out_pdf)


if __name__ == "__main__":
    main()
