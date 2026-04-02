# -*- coding: utf-8 -*-
import os
import re
import glob
import json
import argparse
import pandas as pd


def parse_feature_set_name(fp):
    """
    从文件名中解析 coarse/refine 信息
    例如:
    feature_set__coarse-mutual_info__refine-lasso__top20.csv
    """
    base = os.path.basename(fp).replace(".csv", "")
    m = re.match(r"feature_set__coarse-(.*?)__refine-(.*?)__top(\d+)", base)
    if m is None:
        return {
            "feature_set": base,
            "coarse_method": "unknown",
            "refine_method": "unknown",
            "topk": None
        }
    return {
        "feature_set": base,
        "coarse_method": m.group(1),
        "refine_method": m.group(2),
        "topk": int(m.group(3))
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--featdir",
        type=str,
        default="/home/wangzhaoxiang/RD_RD2DEP_ML/RE_DP_NEW/earlyRD_protein_ml/data/feature_sets"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="/home/wangzhaoxiang/RD_RD2DEP_ML/RE_DP_NEW/earlyRD_protein_ml/results/step3_refine"
    )
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    pattern = os.path.join(args.featdir, f"feature_set__coarse-*__refine-*__top{args.topk}.csv")
    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        raise FileNotFoundError(f"No feature set files found: {pattern}")

    all_records = []
    feature_set_summary = []

    for fp in files:
        info = parse_feature_set_name(fp)
        df = pd.read_csv(fp)

        if "feature" not in df.columns:
            raise ValueError(f"'feature' column not found in: {fp}")

        feats = df["feature"].dropna().astype(str).tolist()

        feature_set_summary.append({
            "feature_set": info["feature_set"],
            "coarse_method": info["coarse_method"],
            "refine_method": info["refine_method"],
            "n_features": len(feats),
            "file": fp
        })

        for feat in feats:
            all_records.append({
                "protein": feat,
                "feature_set": info["feature_set"],
                "coarse_method": info["coarse_method"],
                "refine_method": info["refine_method"]
            })

    long_df = pd.DataFrame(all_records)
    long_df.to_csv(os.path.join(args.outdir, "protein_selection_long_table.csv"), index=False)

    pd.DataFrame(feature_set_summary).to_csv(
        os.path.join(args.outdir, "feature_set_file_summary.csv"),
        index=False
    )

    stat_rows = []
    for protein, sub in long_df.groupby("protein"):
        total_count = int(sub.shape[0])
        coarse_methods = sorted(sub["coarse_method"].dropna().unique().tolist())
        refine_methods = sorted(sub["refine_method"].dropna().unique().tolist())
        feature_sets = sorted(sub["feature_set"].dropna().unique().tolist())

        stat_rows.append({
            "protein": protein,
            "selected_total_count": total_count,
            "selected_in_n_coarse_methods": len(coarse_methods),
            "selected_in_n_refine_methods": len(refine_methods),
            "coarse_methods": ";".join(coarse_methods),
            "refine_methods": ";".join(refine_methods),
            "feature_sets": " | ".join(feature_sets)
        })

    stat_df = pd.DataFrame(stat_rows).sort_values(
        ["selected_total_count", "selected_in_n_coarse_methods", "selected_in_n_refine_methods", "protein"],
        ascending=[False, False, False, True]
    )

    stat_df.to_csv(os.path.join(args.outdir, "protein_selection_frequency_summary.csv"), index=False)

    # 再做一个更适合看稳定性的透视表
    coarse_pivot = (
        long_df.assign(value=1)
        .pivot_table(index="protein", columns="coarse_method", values="value", aggfunc="sum", fill_value=0)
        .reset_index()
    )
    coarse_pivot.to_csv(os.path.join(args.outdir, "protein_selection_by_coarse_method.csv"), index=False)

    refine_pivot = (
        long_df.assign(value=1)
        .pivot_table(index="protein", columns="refine_method", values="value", aggfunc="sum", fill_value=0)
        .reset_index()
    )
    refine_pivot.to_csv(os.path.join(args.outdir, "protein_selection_by_refine_method.csv"), index=False)

    coarse_refine_pivot = (
        long_df.assign(pair=long_df["coarse_method"] + "__" + long_df["refine_method"], value=1)
        .pivot_table(index="protein", columns="pair", values="value", aggfunc="sum", fill_value=0)
        .reset_index()
    )
    coarse_refine_pivot.to_csv(os.path.join(args.outdir, "protein_selection_by_feature_set_matrix.csv"), index=False)

    summary = {
        "n_feature_set_files": len(files),
        "topk_each_set": args.topk,
        "total_selected_records": int(long_df.shape[0]),
        "n_unique_proteins_selected": int(long_df["protein"].nunique()),
        "output_files": {
            "long_table": os.path.join(args.outdir, "protein_selection_long_table.csv"),
            "summary_table": os.path.join(args.outdir, "protein_selection_frequency_summary.csv"),
            "coarse_pivot": os.path.join(args.outdir, "protein_selection_by_coarse_method.csv"),
            "refine_pivot": os.path.join(args.outdir, "protein_selection_by_refine_method.csv"),
            "feature_set_matrix": os.path.join(args.outdir, "protein_selection_by_feature_set_matrix.csv")
        }
    }

    with open(os.path.join(args.outdir, "protein_selection_count_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[INFO] n_feature_set_files =", len(files))
    print("[INFO] total_selected_records =", long_df.shape[0])
    print("[INFO] n_unique_proteins_selected =", long_df['protein'].nunique())
    print("[INFO] saved:", os.path.join(args.outdir, "protein_selection_frequency_summary.csv"))


if __name__ == "__main__":
    main()
