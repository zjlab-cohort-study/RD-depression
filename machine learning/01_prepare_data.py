# -*- coding: utf-8 -*-
import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="/home/wangzhaoxiang/RD_RD2DEP_ML/data/raw/earlyRD_proteins_modeling.csv"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="/home/wangzhaoxiang/RD_RD2DEP_ML/RE_DP_NEW/earlyRD_protein_ml"
    )
    args = parser.parse_args()

    step1_dir = os.path.join(args.outdir, "results", "step1_prepare")
    proc_dir = os.path.join(args.outdir, "data", "processed")
    os.makedirs(step1_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)

    print(f"[INFO] reading: {args.input}")
    df_raw = pd.read_csv(args.input)
    print(f"[INFO] raw shape: {df_raw.shape}")

    cols = list(df_raw.columns)

    eid_col = cols[0]
    rd_col = cols[1]
    y_col = cols[2]
    time_col = cols[3]
    inflam_cols = cols[4:17]   # 第5-17列
    covar_cols = cols[17:27]   # 第18-27列
    protein_cols = cols[27:]   # 第28列开始

    df = df_raw.copy()

    # 关键字段缺失删除
    key_cols = [eid_col, rd_col, y_col, time_col]
    df = df.dropna(subset=key_cols).copy()

    # 标签/时间转数值
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df = df[df[y_col].isin([0, 1])].copy()
    df = df[df[time_col].notna()].copy()

    # baseline RD 检查
    rd_unique = sorted(pd.Series(df[rd_col]).dropna().unique().tolist())

    # 炎症/协变量/蛋白尽量转数值
    for c in inflam_cols + covar_cols + protein_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 删除全缺失蛋白
    protein_missing = df[protein_cols].isna().mean().sort_values(ascending=False)
    drop_all_na = protein_missing[protein_missing >= 1.0].index.tolist()
    if len(drop_all_na) > 0:
        df = df.drop(columns=drop_all_na)
        protein_cols = [c for c in protein_cols if c not in drop_all_na]

    # 蛋白中位数填补
    if len(protein_cols) == 0:
        raise ValueError("No protein columns left after removing all-NA proteins.")
    imputer = SimpleImputer(strategy="median")
    df[protein_cols] = imputer.fit_transform(df[protein_cols])

    # 保存 clean 数据
    clean_path = os.path.join(proc_dir, "earlyRD_clean_for_screening.csv")
    df.to_csv(clean_path, index=False)

    meta = {
        "input_file": args.input,
        "raw_shape": [int(df_raw.shape[0]), int(df_raw.shape[1])],
        "clean_shape": [int(df.shape[0]), int(df.shape[1])],
        "eid_col": eid_col,
        "rd_col": rd_col,
        "label_col": y_col,
        "time_col": time_col,
        "inflammation_cols": inflam_cols,
        "covariate_cols": covar_cols,
        "protein_cols_n": len(protein_cols),
        "protein_cols_head": protein_cols[:20],
        "rd_unique_values": rd_unique,
        "dropped_all_na_proteins_n": len(drop_all_na),
        "dropped_all_na_proteins_head": drop_all_na[:20],
        "label_distribution": {str(k): int(v) for k, v in df[y_col].value_counts(dropna=False).to_dict().items()},
    }

    with open(os.path.join(step1_dir, "prepare_summary.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    pd.DataFrame({
        "column_name": [eid_col, rd_col, y_col, time_col] + inflam_cols + covar_cols + protein_cols,
        "group": (
            ["eid", "rd_baseline", "label", "followup_time"] +
            ["inflammation"] * len(inflam_cols) +
            ["covariate"] * len(covar_cols) +
            ["protein"] * len(protein_cols)
        )
    }).to_csv(os.path.join(step1_dir, "column_groups.csv"), index=False)

    print(f"[INFO] clean saved: {clean_path}")
    print(f"[INFO] proteins: {len(protein_cols)}")
    print("[INFO] finished Step1")


if __name__ == "__main__":
    main()
