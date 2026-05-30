import pandas as pd
import json
import sys
import os
import argparse
import math


def handle_nulls(df, strategy="ignore"):
    if strategy == "ignore":
        return df
    elif strategy == "fill_zero":
        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(0)
        return df
    elif strategy == "fill_mean":
        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
        return df
    else:
        return df


def compute_column_stats(col_data, null_strategy):
    col_stats = {
        "data_type": str(col_data.dtype),
        "count": int(col_data.count()),
    }

    if pd.api.types.is_numeric_dtype(col_data):
        valid_data = col_data.dropna() if null_strategy == "ignore" else col_data
        if not valid_data.empty:
            col_stats.update({
                "missing_count": int(col_data.isna().sum()),
                "missing_ratio": round(float(col_data.isna().sum() / len(col_data) if len(col_data) > 0 else 0), 4),
                "mean": round(float(valid_data.mean()), 4),
                "std": round(float(valid_data.std()), 4),
                "min": round(float(valid_data.min()), 4),
                "max": round(float(valid_data.max()), 4),
                "median": round(float(valid_data.median()), 4),
                "q1": round(float(valid_data.quantile(0.25)), 4),
                "q3": round(float(valid_data.quantile(0.75)), 4)
            })
        else:
            col_stats.update({
                "missing_count": int(col_data.isna().sum()),
                "missing_ratio": round(float(col_data.isna().sum() / len(col_data) if len(col_data) > 0 else 0), 4),
                "mean": None, "std": None, "min": None,
                "max": None, "median": None, "q1": None, "q3": None
            })
    else:
        col_stats.update({
            "missing_count": int(col_data.isna().sum()),
            "missing_ratio": round(float(col_data.isna().sum() / len(col_data) if len(col_data) > 0 else 0), 4),
            "unique_values": int(col_data.nunique()),
            "top_values": col_data.value_counts().head(5).to_dict()
        })

    return col_stats


def compute_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty or len(numeric_df.columns) < 2:
        return None

    corr = numeric_df.corr(method="pearson")
    cols = list(corr.columns)
    matrix = []
    for i, row_col in enumerate(cols):
        for j, col_col in enumerate(cols):
            val = corr.iloc[i, j]
            if math.isnan(val):
                val = None
            else:
                val = round(float(val), 4)
            matrix.append({
                "row": row_col,
                "col": col_col,
                "value": val
            })

    return {"columns": cols, "matrix": matrix}


def compute_group_stats(df, group_by, null_strategy):
    if group_by not in df.columns:
        return {"error": f"分组列 '{group_by}' 不存在"}

    if pd.api.types.is_numeric_dtype(df[group_by]) and df[group_by].nunique() > 20:
        return {"error": f"分组列 '{group_by}' 唯一值过多({df[group_by].nunique()})，建议使用分类列"}

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if group_by in numeric_cols:
        numeric_cols.remove(group_by)

    if not numeric_cols:
        return {"error": f"除分组列 '{group_by}' 外无数值列可供统计"}

    groups = df.groupby(group_by, dropna=False)
    result = {"group_by": group_by, "groups": {}}

    for group_name, group_df in groups:
        group_key = str(group_name) if not pd.isna(group_name) else "NaN"
        group_result = {"row_count": len(group_df), "columns": {}}

        for col in numeric_cols:
            col_data = group_df[col]
            valid_data = col_data.dropna() if null_strategy == "ignore" else col_data.fillna(0)
            if not valid_data.empty:
                group_result["columns"][col] = {
                    "count": int(valid_data.count()),
                    "mean": round(float(valid_data.mean()), 4),
                    "std": round(float(valid_data.std()), 4) if len(valid_data) > 1 else 0.0,
                    "min": round(float(valid_data.min()), 4),
                    "max": round(float(valid_data.max()), 4),
                    "median": round(float(valid_data.median()), 4),
                    "sum": round(float(valid_data.sum()), 4)
                }
            else:
                group_result["columns"][col] = {
                    "count": 0, "mean": None, "std": None,
                    "min": None, "max": None, "median": None, "sum": None
                }

        result["groups"][group_key] = group_result

    return result


def compute_visualization_data(df):
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        return None

    viz_data = {"histogram": {}, "boxplot": {}}

    for col in numeric_cols:
        col_data = df[col].dropna()
        if col_data.empty:
            continue

        hist, bin_edges = pd.cut(col_data, bins=10, retbins=True, duplicates="drop")
        hist_counts = hist.value_counts().sort_index()
        bins_list = []
        for interval, count in hist_counts.items():
            bins_list.append({
                "range": f"{round(float(interval.left), 2)}~{round(float(interval.right), 2)}",
                "count": int(count)
            })

        viz_data["histogram"][col] = {
            "bins": bins_list,
            "bin_edges": [round(float(b), 2) for b in bin_edges]
        }

        viz_data["boxplot"][col] = {
            "min": round(float(col_data.min()), 4),
            "q1": round(float(col_data.quantile(0.25)), 4),
            "median": round(float(col_data.median()), 4),
            "q3": round(float(col_data.quantile(0.75)), 4),
            "max": round(float(col_data.max()), 4),
            "iqr": round(float(col_data.quantile(0.75) - col_data.quantile(0.25)), 4),
            "outliers": []
        }

        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        outlier_values = col_data[(col_data < lower_fence) | (col_data > upper_fence)].tolist()
        if outlier_values:
            viz_data["boxplot"][col]["outliers"] = [round(float(v), 4) for v in outlier_values]

    return viz_data


def analyze_csv(csv_path, null_strategy="ignore", group_by=None, include_corr=True, include_viz=True):
    valid_strategies = ["ignore", "fill_zero", "fill_mean"]
    if null_strategy not in valid_strategies:
        return json.dumps({"error": f"无效的空值处理策略，可选值: {valid_strategies}"}, ensure_ascii=False, indent=2)

    if not os.path.exists(csv_path):
        return json.dumps({"error": f"文件不存在: {csv_path}"}, ensure_ascii=False, indent=2)

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return json.dumps({"error": f"读取CSV失败: {str(e)}"}, ensure_ascii=False, indent=2)

    total_rows = len(df)
    df_processed = handle_nulls(df.copy(), null_strategy)

    report = {
        "file_name": os.path.basename(csv_path),
        "null_strategy": null_strategy,
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "columns": {}
    }

    for col in df.columns:
        report["columns"][col] = compute_column_stats(df_processed[col], null_strategy)

    if include_corr:
        corr_result = compute_correlation_matrix(df_processed)
        if corr_result:
            report["correlation_matrix"] = corr_result
        else:
            report["correlation_matrix"] = "数值列不足，无法计算相关性矩阵"

    if group_by:
        group_result = compute_group_stats(df_processed, group_by, null_strategy)
        report["group_stats"] = group_result

    if include_viz:
        viz_result = compute_visualization_data(df_processed)
        if viz_result:
            report["visualization"] = viz_result
        else:
            report["visualization"] = "无数值列，无法生成可视化数据"

    return json.dumps(report, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="CSV统计分析工具")
    parser.add_argument("csv_file", nargs="?", default=None, help="CSV文件路径")
    parser.add_argument("--null-strategy", choices=["ignore", "fill_zero", "fill_mean"],
                        default="ignore", help="空值处理策略（默认: ignore）")
    parser.add_argument("--group-by", default=None, help="按指定列分组统计")
    parser.add_argument("--no-corr", action="store_true", help="不计算相关性矩阵")
    parser.add_argument("--no-viz", action="store_true", help="不生成可视化数据")

    args = parser.parse_args()

    if args.csv_file:
        csv_file = args.csv_file
    else:
        csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data.csv")

    result = analyze_csv(
        csv_file,
        null_strategy=args.null_strategy,
        group_by=args.group_by,
        include_corr=not args.no_corr,
        include_viz=not args.no_viz
    )
    print(result)


if __name__ == "__main__":
    main()
