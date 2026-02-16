import itertools
import os
from typing import Iterable

import numpy as np
import pandas as pd

def compute_margin_per_product(df: pd.DataFrame) -> pd.DataFrame:
    df["total_cost_row"] = df["cost"] * df["quantity"]
    df["total_price_row"] = df["price"] * df["quantity"]
    grouped = df.groupby(["order_year", "jan_code"]).agg(
        total_cost=pd.NamedAgg(column="total_cost_row", aggfunc="sum"),
        total_price=pd.NamedAgg(column="total_price_row", aggfunc="sum"),
        total_quantity=pd.NamedAgg(column="quantity", aggfunc="sum"),
    ).reset_index()

    grouped["total_margin"] = grouped["total_price"] - grouped["total_cost"]
    grouped["margin_rate"] = grouped.apply(
        lambda row: (row["total_margin"] / row["total_price"] * 100)
        if row["total_price"] > 0
        else 0,
        axis=1,
    )
    return grouped

def get_top_10_margin_products(df: pd.DataFrame) -> pd.DataFrame:
    margin_df = compute_margin_per_product(df)
    total_margin = margin_df["total_margin"].sum()
    top_10_margin_df = margin_df.sort_values("total_margin", ascending=False).head(10)
    top_10_margin = top_10_margin_df["total_margin"].sum()
    print(f"全商品の総粗利: {total_margin:,.0f}円\n上位10商品の総粗利: {top_10_margin:,.0f}円")
    return top_10_margin_df


def get_top_margin_products_for_year(
    df: pd.DataFrame,
    order_year: int,
    top_n: int = 10,
) -> pd.DataFrame:
    margin_df = compute_margin_per_product(df)
    year_df = margin_df[margin_df["order_year"] == order_year]
    return year_df.sort_values("total_margin", ascending=False).head(top_n)


def _monthly_price_series(
    df: pd.DataFrame,
    order_year: int,
    jan_code: str,
    fill_missing: str = "drop",
) -> pd.Series:
    base = df[df["jan_code"].astype(str).str.strip() == str(jan_code)]
    yearly = base[base["order_year"] == order_year]
    monthly = yearly.groupby("order_month")["price"].mean()
    if fill_missing == "drop":
        return monthly.dropna()

    # fill_missing == "impute": use other years' monthly mean for the same JAN
    all_years_monthly = base.groupby("order_month")["price"].mean()
    full = pd.Series(index=range(1, 13), dtype=float)
    for m in range(1, 13):
        if m in monthly.index and pd.notna(monthly.loc[m]):
            full.loc[m] = monthly.loc[m]
        elif m in all_years_monthly.index and pd.notna(all_years_monthly.loc[m]):
            full.loc[m] = all_years_monthly.loc[m]
        else:
            full.loc[m] = np.nan
    return full.dropna()


def cosine_similarity_for_pair(
    df: pd.DataFrame,
    order_year: int,
    jan_a: str,
    jan_b: str,
    fill_missing: str = "drop",
) -> tuple[float | None, int, list[int]]:
    if fill_missing not in {"drop", "impute"}:
        raise ValueError("fill_missing must be 'drop' or 'impute'")

    series_a = _monthly_price_series(df, order_year, jan_a, fill_missing)
    series_b = _monthly_price_series(df, order_year, jan_b, fill_missing)

    common_months = sorted(set(series_a.index).intersection(series_b.index))
    if not common_months:
        return None, 0, []

    vec_a = series_a.loc[common_months].to_numpy(dtype=float)
    vec_b = series_b.loc[common_months].to_numpy(dtype=float)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return None, len(common_months), common_months

    cosine = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    return cosine, len(common_months), common_months


def compute_pairwise_cosine_similarities(
    df: pd.DataFrame,
    order_year: int,
    jan_codes: Iterable[str],
    fill_missing: str = "drop",
) -> pd.DataFrame:
    records = []
    jan_list = [str(j) for j in jan_codes]
    for jan_a, jan_b in itertools.combinations(jan_list, 2):
        cosine, months_used, months = cosine_similarity_for_pair(
            df, order_year, jan_a, jan_b, fill_missing=fill_missing
        )
        records.append(
            {
                "order_year": order_year,
                "jan_a": jan_a,
                "jan_b": jan_b,
                "cosine_similarity": cosine,
                "months_used": months_used,
                "months": months,
                "fill_missing": fill_missing,
            }
        )
    return pd.DataFrame(records)


def _similarity_threshold(
    similarity_df: pd.DataFrame,
    method: str = "literature",
    fixed_threshold: float = 0.8,
    quantile: float = 0.75,
    std_multiplier: float = 0.5,
) -> float:
    valid = similarity_df["cosine_similarity"].dropna()
    if valid.empty:
        return fixed_threshold
    if method == "fixed":
        return fixed_threshold
    if method == "mean":
        return float(valid.mean())
    if method == "mean_plus_std":
        return float(valid.mean() + valid.std() * std_multiplier)
    if method == "quantile":
        return float(valid.quantile(quantile))
    if method == "literature":
        return 0.2
    raise ValueError(
        "method must be 'fixed', 'mean', 'mean_plus_std', 'quantile', or 'literature'"
    )


def categorize_products_by_similarity(
    df: pd.DataFrame,
    similarity_df: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    jan_codes = sorted(
        set(similarity_df["jan_a"]).union(set(similarity_df["jan_b"]))
    )
    graph = {jan: set() for jan in jan_codes}
    for _, row in similarity_df.dropna(subset=["cosine_similarity"]).iterrows():
        if row["cosine_similarity"] >= threshold:
            graph[row["jan_a"]].add(row["jan_b"])
            graph[row["jan_b"]].add(row["jan_a"])

    visited = set()
    groups = []
    for jan in jan_codes:
        if jan in visited:
            continue
        stack = [jan]
        component = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            stack.extend(graph[node])
        groups.append(component)

    name_col = "item_name"
    if name_col not in df.columns and "product-name" in df.columns:
        name_col = "product-name"

    name_map = (
        df.dropna(subset=["jan_code"]).assign(jan_code=df["jan_code"].astype(str))
        .groupby("jan_code")[name_col]
        .agg(lambda x: x.value_counts().index[0])
        .to_dict()
        if name_col in df.columns
        else {}
    )

    records = []
    for idx, group in enumerate(groups, start=1):
        for jan in sorted(group):
            records.append(
                {
                    "category": idx,
                    "jan_code": jan,
                    "product_name": name_map.get(jan, ""),
                }
            )
    return pd.DataFrame(records).sort_values(["category", "jan_code"]).reset_index(
        drop=True
    )


def save_categorized_products(
    categorized_df: pd.DataFrame,
    order_year: int,
    output_dir: str = "cosine-categorize",
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{order_year}.py")
    records = categorized_df.to_dict(orient="records")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Generated by compute.py\n")
        f.write("data = [\n")
        for record in records:
            f.write(f"    {record},\n")
        f.write("]\n")
    return output_path


def cosine_similarity_top_margin_products(
    df: pd.DataFrame,
    order_year: int,
    top_n: int = 10,
    fill_missing: str = "impute",
    threshold_method: str = "literature",
    fixed_threshold: float = 0.8,
    quantile: float = 0.75,
    std_multiplier: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    year_top = get_top_margin_products_for_year(df, order_year, top_n=top_n)
    jan_codes = year_top["jan_code"].astype(str).tolist()

    similarity_df = compute_pairwise_cosine_similarities(
        df, order_year, jan_codes, fill_missing=fill_missing
    )
    threshold = _similarity_threshold(
        similarity_df,
        method=threshold_method,
        fixed_threshold=fixed_threshold,
        quantile=quantile,
        std_multiplier=std_multiplier,
    )
    categorized_df = categorize_products_by_similarity(
        df, similarity_df, threshold=threshold
    )
    output_path = save_categorized_products(categorized_df, order_year)
    return similarity_df, categorized_df, output_path
