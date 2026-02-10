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