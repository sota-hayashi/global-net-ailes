import pandas as pd


def load_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, header=None)

    # Drop the first row, then promote the next row to the header.
    df = df.iloc[1:].reset_index(drop=True)
    header = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)
    df.columns = header

    column_map = {
        "受注年月": "order_ym",
        "得意先コード": "customer_code",
        "得意先名": "customer_name",
        "メーカー名": "maker_name",
        "JANコード": "jan_code",
        "商品名": "item_name",
        "原価": "cost",
        "売価": "price",
        "数量": "quantity",
    }
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

    for col in ("cost", "price", "quantity"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    ym = df["order_ym"].astype(str).str.strip()
    dt = pd.to_datetime(ym, format="%Y-%m", errors="coerce")
    df["order_year"] = dt.dt.year
    df["order_month"] = dt.dt.month

    # 商品名を名称と規格に分割
    # e.g. "エビオス\u3000\u30002000錠" -> "エビオス", "2000錠"
    # e.g. "商品名 Ｎｏ．１" -> "商品名", "Ｎｏ．１"
    if "item_name" in df.columns:
        product_name_parts = df["item_name"].str.extract(
            r"^(.*?)\s+((\d|Ｎｏ).*)$", expand=True
        )
        df["product-name"] = product_name_parts[0].str.strip()
        df["product-spec"] = product_name_parts[1].str.strip()

        # 分割できなかった場合は、元の商品名を名称に入れる
        unmatched_mask = df["product-name"].isnull()
        df.loc[unmatched_mask, "product-name"] = df.loc[
            unmatched_mask, "item_name"
        ].str.strip()
    return df
