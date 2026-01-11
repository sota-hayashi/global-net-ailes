import pandas as pd


def load_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, header=None)

    # Drop the first row, then promote the next row to the header.
    df = df.iloc[1:].reset_index(drop=True)
    header = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)
    df.columns = header

    ym = df["受注年月"].astype(str).str.strip()
    dt = pd.to_datetime(ym, format="%Y-%m", errors="coerce")
    df["受注年"] = dt.dt.year
    df["受注月"] = dt.dt.month
    return df
