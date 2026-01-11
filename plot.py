import matplotlib.pyplot as plt
import pandas as pd
import japanize_matplotlib


def plot_cost_price(
    df: pd.DataFrame,
    order_year: int,
    jan_code: str,
    output_path: str | None = None,
    show: bool = True,
) -> None:
    jan_series = df["JANコード"].astype(str).str.strip()
    target = df[(df["受注年"] == order_year) & (jan_series == str(jan_code))]
    if target.empty:
        raise ValueError("指定条件に該当するデータがありません。")

    monthly = (
        target.groupby("受注月", as_index=False)[["原価", "売価"]]
        .mean()
        .sort_values("受注月")
    )

    fig, ax = plt.subplots()
    ax.plot(monthly["受注月"], monthly["原価"], color="blue", marker="o", label="原価")
    ax.plot(monthly["受注月"], monthly["売価"], color="red", marker="o", label="売価")
    ax.set_xlabel("月")
    ax.set_ylabel("価格")
    ax.set_title(f"{order_year}年 JANコード {jan_code}")
    ax.set_xticks(sorted(monthly["受注月"].unique()))
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
