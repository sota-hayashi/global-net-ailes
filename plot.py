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
    jan_series = df["jan_code"].astype(str).str.strip()
    target = df[(df["order_year"] == order_year) & (jan_series == str(jan_code))]
    if target.empty:
        raise ValueError("指定条件に該当するデータがありません。")

    monthly = (
        target.groupby("order_month", as_index=False)
        .agg(cost=("cost", "mean"), price=("price", "mean"))
        .sort_values("order_month")
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(monthly["order_month"], monthly["cost"], color="blue", marker="o", label="原価")
    ax.plot(monthly["order_month"], monthly["price"], color="red", marker="o", label="売価")
    ax.set_xlabel("月")
    ax.set_ylabel("価格")
    ax.set_title(f"{order_year}年 JANコード {jan_code}")
    ax.set_xticks(sorted(monthly["order_month"].unique()))
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
