import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter, MaxNLocator
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
        .agg(
            cost=("cost", "mean"),
            price=("price", "mean"),
            count=("quantity", "sum"),
        )
        .sort_values("order_month")
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    # 棒グラフ用の第2Y軸
    ax2 = ax.twinx()
    bar_width = 0.4
    ax2.bar(
        monthly["order_month"] + bar_width / 2,
        monthly["count"],
        width=bar_width,
        color="gray",
        alpha=0.5,
        label="個数",
    )
    ax2.set_ylabel("個数")
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

    # 折れ線グラフ
    p1 = ax.plot(
        monthly["order_month"] - bar_width / 2,
        monthly["cost"],
        color="blue",
        marker="o",
        label="原価",
    )
    p2 = ax.plot(
        monthly["order_month"] - bar_width / 2,
        monthly["price"],
        color="red",
        marker="o",
        label="売価",
    )
    ax.set_xlabel("月")
    ax.set_ylabel("価格 (円)")

    ax.set_title(
        f"{order_year}年 {target.iloc[0]['product-name']} {target.iloc[0]['product-spec']} の月別原価/売価と個数"
    )
    ax.set_xticks(sorted(monthly["order_month"].unique()))

    # 凡例をまとめる
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")

    ax.grid(True, linestyle="--", alpha=0.4)

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_margin_summary(
    df: pd.DataFrame,
    jan_code: str,
    output_path: str | None = None,
    show: bool = True,
) -> None:
    jan_series = df["jan_code"].astype(str).str.strip()
    target = df[jan_series == str(jan_code)]
    if target.empty:
        raise ValueError("指定条件に該当するデータがありません。")

    monthly_by_year = (
        target.groupby(["order_year", "order_month"], as_index=False)
        .agg(cost=("cost", "mean"), price=("price", "mean"))
        .dropna(subset=["cost", "price"])
    )
    monthly_by_year = monthly_by_year[monthly_by_year["price"] != 0]
    monthly_by_year["margin_rate"] = (
        monthly_by_year["price"] - monthly_by_year["cost"]
    ) / monthly_by_year["price"]

    summary = (
        monthly_by_year.groupby("order_month")["margin_rate"]
        .agg(
            q1=lambda x: x.quantile(0.25),
            mean="mean",
            q3=lambda x: x.quantile(0.75),
        )
        .reset_index()
        .sort_values("order_month")
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        summary["order_month"],
        summary["mean"],
        color="black",
        marker="o",
    )
    y_err = [summary["mean"] - summary["q1"], summary["q3"] - summary["mean"]]
    ax.errorbar(
        summary["order_month"],
        summary["mean"],
        yerr=y_err,
        fmt="o",  # マーカーのスタイル
        color="black",
        ecolor="gray",  # エラーバーの色
        capsize=4,  # キャップの長さ
        label="平均と第1-第3四分位範囲",
    )
    ax.set_xlabel("月")
    ax.set_ylabel("粗利率")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xticks(sorted(summary["order_month"].unique()))
    ax.grid(True, linestyle="--", alpha=0.4)

    product_name = None
    product_spec = None
    if "product-name" in target.columns:
        product_name = target["product-name"].iloc[0]
    if "product-spec" in target.columns:
        product_spec = target["product-spec"].iloc[0]
    title = f"{product_name} {product_spec} 粗利率（月別）"
    ax.set_title(title)
    ax.legend()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

def plot_gross_margin(
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

    target["gross_profit"] = target["price"] - target["cost"]
    target["gross_margin"] = target["gross_profit"] / target["price"] * 100

    monthly = (
        target.groupby("order_month", as_index=False)
        .agg(gross_margin=("gross_margin", "mean"))
        .sort_values("order_month")
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(monthly["order_month"], monthly["gross_margin"], color="green", marker="o", label="粗利率")
    ax.set_xlabel("月")
    ax.set_ylabel("粗利率 (%)")
    ax.set_title(f"{order_year}年 {target.iloc[0]['product-name']} {target.iloc[0]['product-spec']} の月別粗利率")
    ax.set_xticks(sorted(monthly["order_month"].unique()))
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
