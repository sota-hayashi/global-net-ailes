import argparse

from load import load_dataframe
from plot import (
    plot_cost_price,
    plot_margin_summary,
    plot_product_structure_scatter,
    plot_weighted_gross_profit_boxplot,
    plot_yearly_margin_quantity_scatter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="月別の原価/売価グラフ、または粗利率の集計グラフを作成します。"
    )
    parser.add_argument("--file", default="data-rakuten-ver00.xlsx")
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--jan", default="0000000000000")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--mode",
        choices=["price", "margin", "weighted-box", "scatter", "yearly-scatter"],
        default="margin",
    )
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataframe(args.file)
    if args.mode == "price":
        plot_cost_price(
            df,
            order_year=args.year,
            jan_code=args.jan,
            output_path=args.output,
            show=not args.no_show,
        )
    elif args.mode == "margin":
        plot_margin_summary(
            df,
            jan_code=args.jan,
            output_path=args.output,
            show=not args.no_show,
        )
    elif args.mode == "weighted-box":
        plot_weighted_gross_profit_boxplot(
            df,
            output_path=args.output,
            show=not args.no_show,
        )
    elif args.mode == "scatter":
        plot_product_structure_scatter(
            df,
            order_year=args.year,
            output_path=args.output,
            show=not args.no_show,
        )
    else:
        plot_yearly_margin_quantity_scatter(
            df,
            order_year=args.year,
            output_path=args.output,
            show=not args.no_show,
        )


if __name__ == "__main__":
    main()
