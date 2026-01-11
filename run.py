import argparse

from load import load_dataframe
from plot import plot_cost_price


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="月別の原価/売価グラフを作成します。")
    parser.add_argument("--file", default="data-rakuten-ver00.xlsx")
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--jan", default="0000000000000")
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataframe(args.file)
    plot_cost_price(
        df,
        order_year=args.year,
        jan_code=args.jan,
        output_path=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
