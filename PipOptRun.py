import argparse
import pandas as pd
from PipOpt import PipOpt

def main():
    parser = argparse.ArgumentParser(description="Run backtesting and optimization using PipOpt.")
    parser.add_argument('--signals', required=True, help="Path to the CSV file containing trading signals.")
    parser.add_argument('--prices', required=True, help="Path to the CSV file containing historical price data.")
    parser.add_argument('--mode', choices=['backtest', 'optimize_sl_tp', 'optimize_hours'], required=True,
                        help="The mode to run: 'backtest', 'optimize_sl_tp', or 'optimize_hours'.")
    parser.add_argument('--sl_usd', type=float, default=2.0, help="Initial Stop-Loss in USD.")
    parser.add_argument('--tp_usd', type=float, default=2.0, help="Initial Take-Profit in USD.")
    parser.add_argument('--pop_size', type=int, default=20, help="Population size for the optimizer.")
    parser.add_argument('--gens', type=int, default=10, help="Number of generations for the optimizer.")
    parser.add_argument('--optimizer', choices=['ga', 'pso'], default='ga', help="Optimization algorithm to use ('ga' or 'pso').")
    parser.add_argument('--metric', default='FinalBalance', help="Metric to optimize for.")
    parser.add_argument('--num_hours', type=int, default=4, help="Number of hours to optimize for (hours optimization only).")

    args = parser.parse_args()

    # Load data
    signals_df = pd.read_csv(args.signals, parse_dates=['timestamp'])
    price_df = pd.read_csv(args.prices, parse_dates=['time'])

    # Initialize the optimizer class
    optimizer = PipOpt(signals_df, price_df, stoploss_usd=args.sl_usd, takeprofit_usd=args.tp_usd)

    if args.mode == 'backtest':
        print("--- Running Backtest ---")
        trades, metrics = optimizer.run_backtest()
        print("\n=== Backtest Results ===")
        print(f"Final Balance: ${metrics['FinalBalance']:.2f}")
        print(f"Total Trades: {metrics['TotalTrades']}")
        print(f"Win Rate: {metrics['WinRate']:.2%}")
        optimizer.plot_equity_curve()

    elif args.mode == 'optimize_sl_tp':
        print("--- Running SL/TP Optimization ---")
        best_sl, best_tp, best_metric = optimizer.run_optimizer(
            optimizer_type=args.optimizer,
            optimize_metric=args.metric,
            pop_size=args.pop_size,
            gens=args.gens
        )
        print("\n--- Optimal Parameters ---")
        print(f"Best Stop-Loss: ${best_sl:.2f}")
        print(f"Best Take-Profit: ${best_tp:.2f}")
        print(f"Best {args.metric}: {best_metric:.2f}")

    elif args.mode == 'optimize_hours':
        print("--- Running Trading Hours Optimization ---")
        best_hours, best_metric = optimizer.run_hours_optimizer(
            optimizer_type=args.optimizer,
            sl_value=args.sl_usd,
            tp_value=args.tp_usd,
            optimize_metric=args.metric,
            pop_size=args.pop_size,
            gens=args.gens,
            num_hours=args.num_hours
        )
        print("\n--- Optimal Parameters ---")
        print(f"Best Trading Hours: {best_hours}")
        print(f"Best {args.metric}: {best_metric:.2f}")

if __name__ == "__main__":
    main()