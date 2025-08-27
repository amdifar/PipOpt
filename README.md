## 📊 PipOpt – Trading Strategy Backtesting & Optimization Framework
PipOpt is a **Python-based framework** for backtesting and optimizing trading strategies using signal data and historical market prices.
It supports backtesting, stop-loss/take-profit optimization, and trading hours optimization with evolutionary algorithms like Genetic Algorithm (GA) and Particle Swarm Optimization (PSO).
<p align="center">
  <img src="PipOptLogo.png" width="200" alt="PipOpt Logo">
</p>

### ⚡ Features

  * **Fast Backtests:** Quickly run backtests using your signals and price data.
  * **Parameter Optimization:** Find the best **stop-loss** and **take-profit** levels using powerful GA or PSO algorithms.
  * **Optimal Trading Hours:** Discover the most profitable hours of the day to trade.
  * **Comprehensive Metrics:** Evaluate strategy performance with a built-in suite of metrics.
  * **Flexible Usage:** Work from your command line or use an interactive Jupyter Notebook.

-----

### 📂 Datasets Required

To use PipOpt, you need two CSV datasets with specific column names and formats:

1.  **Signals File** (`Signals/Signals.csv`)
    This file contains your trade signals.

    ```csv
    timestamp,signal_type
    2025-01-01 12:00:00,BUY
    2025-01-01 12:10:00,SELL
    ```

2.  **Price Data** (`Data/xauusd_1m.csv`)
    This file holds the historical 1-minute OHLC data for the trading symbol.

    ```csv
    time,open,high,low,close
    2025-01-01 12:00:00,2050.2,2051.1,2049.8,2050.5
    ```

> ⚠️ **Important:** Datasets must match this exact format. Otherwise, PipOpt won't parse them correctly.

-----

### 🏃 Usage – Command Line

You can run PipOpt using the `PipOptRun.py` entry script.

#### 🔹 Simple Backtest

Run a backtest with specified stop-loss and take-profit values:

```bash
python PipOptRun.py --mode backtest --signals Signals/Signals.csv --prices Data/xauusd_1m.csv --sl_usd 2.5 --tp_usd 5.0
```

  * `--mode backtest`: Tells the script to run a backtest.
  * `--sl_usd 2.5`: Sets the stop-loss to $2.5.
  * `--tp_usd 5.0`: Sets the take-profit to $5.0.

-----

#### 🔹 Optimize Stop-Loss & Take-Profit (GA / PSO)
Find the optimal SL/TP values to maximize a chosen metric, like FinalBalance:

````bash
python PipOpt.py --mode optimize_sl_tp --signals Signals/Signals.csv --prices Data/xauusd_1m.csv --optimizer ga --metric FinalBalance --pop_size 50 --gens 20
````
* `optimizer ga`: Selects the Genetic Algorithm. pso is also an option.

* `metric FinalBalance`: The optimization's goal is to maximize the final account balance.

You can also use other metrics for optimization. The available metrics are WinRate, SharpeRatio, SortinoRatio, ProfitFactor, MaxDrawDown, MaxRepeatedLoss, MaxRepeatedWin, LongTradesWinRate, ShortTradesWinRate, and TotalTrades. For example, to maximize the WinRate, you would use:

```Bash
python PipOpt.py --mode optimize_sl_tp --signals Signals/Signals.csv --prices Data/xauusd_1m.csv --optimizer ga --metric WinRate --pop_size 50 --gens 20
````
-----
#### 🔹 Optimize Trading Hours (GA / PSO)

Discover the best trading hours for a given strategy:

```bash
python PipOpt.py --mode optimize_hours --signals Signals/Signals.csv --prices Data/xauusd_1m.csv --optimizer pso --sl_usd 3.0 --tp_usd 4.5 --metric ProfitFactor --num_hours 5
```

  * `--num_hours 5`: The script will find the top 5 most profitable hours.

-----

### 🧬 Evolutionary Algorithms

PipOpt uses advanced algorithms to find optimal solutions more effectively than brute-force methods.

| Algorithm | Analogy | Description |
| :--- | :--- | :--- |
| **Genetic Algorithm (GA)** | Natural Selection  | Mimics evolution by creating a population of solutions, which "breed" and "mutate" over generations to produce increasingly better solutions. |
| **Particle Swarm Optimization (PSO)** | Bird Flocking  | Inspired by social behavior, particles (solutions) "fly" through the search space, adjusting their path based on their own best-found position and the best-found position of the entire swarm. |

-----

### 📓 Jupyter Notebook Version

For an interactive, hands-on experience, you can use the included Jupyter Notebook version. You can load data, run optimizations, and visualize results directly in your browser.

```bash
jupyter notebook PipOpt_Notebook.ipynb
```

-----

### 📈 Performance Metrics

PipOpt provides a comprehensive suite of metrics to help you analyze your strategy's performance.

| Metric | Meaning |
| :--- | :--- |
| **Win Rate** | The percentage of trades that are profitable. |
| **Profit Factor** | The ratio of total profits to total losses; a value \> 1 indicates a profitable strategy. |
| **Sharpe Ratio** | Measures risk-adjusted return, showing how well a return compensates for risk. |
| **Sortino Ratio** | Measures the risk-adjusted return by using only the downside deviation of returns. |
| **Max Drawdown** | The largest drop in the account balance from a peak to a subsequent trough. |
| **Final Balance** | The total capital remaining after all trades. |
| **Max Consecutive Wins/Losses** | The longest consecutive number of winning or losing trades. |
| **Long/Short Win Rate** | Shows if the strategy is biased towards long or short positions. |

-----

### 🚀 Getting Started

To begin using PipOpt, you first need to prepare your data.

1.  **Prepare Your Data**: Create two CSV files: one for your strategy's entry points and one for the symbol's 1-minute historical data. There are sample datasets included in the repository for a strategy's entry points and a matching `xauusd_1m` timeframe.

2.  **Clone the Repository**:

    ```bash
    git clone https://github.com/yourusername/PipOpt.git
    cd PipOpt
    ```

3.  **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run Backtesting or Optimization**: Now you can run the commands provided in the "Usage – Command Line" section above. For example, to run a simple backtest:

    ```bash
    python PipOpt.py --mode backtest --signals Signals/Signals.csv --prices Data/xauusd_1m.csv --sl_usd 2.5 --tp_usd 5.0
    ```

    ```bash
    python PipOpt.py --mode optimize_sl_tp --signals Signals/Signals.csv --prices Data/xauusd_1m.csv --optimizer ga --metric FinalBalance --pop_size 50 --gens 20
    ```


### 📜 License
This project is licensed under the MIT License.




