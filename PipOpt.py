import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from deap import base, creator, tools, algorithms
import random
import operator

# --- Main PipOpt Class ---
class PipOpt:
    """
    A class for backtesting trading strategies, visualizing results, and optimizing
    parameters using a Genetic Algorithm or Particle Swarm Optimization.
    
    This class handles the entire backtesting process, from loading data and
    simulating trades to calculating performance metrics and generating plots.
    """
    
    def __init__(self, signals_df, price_df, stoploss_usd=2.0, takeprofit_usd=2.0, 
                 position_size=0.05, account_balance=1000, contract_size=100,
                 long_labels=['BUY', 'LONG'], short_labels=['SELL', 'SHORT'], 
                 trading_hours=list(range(24))):
        """
        Initializes the backtester with trading parameters and data.
        
        Args:
            signals_df (pd.DataFrame): DataFrame containing 'timestamp' and 'signal_type' columns.
            price_df (pd.DataFrame): DataFrame with OHLCV data. Requires a 'time' column.
            stoploss_usd (float): Stop-loss distance in USD.
            takeprofit_usd (float): Take-profit distance in USD.
            position_size (float): The position size (e.g., in lots).
            account_balance (float): Starting account balance in USD.
            contract_size (int): The contract size of the instrument (e.g., 100 for XAUUSD).
            long_labels (list): List of strings representing long signals.
            short_labels (list): List of strings representing short signals.
            trading_hours (list): A list of integer hours (0-23) to consider for trading.
        """
        self.stoploss_usd = stoploss_usd
        self.takeprofit_usd = takeprofit_usd
        self.position_size = position_size
        self.account_balance = account_balance
        self.contract_size = contract_size
        self.long_labels = [label.upper() for label in long_labels]
        self.short_labels = [label.upper() for label in short_labels]
        self.trading_hours = trading_hours

        # Prepare and clean data
        self.signals = signals_df.copy()
        self.prices = price_df.copy()

        self.signals['timestamp'] = pd.to_datetime(self.signals['timestamp'])
        self.prices['time'] = pd.to_datetime(self.prices['time'])
        
        # Set the 'time' column as the index for faster lookups
        self.prices = self.prices.set_index('time').sort_index()
        self.trades_df = pd.DataFrame()
        self.metrics = {}
        
        # Check if DEAP creator has been initialized
        try:
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            # Individual for GA optimization is a list of two floats
            creator.create("Individual", list, fitness=creator.FitnessMax)
            # Individual for hours optimization is a list of integers
            creator.create("HoursIndividual", list, fitness=creator.FitnessMax)
            # Particle for PSO optimization
            creator.create("Particle", list, fitness=creator.FitnessMax, speed=list, 
                           smin=None, smax=None, best=None)
            creator.create("HoursParticle", list, fitness=creator.FitnessMax, speed=list, 
                           smin=None, smax=None, best=None)
        except AttributeError:
            # Already created, so we can pass
            pass

    def run_backtest(self):
        """
        Runs the backtest simulation based on the initialized parameters.
        
        This method iterates through each signal, simulates the trade's outcome
        (SL, TP, or market close), and updates the account balance.
        """
        trades = []
        current_balance = self.account_balance
        
        # Filter signals based on the selected trading hours
        filtered_signals = self.signals[self.signals['timestamp'].dt.hour.isin(self.trading_hours)]

        for _, row in filtered_signals.iterrows():
            signal_time = row['timestamp']
            raw_signal = str(row['signal_type']).upper()

            # Normalize signal type
            if raw_signal in self.long_labels:
                signal_type = 'BUY'
            elif raw_signal in self.short_labels:
                signal_type = 'SELL'
            else:
                continue

            # Ensure the signal time exists in the price data
            if signal_time not in self.prices.index:
                continue

            entry_price = self.prices.loc[signal_time, 'open']

            # Calculate SL/TP price levels
            if signal_type == 'BUY':
                sl_price = entry_price - self.stoploss_usd
                tp_price = entry_price + self.takeprofit_usd
            else:  # SELL
                sl_price = entry_price + self.stoploss_usd
                tp_price = entry_price - self.takeprofit_usd

            # Simulate the trade's duration and outcome
            future_prices = self.prices.loc[signal_time:].iloc[1:]
            exit_price = None
            outcome = 'CLOSE'
            
            # Check for SL or TP hit
            if not future_prices.empty:
                for _, p in future_prices.iterrows():
                    low = p['low']
                    high = p['high']

                    if signal_type == 'BUY':
                        if low <= sl_price:
                            exit_price = sl_price
                            outcome = 'SL'
                            break
                        elif high >= tp_price:
                            exit_price = tp_price
                            outcome = 'TP'
                            break
                    else:  # SELL
                        if high >= sl_price:
                            exit_price = sl_price
                            outcome = 'SL'
                            break
                        elif low <= tp_price:
                            exit_price = tp_price
                            outcome = 'TP'
                            break
            
            # If no SL/TP, the trade closes at the last available price
            if exit_price is None:
                exit_price = self.prices.loc[signal_time:, 'close'].iloc[-1]
                
            # Calculate P/L
            profit = 0
            if signal_type == 'BUY':
                profit = (exit_price - entry_price) * self.position_size * self.contract_size
            else: # SELL
                profit = (entry_price - exit_price) * self.position_size * self.contract_size

            current_balance += profit

            trades.append({
                'timestamp': signal_time,
                'signal': signal_type,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'outcome': outcome,
                'profit': profit,
                'account_balance': current_balance,
                'sl_price': sl_price,
                'tp_price': tp_price
            })

        self.trades_df = pd.DataFrame(trades)
        
        # Calculate and store metrics
        self._calculate_metrics()
        
        return self.trades_df, self.metrics

    def _calculate_metrics(self):
        """Calculates and stores performance metrics."""
        trades_df = self.trades_df
        if trades_df.empty:
            self.metrics = {k: 0 for k in ['WinRate', 'SharpeRatio', 'SortinoRatio', 'ProfitFactor', 'MaxDrawDown',
                                           'MaxRepeatedLoss', 'MaxRepeatedWin', 'LongTradesWinRate', 'ShortTradesWinRate',
                                           'FinalBalance', 'TotalTrades']}
            return

        returns = trades_df['profit']
        num_trades = len(returns)
        win_trades = returns[returns > 0]
        loss_trades = returns[returns <= 0]
        
        win_rate = len(win_trades) / num_trades
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        downside_std = returns[returns < 0].std()
        sortino_ratio = returns.mean() / downside_std * np.sqrt(252) if downside_std != 0 else 0
        profit_factor = win_trades.sum() / (-loss_trades.sum()) if -loss_trades.sum() != 0 else np.inf

        # Maximum Drawdown
        equity_curve = trades_df['account_balance']
        rolling_max = equity_curve.cummax()
        drawdown = rolling_max - equity_curve
        max_drawdown = drawdown.max()

        # Maximum repeated win/loss streak (corrected logic)
        streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        for p in returns:
            if p > 0:
                streak = streak + 1 if streak > 0 else 1
            elif p < 0:
                streak = streak - 1 if streak < 0 else -1
            else:
                streak = 0
            max_win_streak = max(max_win_streak, streak if streak > 0 else 0)
            max_loss_streak = min(max_loss_streak, streak if streak < 0 else 0)

        max_repeated_loss = -max_loss_streak
        max_repeated_win = max_win_streak

        # Long / Short win rates
        long_trades = trades_df[trades_df['signal'] == 'BUY']
        short_trades = trades_df[trades_df['signal'] == 'SELL']

        long_win_rate = len(long_trades[long_trades['profit'] > 0]) / len(long_trades) if len(long_trades) > 0 else 0
        short_win_rate = len(short_trades[short_trades['profit'] > 0]) / len(short_trades) if len(short_trades) > 0 else 0

        self.metrics = {
            'WinRate': win_rate,
            'SharpeRatio': sharpe_ratio,
            'SortinoRatio': sortino_ratio,
            'ProfitFactor': profit_factor,
            'MaxDrawDown': max_drawdown,
            'MaxRepeatedLoss': max_repeated_loss,
            'MaxRepeatedWin': max_repeated_win,
            'LongTradesWinRate': long_win_rate,
            'ShortTradesWinRate': short_win_rate,
            'FinalBalance': trades_df['account_balance'].iloc[-1],
            'TotalTrades': num_trades
        }

    def plot_equity_curve(self):
        """Generates a plot of the account's equity curve over time."""
        if self.trades_df.empty:
            print("No trades to plot.")
            return

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.trades_df['timestamp'],
            y=self.trades_df['account_balance'],
            mode='lines',
            name='Equity Curve',
            line=dict(color='lightgreen', width=2)
        ))
        
        # Calculate a reasonable y-axis range
        y_min = self.trades_df['account_balance'].min()
        y_max = self.trades_df['account_balance'].max()
        y_range = y_max - y_min
        y_buffer = y_range * 0.05  # 5% buffer
        y_start = y_min - y_buffer
        y_end = y_max + y_buffer

        fig.update_layout(
            title_text='Equity Curve',
            xaxis_title='Date',
            yaxis_title='Account Balance (USD)',
            template='plotly_dark',
            hovermode='x unified',
            yaxis=dict(range=[y_start, y_end])
        )
        
        fig.show()
    
    # --- Wrapper to choose and run the desired optimization algorithm for SL/TP ---
    def run_optimizer(self, optimizer_type, optimize_metric='FinalBalance', pop_size=20, 
                      gens=10, sl_bounds=(0.1, 10), tp_bounds=(0.1, 20),
                      mut_prob=0.2, cx_prob=0.5, w=0.5, c1=2.0, c2=2.0):
        """
        Wrapper to choose and run the desired optimization algorithm for SL/TP.
        """
        if optimizer_type == 'ga':
            return self.run_genetic_optimization(optimize_metric, pop_size, gens, 
                                                  mut_prob, cx_prob, sl_bounds, tp_bounds)
        elif optimizer_type == 'pso':
            return self.run_pso_optimization(optimize_metric, pop_size, gens, 
                                              sl_bounds, tp_bounds, w, c1, c2)
        else:
            raise ValueError("Invalid optimizer_type. Choose 'ga' or 'pso'.")

    # --- Wrapper to choose and run the desired optimization algorithm for Hours ---
    def run_hours_optimizer(self, optimizer_type, sl_value, tp_value, optimize_metric='FinalBalance', 
                            pop_size=20, gens=10, mut_prob=0.2, cx_prob=0.5, num_hours=4,
                            w=0.5, c1=2.0, c2=2.0):
        """
        Wrapper to choose and run the desired optimization algorithm for trading hours.
        """
        if optimizer_type == 'ga':
            return self.run_genetic_hours_optimization(sl_value, tp_value, optimize_metric, pop_size, gens, 
                                                    mut_prob, cx_prob, num_hours)
        elif optimizer_type == 'pso':
            return self.run_pso_hours_optimization(sl_value, tp_value, optimize_metric, pop_size, gens, 
                                                    num_hours, w, c1, c2)
        else:
            raise ValueError("Invalid optimizer_type. Choose 'ga' or 'pso'.")

    def run_pso_optimization(self, optimize_metric='FinalBalance', pop_size=20, gens=10, 
                             sl_bounds=(0.1, 10), tp_bounds=(0.1, 20), w=0.5, c1=2.0, c2=2.0):
        """
        Runs a Particle Swarm Optimization (PSO) to find optimal stop-loss and take-profit values.
        """
        # --- DEAP PSO setup ---
        toolbox = base.Toolbox()
        
        # PSO-specific functions
        def generate(size, smin, smax):
            # Corrected: Pass a list to the list-based creator.Particle
            part = creator.Particle([random.uniform(sl_bounds[0], sl_bounds[1]), random.uniform(tp_bounds[0], tp_bounds[1])])
            part.speed = [random.uniform(smin, smax) for _ in range(size)]
            part.smin = smin
            part.smax = smax
            return part

        def updateParticle(part, best, phi1, phi2, w):
            u1 = (random.uniform(0, phi1) for _ in range(len(part)))
            u2 = (random.uniform(0, phi2) for _ in range(len(part)))
            v_u1 = list(map(operator.mul, u1, map(operator.sub, part.best, part)))
            v_u2 = list(map(operator.mul, u2, map(operator.sub, best, part)))
            part.speed = list(map(operator.add, map(operator.mul, [w]*len(part), part.speed), map(operator.add, v_u1, v_u2)))

            for i, speed in enumerate(part.speed):
                if speed < part.smin:
                    part.speed[i] = part.smin
                elif speed > part.smax:
                    part.speed[i] = part.smax
            
            part[:] = list(map(operator.add, part, part.speed))

        toolbox.register("particle", generate, size=2, smin=-1, smax=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.particle)

        def evaluate(individual):
            original_sl = self.stoploss_usd
            original_tp = self.takeprofit_usd
            
            self.stoploss_usd = max(0.01, individual[0])
            self.takeprofit_usd = max(0.01, individual[1])
            
            _, metrics = self.run_backtest()
            
            self.stoploss_usd = original_sl
            self.takeprofit_usd = original_tp

            return metrics.get(optimize_metric, 0),

        toolbox.register("evaluate", evaluate)
        toolbox.register("update", updateParticle, phi1=c1, phi2=c2, w=w)

        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)

        print(f"Running PSO to optimize {optimize_metric} for SL and TP...\n")
        
        best = None
        for g in range(gens):
            for part in pop:
                part.fitness.values = toolbox.evaluate(part)

                if not part.best or part.best.fitness < part.fitness:
                    part.best = creator.Particle(part)
                    part.best.fitness.values = part.fitness.values
            
            for part in pop:
                if not best or best.fitness < part.fitness:
                    best = creator.Particle(part)
                    best.fitness.values = part.fitness.values

            for part in pop:
                toolbox.update(part, best)
            
            hof.update(pop)
            best_part = hof[0] if hof else None
            
            if best_part is None:
                print(f"Gen {g+1}: No valid best individual found yet.")
                continue

            fits_values = [ind.fitness.values[0] for ind in pop if ind.fitness.values]
            if fits_values:
                print(f"Gen {g+1}: Best SL={best_part[0]:.2f}, TP={best_part[1]:.2f}, {optimize_metric}={best_part.fitness.values[0]:.2f}, Max={max(fits_values):.2f}, Avg={np.mean(fits_values):.2f}, Min={min(fits_values):.2f}")
            else:
                print(f"Gen {g+1}: No valid fitness values found.")

        best_ind = hof[0]
        print("\n=== SL/TP Optimization Finished ===")
        print(f"Best SL: {best_ind[0]:.2f}, Best TP: {best_ind[1]:.2f}, Best {optimize_metric}: {best_ind.fitness.values[0]:.2f}")
        return best_ind[0], best_ind[1], best_ind.fitness.values[0]

    def run_genetic_optimization(self, optimize_metric='FinalBalance', pop_size=20, gens=10, 
                                 mut_prob=0.2, cx_prob=0.5, sl_bounds=(0.1, 10), tp_bounds=(0.1, 20)):
        """
        Runs a genetic algorithm to find optimal stop-loss and take-profit values.
        """
        # --- DEAP setup ---
        toolbox = base.Toolbox()
        toolbox.register("attr_sl", random.uniform, sl_bounds[0], sl_bounds[1])
        toolbox.register("attr_tp", random.uniform, tp_bounds[0], tp_bounds[1])
        toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_sl, toolbox.attr_tp), n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # --- Fitness function ---
        def evaluate(individual):
            original_sl = self.stoploss_usd
            original_tp = self.takeprofit_usd
            
            self.stoploss_usd = max(0.01, individual[0])
            self.takeprofit_usd = max(0.01, individual[1])
            
            _, metrics = self.run_backtest()
            
            self.stoploss_usd = original_sl
            self.takeprofit_usd = original_tp

            return metrics.get(optimize_metric, 0),

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        
        def safe_mutate(individual, mu=0, sigma=1.0, indpb=0.5):
            tools.mutGaussian(individual, mu, sigma, indpb)
            individual[0] = max(sl_bounds[0], min(individual[0], sl_bounds[1]))
            individual[1] = max(tp_bounds[0], min(individual[1], tp_bounds[1]))
            return individual,
        
        toolbox.register("mutate", safe_mutate)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)

        print(f"Running GA to optimize {optimize_metric} for SL and TP...\n")
        for gen in range(gens):
            offspring = algorithms.varAnd(pop, toolbox, cxpb=cx_prob, mutpb=mut_prob)
            fits = list(map(toolbox.evaluate, offspring))
            for ind, fit in zip(offspring, fits):
                ind.fitness.values = fit
            pop = toolbox.select(offspring, k=len(pop))
            hof.update(pop)
            fits_values = [ind.fitness.values[0] for ind in pop]
            best = hof[0]
            print(f"Gen {gen+1}: Best SL={best[0]:.2f}, TP={best[1]:.2f}, {optimize_metric}={best.fitness.values[0]:.2f}, Max={max(fits_values):.2f}, Avg={np.mean(fits_values):.2f}, Min={min(fits_values):.2f}")

        best_ind = hof[0]
        print("\n=== SL/TP Optimization Finished ===")
        print(f"Best SL: {best_ind[0]:.2f}, Best TP: {best_ind[1]:.2f}, Best {optimize_metric}: {best_ind.fitness.values[0]:.2f}")
        return best_ind[0], best_ind[1], best_ind.fitness.values[0]

    def run_genetic_hours_optimization(self, sl_value, tp_value, optimize_metric='FinalBalance', pop_size=20, gens=10, 
                                 mut_prob=0.2, cx_prob=0.5, num_hours=4):
        """
        Runs a genetic algorithm to find optimal trading hours, given fixed SL/TP.
        
        Note: The method name has been changed to avoid conflict with the SL/TP GA optimizer.
        """
        # --- DEAP setup ---
        toolbox = base.Toolbox()
        def generate_hours():
            return sorted(random.sample(range(24), num_hours))
        toolbox.register("attr_hours", generate_hours)
        toolbox.register("individual", tools.initIterate, creator.HoursIndividual, toolbox.attr_hours)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # --- Fitness function ---
        def evaluate(individual):
            original_hours = self.trading_hours
            original_sl = self.stoploss_usd
            original_tp = self.takeprofit_usd

            self.trading_hours = individual
            self.stoploss_usd = sl_value
            self.takeprofit_usd = tp_value
            
            _, metrics = self.run_backtest()
            
            self.trading_hours = original_hours
            self.stoploss_usd = original_sl
            self.takeprofit_usd = original_tp

            return metrics.get(optimize_metric, 0),

        toolbox.register("evaluate", evaluate)

        # --- Crossover and mutation for hours ---
        def cxHours(ind1, ind2):
            combined_hours = list(set(ind1) | set(ind2))
            ind1[:] = sorted(random.sample(combined_hours, k=num_hours))
            ind2[:] = sorted(random.sample(combined_hours, k=num_hours))
            return ind1, ind2

        def mutHours(individual, indpb=0.5):
            for i in range(len(individual)):
                if random.random() < indpb:
                    new_hour = random.randint(0, 23)
                    while new_hour in individual:
                        new_hour = random.randint(0, 23)
                    individual[i] = new_hour
            individual.sort()
            return individual,
        
        toolbox.register("mate", cxHours)
        toolbox.register("mutate", mutHours)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)

        print(f"Running GA to optimize trading hours for {num_hours} hours...\n")
        for gen in range(gens):
            offspring = algorithms.varAnd(pop, toolbox, cxpb=cx_prob, mutpb=mut_prob)
            fits = list(map(toolbox.evaluate, offspring))
            for ind, fit in zip(offspring, fits):
                ind.fitness.values = fit
            pop = toolbox.select(offspring, k=len(pop))
            hof.update(pop)
            fits_values = [ind.fitness.values[0] for ind in pop]
            best = hof[0]
            print(f"Gen {gen+1}: Best Hours={best}, {optimize_metric}={best.fitness.values[0]:.2f}, Max={max(fits_values):.2f}, Avg={np.mean(fits_values):.2f}, Min={min(fits_values):.2f}")

        best_ind = hof[0]
        print("\n=== Hours Optimization Finished ===")
        print(f"Best Hours: {best_ind}, Best {optimize_metric}: {best_ind.fitness.values[0]:.2f}")
        return best_ind, best_ind.fitness.values[0]
    
    def run_pso_hours_optimization(self, sl_value, tp_value, optimize_metric='FinalBalance', 
                                   pop_size=20, gens=10, num_hours=4, w=0.5, c1=2.0, c2=2.0):
        """
        Runs a Discrete Particle Swarm Optimization (DPSO) to find optimal trading hours.
        """
        toolbox = base.Toolbox()
        
        def generate_hours_particle(size, smin, smax):
            part = creator.HoursParticle(sorted(random.sample(range(24), num_hours)))
            part.speed = [random.uniform(smin, smax) for _ in range(size)]
            part.smin = smin
            part.smax = smax
            return part

        def updateHoursParticle(part, best, phi1, phi2, w):
            u1 = (random.uniform(0, phi1) for _ in range(len(part)))
            u2 = (random.uniform(0, phi2) for _ in range(len(part)))
            v_u1 = list(map(operator.mul, u1, map(operator.sub, part.best, part)))
            v_u2 = list(map(operator.mul, u2, map(operator.sub, best, part)))
            part.speed = list(map(operator.add, map(operator.mul, [w]*len(part), part.speed), map(operator.add, v_u1, v_u2)))

            # Clamp speed to be within bounds
            for i, speed in enumerate(part.speed):
                if speed < part.smin:
                    part.speed[i] = part.smin
                elif speed > part.smax:
                    part.speed[i] = part.smax
            
            # Update position (discrete hours)
            new_hours_raw = list(map(operator.add, part, part.speed))
            new_hours_rounded = sorted([int(round(h)) % 24 for h in new_hours_raw])
            
            # Ensure unique hours
            part[:] = sorted(list(set(new_hours_rounded)))
            # If the number of unique hours is less than num_hours, add new random hours
            while len(part) < num_hours:
                new_hour = random.randint(0, 23)
                if new_hour not in part:
                    part.append(new_hour)
                    part.sort()
            
            part[:] = sorted(random.sample(part, k=num_hours))
        
        toolbox.register("particle", generate_hours_particle, size=num_hours, smin=-1, smax=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.particle)

        def evaluate(individual):
            original_hours = self.trading_hours
            self.trading_hours = individual
            self.stoploss_usd = sl_value
            self.takeprofit_usd = tp_value
            
            _, metrics = self.run_backtest()
            
            self.trading_hours = original_hours
            return metrics.get(optimize_metric, 0),

        toolbox.register("evaluate", evaluate)
        toolbox.register("update", updateHoursParticle, phi1=c1, phi2=c2, w=w)

        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)

        print(f"Running PSO to optimize trading hours for {num_hours} hours...\n")
        
        best = None
        for g in range(gens):
            for part in pop:
                part.fitness.values = toolbox.evaluate(part)

                if not part.best or part.best.fitness < part.fitness:
                    part.best = creator.HoursParticle(part)
                    part.best.fitness.values = part.fitness.values
            
            for part in pop:
                if not best or best.fitness < part.fitness:
                    best = creator.HoursParticle(part)
                    best.fitness.values = part.fitness.values

            for part in pop:
                toolbox.update(part, best)
            
            hof.update(pop)
            best_part = hof[0] if hof else None

            if best_part is None:
                print(f"Gen {g+1}: No valid best individual found yet.")
                continue

            fits_values = [ind.fitness.values[0] for ind in pop if ind.fitness.values]
            if fits_values:
                print(f"Gen {g+1}: Best Hours={best_part}, {optimize_metric}={best_part.fitness.values[0]:.2f}, Max={max(fits_values):.2f}, Avg={np.mean(fits_values):.2f}, Min={min(fits_values):.2f}")
            else:
                print(f"Gen {g+1}: No valid fitness values found.")

        best_ind = hof[0]
        print("\n=== Hours Optimization Finished ===")
        print(f"Best Hours: {best_ind}, Best {optimize_metric}: {best_ind.fitness.values[0]:.2f}")
        return best_ind, best_ind.fitness.values[0]

