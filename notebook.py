#NOMURA QUANT CHALLENGE 2025

#The format for the weights dataframe for the backtester is attached with the question.
#Complete the below codes wherever applicable

import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm


def backtester_without_TC(weights_df):
    """Backtest strategy without transaction costs."""
    # Load and prepare data
    train_data = pd.read_csv('train_data.csv')
    crossval_data = pd.read_csv('crossval_data.csv')
    data = pd.concat([train_data, crossval_data]).sort_values(['Symbol', 'Date'])
    
    # Ensure weights are properly aligned
    weights_df = weights_df.fillna(0)
    weights_df.index = weights_df.index.astype(int)

    # Define date range
    start_date = 3500
    end_date = 3999

    # Calculate returns for all symbols at once
    returns_df = pd.DataFrame()
    for symbol in range(20):
        symbol_data = data[data['Symbol'] == symbol].sort_values('Date')
        symbol_data = symbol_data.set_index('Date')
        # Calculate returns using next day's price
        symbol_returns = symbol_data['Close'].pct_change().shift(-1)
        returns_df[symbol] = symbol_returns
    
    # Align weights and returns
    weights_df = weights_df.loc[start_date:end_date]
    returns_df = returns_df.loc[start_date:end_date]
    
    # Calculate portfolio returns
    portfolio_returns = (weights_df * returns_df).sum(axis=1)
    
    # Calculate performance metrics with proper daily compounding
    initial_notional = 1
    cumulative_returns = (1 + portfolio_returns).cumprod()
    final_notional = initial_notional * cumulative_returns.iloc[-1]
    
    # Calculate annualized Sharpe ratio
    daily_returns = portfolio_returns
    annualized_return = daily_returns.mean() * 252
    annualized_vol = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 1e-8 else 0
    
    net_return = ((final_notional - initial_notional) / initial_notional) * 100

    return [net_return, sharpe_ratio]



def task1_Strategy1():
    """Strategy 1: Average Weekly Returns
    - Calculates weekly returns for each stock
    - Ranks stocks based on their average returns using up to 50 weeks of data
    - Takes long positions in bottom 6 stocks and short positions in top 6 stocks
    """
    # Load and prepare data
    train_data = pd.read_csv('train_data.csv')
    crossval_data = pd.read_csv('crossval_data.csv')
    all_data = pd.concat([train_data, crossval_data]).sort_values(['Symbol', 'Date'])
    
    # Calculate week numbers (1-based dates)
    all_data['Week'] = (all_data['Date'] - 1) // 5
    
    # Precompute weekly returns for all stocks
    weekly_returns_cache = {}
    for symbol in range(20):
        stock_data = all_data[all_data['Symbol'] == symbol].copy()
        if stock_data.empty:
            weekly_returns_cache[symbol] = pd.Series()
            continue
        
        # Group by week and get last close price
        weekly_closes = stock_data.groupby('Week')['Close'].last()
        
        # Ensure complete weeks (5 days)
        complete_weeks = weekly_closes.index[weekly_closes.index.isin(stock_data.groupby('Week').size()[stock_data.groupby('Week').size() == 5].index)]
        weekly_closes = weekly_closes[complete_weeks]
        
        # Calculate weekly returns
        weekly_returns = weekly_closes.pct_change()
        weekly_returns_cache[symbol] = weekly_returns
    
    # Initialize output with numpy zeros
    output_df = pd.DataFrame(0.0, index=range(3500, 4000), columns=range(20), dtype=float)
    
    for current_day in range(3500, 4000):
        current_week = (current_day - 1) // 5
        mean_returns = []
        
        for symbol in range(20):
            returns = weekly_returns_cache[symbol]
            
            # Get valid weeks before current week and sort once
            valid_weeks = sorted(returns.index[returns.index < current_week])
            
            if len(valid_weeks) >= 50:
                # Get last 50 weeks of returns
                latest_50_weeks = valid_weeks[-50:]
                mean_return = returns[latest_50_weeks].mean()
            else:
                # Use all available weeks if less than 50
                if len(valid_weeks) > 0:
                    mean_return = returns[valid_weeks].mean()
                else:
                    mean_return = 0
            
            mean_returns.append((symbol, mean_return))
        
        # Sort by mean return (descending)
        mean_returns.sort(key=lambda x: x[1], reverse=True)
        
        # Assign weights
        weights = [0] * 20
        
        # Top 6 stocks (short)
        for i in range(6):
            weights[mean_returns[i][0]] = -1/6
        
        # Bottom 6 stocks (long)
        for i in range(14, 20):
            weights[mean_returns[i][0]] = 1/6
        
        output_df.loc[current_day] = weights
    
    return output_df


def task1_Strategy2():
    """Strategy 2: SMA vs. LMA
    - Calculates 30-day LMA and 5-day SMA
    - Takes long positions in stocks where SMA > LMA (bottom 5)
    - Takes short positions in stocks where SMA < LMA (top 5)
    """
    # Load and prepare data
    train_data = pd.read_csv('train_data.csv')
    crossval_data = pd.read_csv('crossval_data.csv')
    all_data = pd.concat([train_data, crossval_data]).sort_values(['Symbol', 'Date'])
    
    # Precompute moving averages for all stocks
    ma_cache = {}
    for symbol in range(20):
        stock_data = all_data[all_data['Symbol'] == symbol].copy()
        if stock_data.empty:
            ma_cache[symbol] = {'lma': pd.Series(), 'sma': pd.Series()}
            continue
        
        # Calculate rolling means with proper min_periods and shift
        stock_data['LMA'] = (
            stock_data['Close']
            .rolling(window=30, min_periods=30)
            .mean()
            .shift(1)  # Use data up to D-1
        )
        stock_data['SMA'] = (
            stock_data['Close']
            .rolling(window=5, min_periods=5)
            .mean()
            .shift(1)  # Use data up to D-1
        )
        
        # Store in cache
        ma_cache[symbol] = {
            'lma': stock_data.set_index('Date')['LMA'],
            'sma': stock_data.set_index('Date')['SMA']
        }
    
    # Initialize output with numpy zeros
    output_df = pd.DataFrame(0.0, index=range(3500, 4000), columns=range(20), dtype=float)
    
    for current_day in range(3500, 4000):
        relative_positions = []
        
        for symbol in range(20):
            lma = ma_cache[symbol]['lma']
            sma = ma_cache[symbol]['sma']
            
            if current_day in lma.index and current_day in sma.index:
                lma_value = lma[current_day]
                sma_value = sma[current_day]
                
                if pd.isna(lma_value) or pd.isna(sma_value):
                    relative_position = 0
                elif lma_value != 0:  # Avoid division by zero
                    relative_position = (sma_value - lma_value) / lma_value
                else:
                    relative_position = 0
            else:
                relative_position = 0
            
            relative_positions.append((symbol, relative_position))
        
        # Sort by relative position (descending)
        relative_positions.sort(key=lambda x: x[1], reverse=True)
        
        # Assign weights
        weights = [0] * 20
        
        # Top 5 stocks (short)
        for i in range(5):
            weights[relative_positions[i][0]] = -1/5
        
        # Bottom 5 stocks (long)
        for i in range(15, 20):
            weights[relative_positions[i][0]] = 1/5
        
        output_df.loc[current_day] = weights
    
    return output_df


def task1_Strategy3():
    """Strategy 3: Rate of Change (ROC)
    - Calculates 7-day rate of change
    - Takes long positions in stocks with lowest ROC (bottom 4)
    - Takes short positions in stocks with highest ROC (top 4)
    """
    # Load and prepare data
    train_data = pd.read_csv('train_data.csv')
    crossval_data = pd.read_csv('crossval_data.csv')
    all_data = pd.concat([train_data, crossval_data]).sort_values(['Symbol', 'Date'])
    
    # Precompute ROC for all stocks
    roc_cache = {}
    for symbol in range(20):
        stock_data = all_data[all_data['Symbol'] == symbol].copy()
        if stock_data.empty:
            roc_cache[symbol] = pd.Series()
            continue
        
        # Calculate ROC using data up to D-1
        stock_data['ROC'] = 100 * (
            (stock_data['Close'].shift(1) - stock_data['Close'].shift(8))
            / stock_data['Close'].shift(8)
        )
        roc_cache[symbol] = stock_data.set_index('Date')['ROC']
    
    # Initialize output with numpy zeros
    output_df = pd.DataFrame(0.0, index=range(3500, 4000), columns=range(20), dtype=float)
    
    for current_day in range(3500, 4000):
        roc_values = []
        
        for symbol in range(20):
            roc = roc_cache[symbol]
            
            if current_day in roc.index:
                roc_value = roc[current_day]
                if pd.isna(roc_value):
                    roc_value = 0
            else:
                roc_value = 0
            
            roc_values.append((symbol, roc_value))
        
        # Sort by ROC (descending)
        roc_values.sort(key=lambda x: x[1], reverse=True)
        
        # Assign weights
        weights = [0] * 20
        
        # Top 4 stocks (short)
        for i in range(4):
            weights[roc_values[i][0]] = -1/4
        
        # Bottom 4 stocks (long)
        for i in range(16, 20):
            weights[roc_values[i][0]] = 1/4
        
        output_df.loc[current_day] = weights
    
    return output_df


# Helper functions for data preprocessing and common operations
def load_and_prepare_data():
    """Load and prepare data for all strategies."""
    train_data = pd.read_csv('train_data.csv')
    crossval_data = pd.read_csv('crossval_data.csv')
    return pd.concat([train_data, crossval_data]).sort_values(['Symbol', 'Date'])

def get_stock_data_cache(all_data):
    """Create a cache of stock data for each symbol."""
    stock_cache = {}
    for symbol in range(20):
        stock_data = all_data[all_data['Symbol'] == symbol].copy()
        if not stock_data.empty:
            stock_data = stock_data.sort_values('Date')
            stock_cache[symbol] = stock_data.set_index('Date')
    return stock_cache

def calculate_rolling_metrics(data, window, min_periods=None):
    """Calculate rolling mean and standard deviation.
    
    Args:
        data: Price series
        window: Rolling window size
        min_periods: Minimum number of observations required (defaults to window size)
        
    Returns:
        Dictionary containing rolling metrics
    """
    if min_periods is None:
        min_periods = window
    
    return {
        'mean': data.rolling(window=window, min_periods=min_periods).mean(),
        'std': data.rolling(window=window, min_periods=min_periods).std()
    }

def task1_Strategy4():
    """Strategy 4: Support/Resistance
    - Calculates 21-day rolling mean and standard deviation
    - Identifies support and resistance levels using mean ± 3*std
    - Takes long positions in stocks near support (top 4)
    - Takes short positions in stocks near resistance (top 4)
    """
    # Load and prepare data
    all_data = load_and_prepare_data()
    stock_cache = get_stock_data_cache(all_data)
    
    # Precompute support and resistance metrics for all stocks
    metrics_cache = {}
    for symbol in range(20):
        if symbol in stock_cache:
            stock_data = stock_cache[symbol]
            # Use proper min_periods for 21-day window
            metrics = calculate_rolling_metrics(stock_data['Close'], window=21, min_periods=21)
            
            # Calculate support and resistance levels
            resistance = metrics['mean'] + 3 * metrics['std']
            support = metrics['mean'] - 3 * metrics['std']
            
            # Calculate proximity metrics
            latest_price = stock_data['Close']
            proximity_to_resistance = (latest_price - resistance) / resistance
            proximity_to_support = (latest_price - support) / support
            
            metrics_cache[symbol] = {
                'proximity_to_support': proximity_to_support,
                'proximity_to_resistance': proximity_to_resistance
            }
    
    # Prepare output
    output_df = pd.DataFrame(0.0, index=range(3500, 4000), columns=range(20), dtype=float)
    
    for current_day in range(3500, 4000):
        stock_metrics = []
        
        for symbol in range(20):
            if symbol in metrics_cache:
                support_prox = metrics_cache[symbol]['proximity_to_support']
                resistance_prox = metrics_cache[symbol]['proximity_to_resistance']
                
                if current_day in support_prox.index:
                    support_value = support_prox[current_day]
                    resistance_value = resistance_prox[current_day]
                    
                    if pd.isna(support_value):
                        support_value = 0
                    if pd.isna(resistance_value):
                        resistance_value = 0
                else:
                    support_value = 0
                    resistance_value = 0
            else:
                support_value = 0
                resistance_value = 0
            
            stock_metrics.append((symbol, support_value, resistance_value))
        
        # First rank by proximity to Support (increasing order)
        stock_metrics.sort(key=lambda x: x[1])
        
        # Assign positive weights to top 4 stocks by proximity to Support
        weights = [0] * 20
        for i in range(4):
            weights[stock_metrics[i][0]] = 1/4
        
        # Then rank remaining stocks by proximity to Resistance (decreasing order)
        remaining_stocks = stock_metrics[4:]
        remaining_stocks.sort(key=lambda x: x[2], reverse=True)
        
        # Assign negative weights to top 4 stocks by proximity to Resistance
        for i in range(4):
            weights[remaining_stocks[i][0]] = -1/4
        
        output_df.loc[current_day] = weights
    
    return output_df


def task1_Strategy5():
    """Strategy 5: %K Oscillator
    - Calculates 14-day %K oscillator
    - Takes long positions in stocks with lowest %K (bottom 3)
    - Takes short positions in stocks with highest %K (top 3)
    """
    # Load and prepare data
    train_data = pd.read_csv('train_data.csv')
    crossval_data = pd.read_csv('crossval_data.csv')
    all_data = pd.concat([train_data, crossval_data]).sort_values(['Symbol', 'Date'])
    
    # Precompute %K values for all stocks
    k_cache = {}
    for symbol in range(20):
        stock_data = all_data[all_data['Symbol'] == symbol].copy()
        if stock_data.empty:
            k_cache[symbol] = pd.Series(dtype=float)
            continue
        
        # Calculate rolling high and low with proper min_periods and shift
        high_14 = (
            stock_data['Close']
            .rolling(window=14, min_periods=14)
            .max()
            .shift(1)  # Use data up to D-1
        )
        low_14 = (
            stock_data['Close']
            .rolling(window=14, min_periods=14)
            .min()
            .shift(1)  # Use data up to D-1
        )
        
        # Calculate %K
        latest_close = stock_data['Close'].shift(1)  # Use data up to D-1
        denominator = high_14 - low_14
        k_values = pd.Series(0.0, index=stock_data.index, dtype=float)
        mask = denominator != 0
        k_values[mask] = 100 * (latest_close[mask] - low_14[mask]) / denominator[mask]
        
        k_cache[symbol] = k_values
    
    # Initialize output with numpy zeros
    output_df = pd.DataFrame(0.0, index=range(3500, 4000), columns=range(20), dtype=float)
    
    for current_day in range(3500, 4000):
        k_values = []
        
        for symbol in range(20):
            if symbol in k_cache:
                k_series = k_cache[symbol]
                if current_day in k_series.index:
                    k_value = k_series[current_day]
                    if pd.isna(k_value):
                        k_value = 0
                else:
                    k_value = 0
            else:
                k_value = 0
            
            k_values.append((symbol, k_value))
        
        # Sort by %K (descending)
        k_values.sort(key=lambda x: x[1], reverse=True)
        
        # Assign weights
        weights = [0] * 20
        
        # Top 3 stocks (highest %K) get -1/3
        for i in range(3):
            weights[k_values[i][0]] = -1/3
        
        # Bottom 3 stocks (lowest %K) get +1/3
        for i in range(17, 20):
            weights[k_values[i][0]] = 1/3
        
        output_df.loc[current_day] = weights
    
    return output_df



def task1():
    Strategy1 = task1_Strategy1()
    Strategy2 = task1_Strategy2()
    Strategy3 = task1_Strategy3()
    Strategy4 = task1_Strategy4()
    Strategy5 = task1_Strategy5()

    performanceStrategy1 = backtester_without_TC(Strategy1)
    performanceStrategy2 = backtester_without_TC(Strategy2)
    performanceStrategy3 = backtester_without_TC(Strategy3)
    performanceStrategy4 = backtester_without_TC(Strategy4)
    performanceStrategy5 = backtester_without_TC(Strategy5)

    output_df = pd.DataFrame({'Strategy1':performanceStrategy1, 'Strategy2': performanceStrategy2, 'Strategy3': performanceStrategy3, 'Strategy4': performanceStrategy4, 'Strategy5': performanceStrategy5})
    output_df.to_csv('task1.csv')
    return



def calculate_strategy_metrics(weights_df, train_data, start_day, end_day):
    """Calculate strategy performance metrics for a given period."""
    daily_returns = []
    prev_weights = None
    
    for day in range(start_day, end_day):
        day_returns = []
        day_weights = weights_df.loc[day]
        
        # Calculate transaction costs if applicable
        if prev_weights is not None:
            turnover = sum(abs(day_weights - prev_weights))
            transaction_cost = turnover * 0.01
        else:
            transaction_cost = 0
        
        # Calculate returns for each symbol
        for symbol in range(20):
            price_today = train_data[(train_data['Date'] == day) & (train_data['Symbol'] == symbol)]
            price_tomorrow = train_data[(train_data['Date'] == day + 1) & (train_data['Symbol'] == symbol)]
            
            if not price_today.empty and not price_tomorrow.empty:
                ret = (price_tomorrow['Close'].values[0] / price_today['Close'].values[0]) - 1
                day_returns.append(ret * day_weights[symbol])
        
        # Calculate daily portfolio return
        if day_returns:
            daily_return = sum(day_returns) - transaction_cost
            daily_returns.append(daily_return)
        
        prev_weights = day_weights
    
    if not daily_returns:
        return 0, 0, 0
    
    # Calculate metrics
    returns = np.array(daily_returns)
    mean_return = np.mean(returns)
    volatility = np.std(returns) * np.sqrt(252)  # Annualized
    sharpe = mean_return * np.sqrt(252) / volatility if volatility > 0 else 0
    
    return mean_return, volatility, sharpe

def train_strategy_selector(strategy_weights, train_data, lookback_days=20):
    """Train a model to select the best strategy based on historical performance."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import joblib
    
    # Prepare training data
    X = []  # Features
    y = []  # Labels (best strategy)
    
    for current_day in range(3520, 4000):  # Start after lookback period
        # Calculate features for each strategy
        strategy_features = []
        for strategy_num, weights_df in enumerate(strategy_weights, 1):
            # Calculate performance metrics for lookback period
            mean_return, volatility, sharpe = calculate_strategy_metrics(
                weights_df, train_data, 
                current_day - lookback_days, current_day
            )
            
            # Add features
            strategy_features.extend([mean_return, volatility, sharpe])
        
        # Find best performing strategy
        best_strategy = 1
        best_sharpe = -float('inf')
        
        for i in range(5):  # 5 strategies
            sharpe = strategy_features[i*3 + 2]  # Sharpe is the third metric
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_strategy = i + 1
        
        X.append(strategy_features)
        y.append(best_strategy)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # Save model and scaler
    joblib.dump(model, 'task2_model.pkl')
    joblib.dump(scaler, 'task2_scaler.pkl')
    
    return model, scaler

def calculate_daily_returns(data):
    """Calculate daily returns for all symbols efficiently."""
    # Pivot data to get close prices for all symbols
    close_prices = data.pivot(index='Date', columns='Symbol', values='Close')
    
    # Calculate returns using next day's price
    returns = close_prices.pct_change().shift(-1)
    
    return returns

# Constants for Task 2
START_DAY = 3500
END_DAY = 4000
N_SYMBOLS = 20
LOOKBACK_DAYS = 20
TRADING_DAYS_PER_YEAR = 252

class SimpleStrategySelector:
    """Selects strategy with highest Sharpe ratio using walk-forward validation."""
    def __init__(self, lookback_days=LOOKBACK_DAYS):
        self.lookback_days = lookback_days
        self.sharpe_ratios = None
        self.strategy_history = []
    
    def fit(self, returns):
        """Calculate Sharpe ratios for the lookback period."""
        # Calculate rolling mean and std
        mean_returns = returns.rolling(window=self.lookback_days, min_periods=self.lookback_days).mean()
        std_returns = returns.rolling(window=self.lookback_days, min_periods=self.lookback_days).std()
        
        # Calculate Sharpe ratios (annualized)
        self.sharpe_ratios = (mean_returns * np.sqrt(TRADING_DAYS_PER_YEAR)) / (std_returns * np.sqrt(TRADING_DAYS_PER_YEAR))
    
    def predict(self, returns):
        """Predict best strategy based on Sharpe ratios."""
        if self.sharpe_ratios is None:
            self.fit(returns)
        
        # Get latest Sharpe ratios
        latest_sharpe = self.sharpe_ratios.iloc[-1]
        
        # Return strategy with highest Sharpe ratio
        best_strategy = latest_sharpe.idxmax()
        self.strategy_history.append(best_strategy)
        return best_strategy
    
    def get_strategy_stats(self):
        """Get statistics about strategy selection."""
        from collections import Counter
        strategy_counts = Counter(self.strategy_history)
        total_days = len(self.strategy_history)
        
        stats = {
            'strategy_counts': strategy_counts,
            'strategy_percentages': {k: v/total_days for k, v in strategy_counts.items()},
            'total_days': total_days,
            'unique_strategies': len(strategy_counts)
        }
        return stats

def task2():
    """Optimal Strategy Selection using Walk-Forward Validation.
    
    Implements a walk-forward validation approach to select the best strategy each day:
    1. Uses only data available up to day-1 for decisions
    2. Efficiently calculates returns using vectorized operations
    3. Maintains temporal integrity with proper data handling
    4. Uses a simple but robust Sharpe-based selector
    """
    # Load all historical data
    full_data = pd.concat([
        pd.read_csv('train_data.csv'), 
        pd.read_csv('crossval_data.csv')
    ]).sort_values('Date')
    
    # Get strategy weights
    strategies = {
        1: task1_Strategy1(),
        2: task1_Strategy2(),
        3: task1_Strategy3(),
        4: task1_Strategy4(),
        5: task1_Strategy5()
    }
    
    # Precompute daily returns for all symbols
    daily_returns = calculate_daily_returns(full_data)
    
    # Precompute strategy returns
    strategy_returns = pd.DataFrame(index=range(START_DAY, END_DAY))
    for strat in strategies:
        weights = strategies[strat].loc[START_DAY:END_DAY-1]
        # Calculate strategy returns using vectorized operations
        returns = (weights * daily_returns.loc[START_DAY:END_DAY-1]).sum(axis=1)
        strategy_returns[strat] = returns
    
    # Initialize strategy selector
    model = SimpleStrategySelector(lookback_days=LOOKBACK_DAYS)
    
    # Walk-forward selection
    selected_weights = pd.DataFrame(0, index=strategy_returns.index, columns=range(N_SYMBOLS))
    
    for day in strategy_returns.index:
        if day > START_DAY + LOOKBACK_DAYS:
            # Use only data up to day-1 for training
            train_window = slice(max(START_DAY, day-LOOKBACK_DAYS), day-1)
            model.fit(strategy_returns.loc[train_window])
            
            # Predict best strategy using data up to day-1
            best_strat = model.predict(strategy_returns.loc[:day-1])
            
            # Assign weights for current day
            selected_weights.loc[day] = strategies[best_strat].loc[day]
    
    # Calculate strategy selection statistics
    strategy_stats = model.get_strategy_stats()
    print("\nStrategy Selection Statistics:")
    print(f"Total trading days: {strategy_stats['total_days']}")
    print(f"Unique strategies used: {strategy_stats['unique_strategies']}")
    print("\nStrategy usage:")
    for strat, count in strategy_stats['strategy_counts'].items():
        print(f"Strategy {strat}: {count} days ({strategy_stats['strategy_percentages'][strat]:.1%})")
    
    # Save outputs
    selected_weights.to_csv('task2_weights.csv')
    
    # Calculate and save performance metrics
    strategy_returns['selected'] = (selected_weights * daily_returns.loc[START_DAY:END_DAY-1]).sum(axis=1)
    performance = {
        'Net Returns': ((1 + strategy_returns['selected']).cumprod().iloc[-1] - 1) * 100,
        'Sharpe Ratio': (strategy_returns['selected'].mean() * np.sqrt(TRADING_DAYS_PER_YEAR)) / 
                       (strategy_returns['selected'].std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    }
    pd.DataFrame([performance]).to_csv('task_2.csv')
    
    # Save model and statistics
    pd.to_pickle({
        'model': model,
        'strategy_stats': strategy_stats
    }, 'task2_model.pkl')
    
    return selected_weights




def calculate_turnover(prev_weights, curr_weights):
    """Calculate portfolio turnover with proper handling of edge cases."""
    if prev_weights is None:
        return 0
    
    # Ensure both are Series and aligned
    if isinstance(prev_weights, pd.Series):
        prev_weights = prev_weights.copy()
    else:
        prev_weights = pd.Series(prev_weights)
    
    if isinstance(curr_weights, pd.Series):
        curr_weights = curr_weights.copy()
    else:
        curr_weights = pd.Series(curr_weights)
    
    # Handle NaN values
    prev_weights = prev_weights.fillna(0)
    curr_weights = curr_weights.fillna(0)
    
    # Calculate turnover
    return abs(curr_weights - prev_weights).sum()

def evaluate_strategy_performance(weights_df, returns_df, transaction_cost=0.01):
    """Evaluate strategy performance with comprehensive metrics."""
    # Calculate portfolio returns
    portfolio_returns = (weights_df * returns_df).sum(axis=1)
    
    # Calculate turnover
    turnover = calculate_turnover_vectorized(weights_df)
    
    # Calculate metrics
    metrics = {
        'Total Return': (1 + portfolio_returns).prod() - 1,
        'Annualized Return': portfolio_returns.mean() * TRADING_DAYS_PER_YEAR,
        'Annualized Volatility': portfolio_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR),
        'Sharpe Ratio': calculate_sharpe(portfolio_returns),
        'Max Drawdown': calculate_max_drawdown(portfolio_returns),
        'Average Turnover': turnover.mean(),
        'Win Rate': (portfolio_returns > 0).mean(),
        'Profit Factor': abs(portfolio_returns[portfolio_returns > 0].sum() / 
                           portfolio_returns[portfolio_returns < 0].sum()) if len(portfolio_returns[portfolio_returns < 0]) > 0 else float('inf')
    }
    
    return metrics

def preprocess_data():
    """Preprocess and cache returns data efficiently."""
    full_data = pd.concat([
        pd.read_csv('train_data.csv'), 
        pd.read_csv('crossval_data.csv')
    ]).sort_values('Date')
    # Use pivot to get close prices for all symbols
    close_prices = full_data.pivot(index='Date', columns='Symbol', values='Close')
    returns = close_prices.pct_change()
    return returns

def calculate_turnover_vectorized(weights_df):
    """Calculate turnover efficiently using vectorized operations."""
    return (weights_df - weights_df.shift(1)).abs().sum(axis=1).fillna(0)

class TransactionCostCalculator:
    """Handles transaction cost calculations."""
    def __init__(self, tc_rate=0.01):
        self.tc_rate = tc_rate
    
    def calculate_net_returns(self, weights, returns):
        """Calculate net returns including transaction costs."""
        # Ensure proper alignment: use t-1 weights for t returns
        # Align columns and indices
        aligned_weights, aligned_returns = weights.align(returns, join='inner', axis=1)
        # Align indices for shift
        aligned_weights = aligned_weights.reindex(aligned_returns.index)
        portfolio_returns = (aligned_weights.shift(1) * aligned_returns).sum(axis=1)
        turnover = calculate_turnover_vectorized(aligned_weights)
        net_returns = portfolio_returns - (turnover * self.tc_rate)
        return net_returns

# Constants for Task 3
TRADING_DAYS_PER_YEAR = 252
START_DAY = 3500
END_DAY = 4000
LOOKBACK_DAYS = 63
N_SYMBOLS = 20
EPSILON = 1e-8
RISK_FREE_RATE = 0.02  # Annual risk-free rate

class EnsembleSelector:
    """Selects best strategy using walk-forward validation."""
    def __init__(self, strategies, lookback=LOOKBACK_DAYS, tc_calculator=None):
        self.strategies = strategies
        self.lookback = lookback
        self.tc_calculator = tc_calculator or TransactionCostCalculator()
        self.strategy_history = []
        self.strategy_returns = {}  # Track returns for each strategy
    
    def calculate_sharpe(self, returns, use_rf=True):
        """Calculate annualized Sharpe ratio.
        
        Args:
            returns: Series of returns
            use_rf: Whether to consider risk-free rate
            
        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) < 2:
            return 0
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            return 0
        
        # Adjust for risk-free rate if needed
        if use_rf:
            daily_rf = (1 + RISK_FREE_RATE) ** (1/TRADING_DAYS_PER_YEAR) - 1
            mean_return = mean_return - daily_rf
        
        # Apply annualization factor
        return mean_return * np.sqrt(TRADING_DAYS_PER_YEAR) / std_return
    
    def select_strategy(self, day, returns):
        """Select best strategy based on historical performance.
        
        Args:
            day: Current day (integer index)
            returns: DataFrame of returns
            
        Returns:
            Index of best performing strategy
        """
        scores = []
        
        for name, weights in self.strategies.items():
            # Get evaluation window using integer indexing
            start_idx = max(0, day - self.lookback)
            eval_weights = weights.iloc[start_idx:day]
            eval_returns = returns.iloc[start_idx:day]
            
            # Calculate net returns including transaction costs
            net_returns = self.tc_calculator.calculate_net_returns(eval_weights, eval_returns)
            
            # Store returns for diversity analysis
            self.strategy_returns[name] = net_returns
            
            # Calculate Sharpe ratio
            sharpe = self.calculate_sharpe(net_returns)
            scores.append((name, sharpe))
        
        # Select strategy with highest Sharpe ratio
        best_strategy = max(scores, key=lambda x: x[1])[0]
        self.strategy_history.append(best_strategy)
        return best_strategy
    
    def get_strategy_stats(self):
        """Get detailed statistics about strategy selection and performance."""
        from collections import Counter
        strategy_counts = Counter(self.strategy_history)
        total_days = len(self.strategy_history)
        
        # Calculate strategy diversity metrics
        diversity_metrics = {
            'strategy_counts': strategy_counts,
            'strategy_percentages': {k: v/total_days for k, v in strategy_counts.items()},
            'total_days': total_days,
            'unique_strategies': len(strategy_counts),
            'dominance_ratio': max(strategy_counts.values()) / total_days if total_days > 0 else 0,
            'strategy_returns': {
                name: {
                    'mean': returns.mean(),
                    'std': returns.std(),
                    'sharpe': self.calculate_sharpe(returns)
                }
                for name, returns in self.strategy_returns.items()
            }
        }
        
        return diversity_metrics

def optimize_lookback_window(strategies, returns, min_lookback=20, max_lookback=126, step=5):
    """Optimize the lookback window using cross-validation.
    
    Args:
        strategies: Dictionary of strategy weights
        returns: DataFrame of returns
        min_lookback: Minimum lookback period
        max_lookback: Maximum lookback period
        step: Step size for lookback periods
        
    Returns:
        Optimal lookback period and its performance
    """
    results = []
    
    for lookback in range(min_lookback, max_lookback + 1, step):
        # Initialize selector with current lookback
        selector = EnsembleSelector(strategies, lookback=lookback)
        
        # Track performance for this lookback period
        performance = []
        
        # Walk-forward validation
        for day in range(START_DAY + lookback, END_DAY):
            try:
                best_strat = selector.select_strategy(day, returns)
                if day in strategies[best_strat].index:
                    weights = normalize_weights(strategies[best_strat].loc[day])
                    # Calculate next day's return
                    if day + 1 < len(returns):
                        day_return = (returns.iloc[day] * weights).sum()
                        performance.append(day_return)
            except Exception:
                continue
        
        if performance:
            # Calculate Sharpe ratio for this lookback period
            sharpe = selector.calculate_sharpe(pd.Series(performance))
            results.append((lookback, sharpe))
    
    # Return best lookback period
    if results:
        return max(results, key=lambda x: x[1])
    return LOOKBACK_DAYS, 0

def calculate_drawdown(returns):
    """Calculate drawdown metrics from returns series.
    
    Args:
        returns: Series of returns
        
    Returns:
        Dictionary containing drawdown metrics
    """
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()
    
    # Calculate running maximum
    running_max = cum_returns.cummax()
    
    # Calculate drawdown
    drawdown = (cum_returns - running_max) / running_max
    
    # Calculate drawdown metrics
    max_drawdown = drawdown.min()
    avg_drawdown = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0
    drawdown_duration = (drawdown < 0).astype(int).groupby((drawdown < 0).astype(int).cumsum()).cumsum().max()
    
    return {
        'max_drawdown': max_drawdown,
        'avg_drawdown': avg_drawdown,
        'drawdown_duration': drawdown_duration,
        'drawdown_series': drawdown
    }

def calculate_rolling_performance_metrics(returns, window=63):
    """Calculate rolling performance metrics.
    
    Args:
        returns: Series of returns
        window: Rolling window size in days
        
    Returns:
        Dictionary containing rolling metrics
    """
    # Calculate rolling volatility (annualized)
    rolling_vol = returns.rolling(window=window, min_periods=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    # Calculate rolling Sharpe ratio
    rolling_mean = returns.rolling(window=window, min_periods=window).mean()
    rolling_sharpe = (rolling_mean * np.sqrt(TRADING_DAYS_PER_YEAR)) / rolling_vol
    
    # Calculate rolling drawdown
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.rolling(window=window, min_periods=window).max()
    rolling_drawdown = (cum_returns - rolling_max) / rolling_max
    
    return {
        'rolling_volatility': rolling_vol,
        'rolling_sharpe': rolling_sharpe,
        'rolling_drawdown': rolling_drawdown
    }

def calculate_performance_metrics(returns, weights=None, transaction_costs=None):
    """Calculate comprehensive performance metrics.
    
    Args:
        returns: Series of returns
        weights: Optional DataFrame of weights
        transaction_costs: Optional transaction cost rate
        
    Returns:
        Dictionary of performance metrics
    """
    # Basic return metrics
    total_return = (1 + returns).prod() - 1
    annualized_return = returns.mean() * TRADING_DAYS_PER_YEAR
    annualized_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    # Risk-adjusted return metrics
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
    
    # Drawdown analysis
    drawdown_metrics = calculate_drawdown(returns)
    
    # Calmar ratio (annualized return / max drawdown)
    calmar_ratio = abs(annualized_return / drawdown_metrics['max_drawdown']) if drawdown_metrics['max_drawdown'] != 0 else 0
    
    # Rolling metrics
    rolling_metrics = calculate_rolling_performance_metrics(returns)
    
    # Turnover and transaction costs
    if weights is not None:
        turnover = calculate_turnover_vectorized(weights)
        avg_turnover = turnover.mean()
        total_turnover = turnover.sum()
        
        if transaction_costs is not None:
            cost_impact = turnover * transaction_costs
            net_return = total_return - cost_impact.sum()
        else:
            cost_impact = 0
            net_return = total_return
    else:
        turnover = 0
        avg_turnover = 0
        total_turnover = 0
        cost_impact = 0
        net_return = total_return
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_vol,
        'sharpe_ratio': sharpe_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': drawdown_metrics['max_drawdown'],
        'avg_drawdown': drawdown_metrics['avg_drawdown'],
        'drawdown_duration': drawdown_metrics['drawdown_duration'],
        'avg_turnover': avg_turnover,
        'total_turnover': total_turnover,
        'cost_impact': cost_impact,
        'net_return': net_return,
        'rolling_metrics': rolling_metrics
    }

def normalize_weights(weights):
    """Normalize weights to ensure market neutrality.
    
    Args:
        weights: Series or array of weights
        
    Returns:
        Normalized weights that sum to 0 (market neutral)
    """
    # Convert to numpy array for efficiency
    weights = np.array(weights)
    
    # Handle NaN values
    weights = np.nan_to_num(weights, nan=0.0)
    
    # Calculate sum of absolute weights
    abs_sum = np.abs(weights).sum()
    
    # Normalize if sum is not zero
    if abs_sum > EPSILON:
        return weights / abs_sum
    return weights

def task3():
    print("Starting Task 3...")
    
    # Load and preprocess data
    train_data = pd.read_csv('train_data.csv')
    crossval_data = pd.read_csv('crossval_data.csv')
    data = pd.concat([train_data, crossval_data])
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')
    
    # Create a mapping between unique dates and indices
    unique_dates = data['Date'].unique()
    date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
    
    # Precompute returns for all symbols
    returns = data.pivot(index='Date', columns='Symbol', values='Close').pct_change()
    
    # Initialize strategy weights with float dtype
    strategy_weights = pd.DataFrame(0.0, index=unique_dates, columns=data['Symbol'].unique())
    
    # Process each day
    for i, current_date in enumerate(unique_dates):
        if i % 50 == 0:
            print(f"Processing day {i+1}/{len(unique_dates)}...")
            
        # Get data up to current date
        current_data = data[data['Date'] <= current_date].copy()
        
        # Compute weights for each strategy
        weights = {}
        for strategy_num in range(1, 6):
            w = compute_weights(current_data, strategy_num, current_date)
            if w is not None:
                weights[strategy_num] = w
        if not weights:
            continue
            
        # Calculate strategy returns
        strategy_returns = {}
        for strategy_num in weights:
            strategy_returns[strategy_num] = returns.loc[current_date] * weights[strategy_num]
        
        # Calculate strategy weights based on recent performance
        strategy_performance = {}
        for strategy_num in weights:
            # Calculate rolling Sharpe ratio
            recent_returns = returns.loc[:current_date].iloc[-20:] * weights[strategy_num]
            sharpe = recent_returns.mean() / (recent_returns.std() + 1e-6)
            strategy_performance[strategy_num] = sharpe.mean()
        
        # Normalize strategy weights
        total_performance = sum(max(0, perf) for perf in strategy_performance.values())
        if total_performance > 0:
            strategy_weights_dict = {
                num: max(0, perf) / total_performance 
                for num, perf in strategy_performance.items()
            }
        else:
            # Equal weights if no positive performance
            strategy_weights_dict = {num: 1.0/len(strategy_performance) for num in strategy_performance}
        
        # Combine strategy weights
        combined_weights = pd.Series(0.0, index=returns.columns)
        for strategy_num, weight in strategy_weights_dict.items():
            combined_weights += weight * weights[strategy_num]
        
        # Normalize combined weights
        if combined_weights.abs().sum() > 0:
            combined_weights = combined_weights / combined_weights.abs().sum()
        
        # Store weights
        strategy_weights.loc[current_date] = combined_weights
    
    # Calculate portfolio returns
    portfolio_returns = pd.Series(0.0, index=unique_dates)
    for i in range(1, len(unique_dates)):
        current_date = unique_dates[i]
        prev_date = unique_dates[i-1]
        
        # Get current weights and returns
        weights = strategy_weights.loc[prev_date]
        returns_today = data[data['Date'] == current_date].set_index('Symbol')['Close'].pct_change()
        
        # Calculate portfolio return
        portfolio_returns[current_date] = (weights * returns_today).sum()
    
    # Calculate metrics
    metrics = {
        'Total Return': (1 + portfolio_returns).prod() - 1,
        'Annualized Return': portfolio_returns.mean() * 252,
        'Annualized Volatility': portfolio_returns.std() * np.sqrt(252),
        'Sharpe Ratio': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252) + 1e-6),
        'Max Drawdown': (portfolio_returns.cumsum() - portfolio_returns.cumsum().cummax()).min(),
        'Average Turnover': strategy_weights.diff().abs().mean().mean(),
        'Win Rate': (portfolio_returns > 0).mean(),
        'Profit Factor': abs(portfolio_returns[portfolio_returns > 0].sum() / (portfolio_returns[portfolio_returns < 0].sum() + 1e-6))
    }
    
    # Save outputs
    pd.DataFrame(metrics, index=[0]).to_csv('task3_metrics.csv')
    strategy_weights.to_csv('task3_weights.csv')
    
    print("Task 3 completed successfully!")
    return metrics

def calculate_sharpe(returns):
    """Calculate Sharpe ratio with proper annualization."""
    if returns.empty:
        return 0.0
    excess_returns = returns - RISK_FREE_RATE/TRADING_DAYS_PER_YEAR
    return np.sqrt(TRADING_DAYS_PER_YEAR) * excess_returns.mean() / (returns.std() + EPSILON)

def calculate_turnover_vectorized(weights):
    """Calculate turnover using vectorized operations."""
    return abs(weights - weights.shift(1)).sum(axis=1)

def calculate_max_drawdown(returns):
    """Calculate maximum drawdown efficiently."""
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = cum_returns / rolling_max - 1
    return drawdowns.min()

def compute_weights(data, strategy_num, current_date):
    """Compute weights for a given strategy efficiently."""
    if data.empty:
        return None
        
    # Get data up to current_date
    current_data = data[data['Date'] <= current_date].copy()
    
    # Calculate returns once
    returns = current_data.pivot(index='Date', columns='Symbol', values='Close').pct_change()
    
    if strategy_num == 1:
        # Strategy 1: Mean reversion (50-day lookback)
        recent_returns = returns.iloc[-50:].mean()
        # Long bottom 6, short top 6
        sorted_returns = recent_returns.sort_values()
        weights = pd.Series(0.0, index=range(20))
        weights[sorted_returns.index[:6]] = 1/6  # Long bottom 6
        weights[sorted_returns.index[-6:]] = -1/6  # Short top 6
        return weights
        
    elif strategy_num == 2:
        # Strategy 2: Momentum (20-day lookback)
        recent_returns = returns.iloc[-20:].mean()
        # Long top 5, short bottom 5
        sorted_returns = recent_returns.sort_values()
        weights = pd.Series(0.0, index=range(20))
        weights[sorted_returns.index[-5:]] = 1/5  # Long top 5
        weights[sorted_returns.index[:5]] = -1/5  # Short bottom 5
        return weights
        
    elif strategy_num == 3:
        # Strategy 3: Volatility (20-day lookback)
        volatility = returns.iloc[-20:].std()
        # Long low vol 4, short high vol 4
        sorted_vol = volatility.sort_values()
        weights = pd.Series(0.0, index=range(20))
        weights[sorted_vol.index[:4]] = 1/4  # Long low vol 4
        weights[sorted_vol.index[-4:]] = -1/4  # Short high vol 4
        return weights
        
    elif strategy_num == 4:
        # Strategy 4: Volume
        volume = current_data.pivot(index='Date', columns='Symbol', values='Volume').iloc[-1]
        # Long low volume 4, short high volume 4
        sorted_vol = volume.sort_values()
        weights = pd.Series(0.0, index=range(20))
        weights[sorted_vol.index[:4]] = 1/4  # Long low volume 4
        weights[sorted_vol.index[-4:]] = -1/4  # Short high volume 4
        return weights
        
    elif strategy_num == 5:
        # Strategy 5: Price level
        prices = current_data.pivot(index='Date', columns='Symbol', values='Close').iloc[-1]
        # Long low price 3, short high price 3
        sorted_prices = prices.sort_values()
        weights = pd.Series(0.0, index=range(20))
        weights[sorted_prices.index[:3]] = 1/3  # Long low price 3
        weights[sorted_prices.index[-3:]] = -1/3  # Short high price 3
        return weights
    
    return None

def detect_market_regime(returns, window=63):
    """Detect market regime using volatility and trend."""
    # Calculate rolling metrics
    vol = returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    trend = returns.rolling(window).mean() * TRADING_DAYS_PER_YEAR
    
    # Fill NaN values with forward fill
    vol = vol.fillna(method='ffill')
    trend = trend.fillna(method='ffill')
    
    # Define regimes
    regime = pd.Series(index=returns.index, dtype=str)
    regime[(vol > vol.quantile(0.7)) & (trend < 0)] = 'HIGH_VOL_DOWN'
    regime[(vol > vol.quantile(0.7)) & (trend > 0)] = 'HIGH_VOL_UP'
    regime[(vol <= vol.quantile(0.7)) & (trend < 0)] = 'LOW_VOL_DOWN'
    regime[(vol <= vol.quantile(0.7)) & (trend > 0)] = 'LOW_VOL_UP'
    
    # Fill any remaining NaN values with the most common regime
    regime = regime.fillna(regime.mode().iloc[0])
    
    return regime

def calculate_strategy_weights(strategy_returns, lookback=20, regime=None):
    """Calculate strategy weights using exponential weighted Sharpe ratios."""
    weights = {}
    
    for strategy_num, returns in strategy_returns.items():
        # Calculate rolling Sharpe ratios
        rolling_sharpe = returns.rolling(lookback).apply(calculate_sharpe)
        rolling_sharpe = rolling_sharpe.fillna(0)  # Fill NaN values with 0
        
        # Apply regime-specific adjustments if regime is provided
        if regime is not None:
            # Calculate regime-specific Sharpe ratios
            regime_sharpe = {}
            for reg in regime.unique():
                mask = regime == reg
                if mask.any():
                    regime_sharpe[reg] = calculate_sharpe(returns[mask])
                else:
                    regime_sharpe[reg] = 0
            
            # Apply regime adjustments
            regime_adjustments = pd.Series(index=returns.index)
            for reg in regime.unique():
                mask = regime == reg
                regime_adjustments[mask] = regime_sharpe[reg]
            
            rolling_sharpe = rolling_sharpe * regime_adjustments
        
        # Use exponential weighting with protection against extreme values
        exp_sharpe = np.exp(np.clip(rolling_sharpe, -10, 10))
        weights[strategy_num] = exp_sharpe / exp_sharpe.sum()
    
    return weights

def apply_volatility_targeting(weights, returns, target_vol=0.10):
    """Scale weights to target a specific annualized volatility."""
    # Calculate realized volatility
    realized_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    # Calculate scaling factor
    scaling_factor = target_vol / (realized_vol + EPSILON)
    scaling_factor = np.clip(scaling_factor, 0, 2)  # Limit leverage
    
    return weights * scaling_factor

def apply_drawdown_control(weights, cumulative_returns, max_drawdown=-0.15):
    """Reduce exposure when drawdown exceeds threshold."""
    drawdown = calculate_max_drawdown(cumulative_returns)
    if drawdown < max_drawdown:
        # Reduce exposure proportionally to drawdown severity
        reduction_factor = 1 - (abs(drawdown) - abs(max_drawdown)) / abs(max_drawdown)
        return weights * reduction_factor
    return weights



def optimize_strategy_parameters(strategy_func, param_grid, metric='sharpe_ratio'):
    """Optimize strategy parameters using grid search."""
    best_params = None
    best_score = float('-inf')
    
    # Generate all parameter combinations
    param_combinations = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    
    for params in param_combinations:
        # Run strategy with current parameters
        weights_df = strategy_func(**params)
        
        # Calculate performance metrics
        metrics = calculate_portfolio_metrics(weights_df)
        
        # Update best parameters if current combination is better
        if metrics[metric] > best_score:
            best_score = metrics[metric]
            best_params = params
    
    return best_params, best_score

# Example parameter grid for Strategy 1
strategy1_param_grid = {
    'lookback_weeks': [30, 40, 50, 60],
    'num_positions': [4, 6, 8],
    'volatility_target': [0.10, 0.15, 0.20]
}

# Example usage:
# best_params, best_score = optimize_strategy_parameters(task1_Strategy1, strategy1_param_grid)



if __name__ == '__main__':
    task1()
    task2()
    task3()