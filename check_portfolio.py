"""
This script performs a comprehensive portfolio analysis, including data loading,
market data download, time series construction, performance metric calculation,
risk contribution analysis, chart generation, and HTML report creation.

It supports both 'defensive' and 'active' portfolio components and provides
a detailed backtest report with key financial metrics and visualizations.
"""

# Standard library imports
import os
import webbrowser
from datetime import datetime, timedelta

# Third-party library imports
import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px
from jinja2 import Template


# =============================================================================
# LOAD PORTFOLIO
# =============================================================================

def load_portfolio(core_file='core_portfolio.xlsx', active_file='active_portfolio.xlsx'):
    """
    Loads portfolio data from core and active Excel files, categorizes them,
    and consolidates them into a single DataFrame.

    Args:
        core_file (str): Path to the Excel file containing core (defensive) portfolio holdings.
        active_file (str): Path to the Excel file containing active portfolio holdings.

    Returns:
        pd.DataFrame: A consolidated DataFrame with all portfolio holdings,
                      including a 'type' column indicating 'defensive' or 'active'.
    """
    # Load defensive portfolio holdings
    core_df = pd.read_excel(core_file)
    core_df['type'] = 'defensive'

    # Load active portfolio holdings
    active_df = pd.read_excel(active_file)
    active_df['type'] = 'active'

    # Combine core and active portfolios
    portfolio = pd.concat([core_df, active_df], ignore_index=True)

    # Ensure the 'ticker' column is of string type for consistent handling
    portfolio['ticker'] = portfolio['ticker'].astype(str)

    return portfolio


# =============================================================================
# DOWNLOAD MARKET DATA
# =============================================================================

def download_price_data(tickers, start_date, end_date):
    """
    Downloads historical stock price data for a given list of tickers
    within a specified date range using yfinance.

    Args:
        tickers (list or str): A single ticker symbol (str) or a list of ticker symbols (list).
        start_date (str or datetime): The start date for data download (inclusive).
        end_date (str or datetime): The end date for data download (exclusive).

    Returns:
        pd.DataFrame: A DataFrame containing the 'Close' prices for the specified tickers.
                      The DataFrame index is a DatetimeIndex.

    Raises:
        ValueError: If no data is downloaded for the given tickers and date range.
    """
    # Download data using yfinance
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False, # Use False to get both Close and Adj Close
        progress=False     # Do not display download progress bar
    )

    # Check if any data was downloaded
    if data.empty:
        raise ValueError("No data downloaded for the specified tickers and date range.")

    # Handle cases where yfinance returns a MultiIndex (multiple tickers)
    # or a single-level column index (single ticker).
    if isinstance(data.columns, pd.MultiIndex):
        # Extract 'Close' prices for multiple tickers
        # Note: We specifically use 'Close' for capital gains calculation.
        # Total return (including yield) is calculated separately.
        prices = data["Close"]
    else:
        # If only one ticker, 'data' itself contains the prices
        if "Close" in data.columns:
            prices = data["Close"]
        else:
            prices = data

    # Check for tickers that have very little data (e.g., less than 1 month)
    # This identifies "young" tickers that might be truncating the whole dataset if we use dropna()
    data_counts = prices.count()
    max_count = data_counts.max()
    threshold = 20 # Minimum trading days
    
    young_tickers = data_counts[data_counts < threshold].index.tolist()
    if young_tickers:
        print(f"Warning: The following tickers have very little data and will be excluded to preserve backtest length: {young_tickers}")
        prices = prices.drop(columns=young_tickers)

    # Instead of dropping rows with ANY NaN (which truncates the whole history to the youngest asset),
    # we return the raw prices and handle alignment later in the value calculation.
    return prices


# =============================================================================
# SECTOR AND INDUSTRY DATA
# =============================================================================

def get_sector_industry_data(tickers):
    """
    Fetches comprehensive information including sector, industry, market cap, growth, and P/E for a list of tickers using yfinance.

    Args:
        tickers (list): A list of ticker symbols.

    Returns:
        pd.DataFrame: A DataFrame with tickers as index and metadata as columns.
    """
    data = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            sector = info.get('sector', 'Others')
            industry = info.get('industry', 'Others')
            div_yield = info.get('dividendYield', 0.0)
            name = info.get('longName', ticker)
            market_cap = info.get('marketCap', 0.0)
            eps_growth = info.get('earningsGrowth', 0.0)
            forward_pe = info.get('forwardPE', 0.0)
            
            # yfinance often returns dividendYield as a decimal (e.g. 0.05 for 5%)
            if div_yield is None:
                div_yield = 0.0
            
            # yfinance dividend yield format can be inconsistent (decimal vs percentage).
            # Examples from the current portfolio:
            # GNP.AX: 0.71  -> Intended as 0.71% (0.0071)
            # ROYL.AX: 5.05 -> Intended as 5.05% (0.0505)
            # Most stocks don't have > 20% yield. If it's > 0.2, it's almost certainly
            # returned in percentage format (e.g., 5.0 for 5%) rather than decimal.
            # Even for 0.71, it's more likely 0.71% (returned as 0.71) than 71% (returned as 0.71).
            if div_yield > 0.0:
                div_yield = div_yield / 100.0

            data.append({
                'ticker': ticker,
                'name': name,
                'sector': sector,
                'industry': industry,
                'dividendYield': div_yield,
                'marketCap': market_cap,
                'eps_growth': eps_growth,
                'forward_pe': forward_pe
            })
        except Exception as e:
            print(f"Warning: Could not fetch info for {ticker}: {e}")
            data.append({
                'ticker': ticker,
                'name': ticker,
                'sector': 'Others',
                'industry': 'Others',
                'dividendYield': 0.0,
                'marketCap': 0.0,
                'eps_growth': 0.0,
                'forward_pe': 0.0
            })
    
    return pd.DataFrame(data).set_index('ticker')


# =============================================================================
# BUILD PORTFOLIO TIMESERIES
# =============================================================================
def build_portfolio_timeseries(price_data, portfolio_df, total_investment=None, rebalance=False, force_percentage=None, percentage_format=None):
    """
    Constructs time series for portfolio position values, separating them into
    'defensive' and 'active' layers, and calculates the total portfolio value.

    Args:
        price_data (pd.DataFrame): DataFrame of historical closing prices for all tickers,
                                   with dates as index and tickers as columns.
        portfolio_df (pd.DataFrame): DataFrame containing portfolio holdings with 'ticker',
                                     'quantity', and 'type' (e.g., 'defensive', 'active') columns.
        total_investment (float, optional): Total capital to invest if quantities are given as percentages.
        rebalance (bool): Whether to rebalance the portfolio to maintain target weights daily.
        force_percentage (bool, optional): If True, treat quantities as percentages. If False, treat as shares.
                                          If None, auto-detect.
        percentage_format (str, optional): 'decimal' (0.05=5%) or 'whole' (5=5%). Only used if force_percentage is True.

    Returns:
        dict: A dictionary containing pandas Series for:
              - 'defensive': Total value of defensive positions over time.
              - 'active': Total value of active positions over time.
              - 'total': Total portfolio value over time.
              - 'positions': DataFrame of individual position values over time.
    """
    # Initialize a DataFrame to store the value of each position over time
    pos_values = pd.DataFrame(index=price_data.index)

    # Check if quantities are percentages (e.g., sum to ~1 or ~100)
    # or if any quantity looks like a percentage (small value).
    
    total_qty = portfolio_df["quantity"].sum()
    max_qty = portfolio_df["quantity"].max()
    
    is_percentage = False
    scale_factor = 1.0
    
    if force_percentage is True:
        is_percentage = True
        if percentage_format == 'decimal':
            scale_factor = 1.0
            print("Forcing quantity as decimal percentage (e.g. 0.05 = 5%).")
        elif percentage_format == 'whole':
            scale_factor = 100.0
            print("Forcing quantity as whole-number percentage (e.g. 5 = 5%).")
        else:
            # Fallback to heuristic if format not specified
            if max_qty <= 1.05:
                scale_factor = 1.0
                print("Forcing quantity as decimal percentage (scale 1.0).")
            else:
                scale_factor = 100.0
                print("Forcing quantity as whole-number percentage (scale 100.0).")
    elif force_percentage is False:
        is_percentage = False
        print("Forcing quantity as actual share counts.")
    else:
        # Auto-detection heuristic
        if max_qty <= 1.0:
            is_percentage = True
            scale_factor = 1.0
            print(f"Auto-detected quantity as decimal percentage (Max Qty: {max_qty:.4f}).")
        elif 2 <= total_qty <= 1000: # Expanded range to be more inclusive of whole-number percentages
            is_percentage = True
            scale_factor = 100.0
            print(f"Auto-detected quantity as whole-number percentage (Total Qty: {total_qty:.2f}).")
        elif 0.5 <= total_qty <= 2.0:
            is_percentage = True
            scale_factor = 1.0
            print(f"Auto-detected quantity as decimal percentage (Total Qty: {total_qty:.4f}).")
        elif any(portfolio_df["quantity"].between(0.0001, 0.2)) and total_qty < 5.0:
            is_percentage = True
            scale_factor = 1.0
            print(f"Auto-detected quantity as decimal percentage based on small values (Sum: {total_qty:.4f}).")

    if is_percentage and total_investment is None:
        total_investment = 1000000 # Default to $1M if not provided
        print(f"Quantities detected as percentages. Using default total investment of ${total_investment:,.2f}")
        
        # If the total percentage is significantly different from 100% (or 1.0),
        # we should warn the user as it might lead to "missing" value or unexpected weighting.
        if (total_qty/scale_factor) < 0.98:
            print(f"Warning: Total portfolio quantity ({total_qty/scale_factor*100:.1f}%) is less than 100%. Remaining will be treated as uninvested (Cash).")

    # Separate tickers into core (defensive) and active lists
    core_tickers = portfolio_df.loc[portfolio_df["type"] == "defensive", "ticker"].tolist()
    active_tickers = portfolio_df.loc[portfolio_df["type"] == "active", "ticker"].tolist()

    if is_percentage and rebalance:
        print("Rebalancing enabled. Maintaining target weights quarterly.")
        # Rebalancing logic
        # 1. Calculate daily returns for all assets
        returns_df = price_data.pct_change().fillna(0)
        
        # 2. Setup target weights
        # Note: scale_factor handles whether input was 0.1 or 10 for 10%
        # We do NOT normalize here to 1.0 because the user might have specified a total allocation < 100%
        target_weights = {row["ticker"]: row["quantity"] / scale_factor for _, row in portfolio_df.iterrows()}
        
        # Re-detect scale factor if weights look like they were actually meant to be whole but were treated as decimal
        # (e.g. if 5 was intended as 5% but scale_factor was 1, it becomes 500%)
        total_target_sum = sum(target_weights.values())
        if total_target_sum > 2.0 and scale_factor == 1.0:
             print(f"Warning: Target weights sum to {total_target_sum:.2f}, which is unusually high for decimal format. Re-scaling to whole-number percentage.")
             scale_factor = 100.0
             target_weights = {row["ticker"]: row["quantity"] / scale_factor for _, row in portfolio_df.iterrows()}
             total_target_sum = sum(target_weights.values())

        # Initialize total portfolio value
        total_ts = pd.Series(0.0, index=price_data.index)
        
        # Initialize position values
        for ticker in target_weights:
            if ticker in price_data.columns:
                pos_values[ticker] = 0.0
        
        # 3. Iterate through time and rebalance quarterly
        current_total = total_investment
        current_weights = {}
        
        # Pre-calculate the total target weight to handle uninvested cash
        total_target_sum = sum(target_weights.values())

        last_rebalance_month = -1

        for i in range(len(price_data.index)):
            date = price_data.index[i]
            
            # A. Update portfolio value and drift weights based on returns
            if i > 0:
                daily_ret = 0
                for t, w in current_weights.items():
                    asset_ret = returns_df.loc[date, t] if not pd.isna(returns_df.loc[date, t]) else 0
                    daily_ret += w * asset_ret
                
                new_total = current_total * (1 + daily_ret)
                
                # Update weights based on drift (before rebalancing)
                if new_total > 0:
                    current_weights = {t: (w * current_total * (1 + (returns_df.loc[date, t] if not pd.isna(returns_df.loc[date, t]) else 0))) / new_total
                                      for t, w in current_weights.items()}
                current_total = new_total
            
            total_ts.iloc[i] = current_total
            
            # B. Check for Rebalancing (Quarterly: Jan, Apr, Jul, Oct)
            # Rebalance on the first available day of the quarter or the very first day of the simulation
            current_month = date.month
            is_quarter_start_month = current_month in [1, 4, 7, 10]
            
            should_rebalance = (i == 0) or (is_quarter_start_month and current_month != last_rebalance_month)
            
            if should_rebalance:
                last_rebalance_month = current_month
                
                # Find which tickers are available on this date
                available_tickers = [t for t in target_weights if t in price_data.columns and not pd.isna(price_data.loc[date, t])]
                
                sum_available_target = sum(target_weights[t] for t in available_tickers)
                if sum_available_target > 0:
                    # Scale up available weights to reach the total intended allocation
                    scale_up = total_target_sum / sum_available_target
                    current_weights = {t: target_weights[t] * scale_up for t in available_tickers}
                else:
                    current_weights = {}

                # Ensure all tickers in target_weights are present in current_weights (even if 0)
                for t in target_weights:
                    if t not in current_weights:
                        current_weights[t] = 0.0

            # SPECIAL FIX: Force final weights on the last day to match target weights EXACTLY
            # if they are all available, to ensure the report shows the requested percentages.
            if i == len(price_data.index) - 1:
                available_tickers = [t for t in target_weights if t in price_data.columns and not pd.isna(price_data.loc[date, t])]
                if len(available_tickers) == len(target_weights):
                    current_weights = target_weights
            
            # C. Assign position values for today's report
            for t, w in current_weights.items():
                pos_values.loc[date, t] = current_total * w

    else:
        # ORIGINAL LOGIC: No rebalancing (Buy and Hold)
        # Iterate through each holding in the portfolio to calculate its value over time
        for _, row in portfolio_df.iterrows():
            ticker = row["ticker"]
            qty = row["quantity"]
            
            # Only include tickers for which price data was successfully downloaded
            if ticker in price_data.columns:
                # We must NOT fill forward from the start of the dataframe index if the asset didn't exist yet.
                # yf.download might return NaNs for the period before a stock was listed.
                ticker_prices = price_data[ticker]
                # Find the first non-NaN index (inception)
                first_valid = ticker_prices.first_valid_index()
                if first_valid is not None:
                    # Fill forward only from inception
                    prices_from_inception = ticker_prices.loc[first_valid:].ffill()
                    # Create a series for the full index, initialized with NaN, then update with prices from inception
                    full_ticker_prices = pd.Series(np.nan, index=price_data.index)
                    full_ticker_prices.update(prices_from_inception)
                    
                    if is_percentage:
                        # For percentage-based portfolios, we assume the allocation is made at the INITIAL prices
                        # available for that asset or at the start of the simulation.
                        # Here we treat it as: (Weight / Scale) * Total_Investment / Initial_Price = Quantity
                        initial_price = full_ticker_prices.loc[first_valid]
                        calculated_qty = (qty / scale_factor) * total_investment / initial_price
                        pos_values[ticker] = full_ticker_prices * calculated_qty
                    else:
                        pos_values[ticker] = full_ticker_prices * qty

    # For the portfolio calculation, we only consider dates where we have data.
    # We fill NaNs with 0 ONLY for assets that haven't launched yet or have no data,
    # so they don't contribute to total value but also don't cause the sum to be NaN.
    pos_values = pos_values.fillna(0)

    # RE-SYNC TOTAL_TS: In rebalance mode, pos_values.sum() might have drift due to daily rounding.
    # We ensure total_ts reflects the sum of position values for consistent weighting.
    if is_percentage and rebalance:
        total_ts = pos_values.sum(axis=1)

    # Filter these lists to include only tickers for which position values were calculated
    core_cols = [t for t in core_tickers if t in pos_values.columns]
    active_cols = [t for t in active_tickers if t in pos_values.columns]

    # Calculate the total value time series for each layer and the overall portfolio
    core_ts = pos_values[core_cols].sum(axis=1)
    active_ts = pos_values[active_cols].sum(axis=1)
    
    if not (is_percentage and rebalance):
        total_ts = pos_values.sum(axis=1)

    return {
        "defensive": core_ts,
        "active": active_ts,
        "total": total_ts,
        "positions": pos_values
    }


# =============================================================================
# RETURNS
# =============================================================================

def calculate_returns(timeseries_dict):
    """
    Calculates the daily percentage returns for each time series in the input dictionary.

    Args:
        timeseries_dict (dict): A dictionary where keys are identifiers (e.g., 'total', 'defensive', 'active')
                                and values are pandas Series or DataFrames representing asset values over time.

    Returns:
        dict: A dictionary of the same structure as `timeseries_dict`, but with values
              transformed into daily percentage returns (pandas Series or DataFrames).
              Rows with NaN values (e.g., the first return) are dropped.
    """
    returns = {}

    for k, ts in timeseries_dict.items():
        # Ensure the item is a pandas Series or DataFrame before calculating returns
        if isinstance(ts, pd.Series) or isinstance(ts, pd.DataFrame):
            # Calculate percentage change and drop the first row (which will be NaN)
            returns[k] = ts.pct_change().dropna()

    return returns


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

def calculate_performance_metrics(returns_series, benchmark_returns, risk_free_rate=0.02, annual_yield=0.0, benchmark_yield=0.0):
    """
    Calculates a suite of performance metrics for a given series of returns,
    including risk-adjusted returns and drawdown metrics.

    Args:
        returns_series (pd.Series): A pandas Series of daily returns for the portfolio or asset.
        benchmark_returns (pd.Series): A pandas Series of daily returns for the benchmark.
        risk_free_rate (float): The annual risk-free rate (default is 0.02, or 2%).
        annual_yield (float): The annual yield (dividends/coupons) to be included in the performance.
        benchmark_yield (float): The annual yield of the benchmark for a fair comparison.

    Returns:
        dict: A dictionary containing the calculated performance metrics:
              - "Annualized Return" (float)
              - "Cumulative Return" (float)
              - "Volatility" (float): Annualized standard deviation of returns.
              - "Sharpe Ratio" (float)
              - "Sortino Ratio" (float)
              - "Max Drawdown" (float)
              - "Beta" (float or NaN): Sensitivity to benchmark, NaN if insufficient common data.
              - "Market Exposure Effect (Cum.)" (float or NaN): The total return derived solely from benchmark exposure (Beta).
              - "Alpha (Risk-Adj) Annualized" (float or NaN): Jensen's Alpha per year.
              - "Alpha (Risk-Adj) Cumulative" (float or NaN): Total excess return over the whole period due to alpha.
              - "Outperformance Annualized" (float or NaN): Simple difference in annual returns.
              - "Outperformance Cumulative" (float or NaN): Simple difference in total returns.
              - "Information Ratio" (float or NaN): Excess return per unit of tracking error.
    """
    # Adjust returns to include the annual yield
    # Convert annual yield to daily yield
    daily_yield = (1 + annual_yield) ** (1/252) - 1
    adjusted_returns = returns_series + daily_yield

    # Adjust benchmark returns for yield (Fair Comparison)
    daily_b_yield = (1 + benchmark_yield) ** (1/252) - 1
    adjusted_benchmark_returns = benchmark_returns + daily_b_yield

    # Convert annual risk-free rate to a daily rate
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    # Calculate cumulative return
    cum_ret = (1 + adjusted_returns).prod() - 1
    n_days = len(adjusted_returns)
    # Calculate annualized return
    ann_ret = (1 + cum_ret) ** (252/n_days) - 1
    # Calculate annualized volatility
    vol = adjusted_returns.std() * np.sqrt(252)
    # Calculate Sharpe Ratio
    sharpe = ((adjusted_returns.mean() - daily_rf) / adjusted_returns.std()) * np.sqrt(252)
    # Isolate downside returns for Sortino Ratio calculation
    downside = adjusted_returns[adjusted_returns < 0]
    # Calculate Sortino Ratio
    sortino = (ann_ret - risk_free_rate) / (downside.std() * np.sqrt(252))
    # Calculate cumulative product for drawdown
    cumulative = (1 + adjusted_returns).cumprod()
    # Find the running maximum for drawdown calculation
    peak = cumulative.cummax()
    # Calculate drawdown
    drawdown = (cumulative - peak) / peak
    # Determine the maximum drawdown
    max_dd = drawdown.min()
    # Find common dates between portfolio and benchmark returns for relative metrics
    common = adjusted_returns.index.intersection(benchmark_returns.index)

    # Calculate Beta, Alpha, and Information Ratio only if sufficient common data exists
    if len(common) > 30: # A reasonable threshold for statistical significance
        y = adjusted_returns.loc[common]
        x = adjusted_benchmark_returns.loc[common]
        # Perform linear regression to get beta (slope)
        slope, intercept, r, p, stderr = stats.linregress(x, y)
        beta = slope
        # Calculate annualized benchmark return for the COMMON period (geometric)
        benchmark_cum_common = (1 + x).prod() - 1
        benchmark_ann_common = (1 + benchmark_cum_common)**(252/len(x)) - 1
        
        # Calculate portfolio annualized return for the SAME common period (geometric)
        y_cum_common = (1 + y).prod() - 1
        y_ann_common = (1 + y_cum_common)**(252/len(y)) - 1

        # Calculate Annualized Jensen's Alpha
        alpha_ann = y_ann_common - (risk_free_rate + beta * (benchmark_ann_common - risk_free_rate))
        
        # Calculate Market Exposure Effect (Beta-driven part of return)
        # Market Effect = Beta * (Benchmark_Cum - RiskFree_Cum)
        risk_free_cum = (1 + risk_free_rate)**(len(y)/252) - 1
        market_effect_cum = beta * (benchmark_cum_common - risk_free_cum)

        # Calculate Cumulative Jensen's Alpha
        # Total_Cum = RiskFree_Cum + Market_Effect_Cum + Alpha_Cum
        alpha_cum = y_cum_common - (risk_free_cum + market_effect_cum)
        
        # Simple Outperformance
        outperformance_ann = y_ann_common - benchmark_ann_common
        outperformance_cum = y_cum_common - benchmark_cum_common

        # Calculate excess returns relative to the benchmark
        excess_returns = y - x
        # Calculate Information Ratio, handle division by zero for standard deviation
        information_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() != 0 else np.nan
    else:
        # If not enough common data, set benchmark-relative metrics to NaN
        beta = np.nan
        alpha = np.nan
        outperformance = np.nan
        information_ratio = np.nan

    return {
        "Annualized Return": ann_ret,
        "Cumulative Return": cum_ret,
        "Volatility": vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_dd,
        "Beta": beta,
        "Market Exposure Effect (Cum.)": market_effect_cum,
        "Alpha (Risk-Adj) Annualized": alpha_ann,
        "Alpha (Risk-Adj) Cumulative": alpha_cum,
        "Outperformance Annualized": outperformance_ann,
        "Outperformance Cumulative": outperformance_cum,
        "Information Ratio": information_ratio
    }


# =============================================================================
# VAR / CVAR
# =============================================================================

def calculate_var_cvar(returns, confidence=0.95):
    """
    Calculates Value at Risk (VaR) and Conditional Value at Risk (CVaR)
    for a given series of returns.

    Args:
        returns (pd.Series): A pandas Series of daily returns.
        confidence (float): The confidence level for VaR and CVaR calculation (e.9., 0.95 for 95%).

    Returns:
        dict: A dictionary containing:
              - "VaR" (float): Value at Risk at the specified confidence level.
              - "CVaR" (float): Conditional Value at Risk (Expected Shortfall) at the specified confidence level.
    """
    # Calculate Value at Risk (VaR) at the specified confidence level.
    # VaR represents the maximum expected loss over a given time horizon at a given confidence level.
    var = np.percentile(returns, (1 - confidence) * 100)

    # Calculate Conditional Value at Risk (CVaR) or Expected Shortfall.
    # CVaR is the expected loss given that the loss is greater than or equal to the VaR.
    cvar = returns[returns <= var].mean()

    return {"VaR": var, "CVaR": cvar}


# =============================================================================
# MONTE CARLO SIMULATION
# =============================================================================

def run_monte_carlo_simulation(initial_value, returns_series, num_simulations=10000, forecast_days=252):
    """
    Runs a Monte Carlo simulation to forecast future portfolio values.

    Args:
        initial_value (float): The starting value of the portfolio for the simulation.
        returns_series (pd.Series): Historical daily returns of the portfolio.
        num_simulations (int): The number of simulation paths to generate.
        forecast_days (int): The number of trading days to forecast into the future.

    Returns:
        pd.DataFrame: A DataFrame where each column represents a single simulation path
                      of cumulative portfolio values.
    """
    # Calculate daily drift and volatility from historical returns
    # Drift represents the average daily return
    drift = returns_series.mean()
    # Volatility represents the standard deviation of daily returns
    volatility = returns_series.std()

    # Generate random daily returns for each simulation path
    # The formula used here is based on Geometric Brownian Motion:
    # daily_returns = exp(drift - 0.5 * volatility^2 + volatility * Z)
    # where Z is a random variable from a standard normal distribution.
    # We use norm.ppf to get the inverse of the cumulative distribution function
    # for generating random numbers based on a normal distribution.
    # np.random.rand generates uniform random numbers between 0 and 1.
    random_shocks = norm.ppf(np.random.rand(forecast_days, num_simulations))
    daily_returns = np.exp(drift - 0.5 * volatility**2 + volatility * random_shocks)

    # Initialize a DataFrame to store all simulation paths
    simulation_df = pd.DataFrame(index=range(forecast_days + 1), columns=range(num_simulations))
    simulation_df.iloc[0] = initial_value

    # Generate each simulation path
    for i in range(num_simulations):
        # Calculate cumulative simulated values for the current path
        # Each column represents a simulation path
        simulation_df.iloc[1:, i] = initial_value * daily_returns[:, i].cumprod()

    return simulation_df


# =============================================================================
# RISK CONTRIBUTION
# =============================================================================

def calculate_risk_contribution(pos_values, total_ts, asset_returns=None):
    """
    Calculates the risk contribution of each position to the total portfolio volatility.

    Args:
        pos_values (pd.DataFrame): DataFrame of individual position values over time.
        total_ts (pd.Series): Series of total portfolio value over time.
        asset_returns (pd.DataFrame, optional): Pre-calculated daily returns for the underlying assets.
                                               If None, calculates returns from pos_values.

    Returns:
        pd.DataFrame: A DataFrame containing:
                      - "Weight": The current weight of each position in the portfolio.
                      - "Risk Contribution": The contribution of each position to the total portfolio risk.
                      - "% Risk Contribution": The percentage contribution of each position to the total portfolio risk.
    """
    # Calculate daily returns for individual positions and drop any NaN values
    if asset_returns is not None:
        # Filter asset returns to match the tickers in pos_values
        returns = asset_returns[pos_values.columns].dropna()
    else:
        returns = pos_values.pct_change().dropna()
    
    # Calculate the current weight of each position in the portfolio
    weights = pos_values.iloc[-1] / total_ts.iloc[-1]
    # Calculate the annualized covariance matrix of returns
    cov = returns.cov() * 252
    # Calculate total portfolio volatility
    port_vol = np.sqrt(weights.T @ cov @ weights)
    # Calculate Marginal Contribution to Risk (MCR)
    mcr = cov @ weights / port_vol
    # Calculate Component Contribution to Risk (CCR)
    ccr = weights * mcr
    # Calculate Percentage Risk Contribution
    pct = ccr / port_vol
    return pd.DataFrame({
        "Weight": weights,
        "Risk Contribution": ccr,
        "% Risk Contribution": pct
    })


# =============================================================================
# CHARTS
# =============================================================================

def generate_charts(ts_data, risk_contrib, corr_matrix, benchmark_ticker, annual_yield=0.0):
    """
    Generates interactive Plotly charts for portfolio performance, asset allocation,
    and correlation matrix.

    Args:
        ts_data (dict): A dictionary containing time series data, including:
                        - 'total' (pd.Series): Total portfolio value over time.
                        - 'benchmark' (pd.Series): Benchmark value over time (can be empty).
        risk_contrib (pd.DataFrame): DataFrame with risk contribution data, including "Weight" column.
        corr_matrix (pd.DataFrame): Correlation matrix of position returns.
        benchmark_ticker (str): Ticker symbol of the benchmark asset.
        annual_yield (float): The annual yield to include in total return performance.

    Returns:
        dict: A dictionary where keys are chart identifiers and values are HTML strings
              representing the generated Plotly charts (e.g., 'value', 'allocation', 'corr').
    """
    charts = {}
    
    # 1. Portfolio Performance Chart
    fig = go.Figure()
    fig.update_layout(height=700, title="Portfolio Performance Over Time")

    # Calculate percentage performance relative to the start of the data for both portfolio and benchmark.
    # The main function ensures ts_data["total"] and ts_data["benchmark"] (if not empty)
    # are aligned to a common starting point. So we can use the first value of total for normalization.
    initial_reference_value = ts_data["total"].iloc[0]

    # Add portfolio performance trace (including yield) if yield is present
    total_returns = ts_data["total"].pct_change().fillna(0)
    
    if annual_yield > 0:
        daily_yield = (1 + annual_yield) ** (1/252) - 1
        adjusted_returns = total_returns + daily_yield
        # Fix the first day (which was set to 0 but should be 0 + daily_yield if we want to be precise,
        # but the very first day has no return. Let's keep first return as 0 but yield starts from day 2?)
        # Actually, let's just use cumulative product of (1 + adjusted_returns)
        performance_total_return = ((1 + adjusted_returns).cumprod() - 1) * 100
        
        fig.add_trace(go.Scatter(x=performance_total_return.index, y=performance_total_return, name="Portfolio % Total Return (inc. Yield)", mode='lines'))

    # Add portfolio capital gain only trace for comparison (Solid Red)
    # Important: We recalculate cumulative performance from returns to handle
    # the changing denominator of a multi-asset portfolio correctly.
    performance_cap_gain = ((1 + total_returns).cumprod() - 1) * 100
    fig.add_trace(go.Scatter(x=performance_cap_gain.index, y=performance_cap_gain, name="Portfolio % Capital Gain", mode='lines', line=dict(color='red')))

    # Add benchmark performance trace if available (Solid Black)
    if not ts_data["benchmark"].empty:
        performance_benchmark = ((ts_data["benchmark"] / initial_reference_value) - 1) * 100
        fig.add_trace(go.Scatter(x=performance_benchmark.index, y=performance_benchmark, name=f"Benchmark ({benchmark_ticker}) % Performance", mode='lines', line=dict(color='black')))

    # Format y-axis as percentage
    fig.update_layout(yaxis_tickformat=".0f%")
    charts["value"] = fig.to_html(full_html=False)

    # 2. Portfolio Allocation Pie Chart
    weights = risk_contrib["Weight"]
    pie = px.pie(values=weights, names=weights.index, height=600, title="Portfolio Allocation by Weight")
    charts["allocation"] = pie.to_html(full_html=False)

    # 3. Correlation Heatmap
    # Custom colorscale: gray for sentinel (-2.0), blue for strong negative, green for less correlated (around 0), red for strong positive
    colorscale = [
        [0.0, "rgb(128,128,128)"],  # Sentinel value -2.0 (mapped to 0.0 in normalized scale) -> Gray
        [0.33, "blue"],             # -1.0 (mapped to ~0.33 in normalized scale) -> Blue
        [0.66, "green"],            # 0.0 (mapped to ~0.66 in normalized scale) -> Green
        [1.0, "red"]                # 1.0 (mapped to 1.0 in normalized scale) -> Red
    ]

    # Create a go.Heatmap trace
    heatmap_trace = go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale=colorscale,
        zmin=-2.0,  # Extend zmin to include the sentinel value
        zmax=1,
        colorbar_title="Correlation"
    )

    # Create the figure and add the heatmap trace
    fig_corr = go.Figure(data=[heatmap_trace])
    fig_corr.update_layout(
        title="Asset Correlation Matrix",
        height=600,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange="reversed" # To display y-axis in the correct order
    )

    # Add text annotations for correlation values (excluding NaN diagonal)
    annotations = []
    for i, row in enumerate(corr_matrix.index):
        for j, col in enumerate(corr_matrix.columns):
            # Only add annotation if the value is not the sentinel value (-2.0)
            if corr_matrix.iloc[i, j] != -2.0:
                annotations.append(
                    dict(
                        x=col,
                        y=row,
                        text=f"{corr_matrix.iloc[i, j]:.2f}",
                        xref="x1",
                        yref="y1",
                        showarrow=False,
                        font=dict(color="black") # You might need to adjust text color based on background color for readability
                    )
                )
    fig_corr.update_layout(annotations=annotations)

    charts["corr"] = fig_corr.to_html(full_html=False)

    return charts


def generate_sector_industry_analysis(risk_contrib, sector_industry_df, include_yield=True):
    """
    Analyzes portfolio weighting per sector and industry.

    Args:
        risk_contrib (pd.DataFrame): DataFrame with "Weight" for each ticker.
        sector_industry_df (pd.DataFrame): DataFrame with 'sector' and 'industry' for each ticker.
        include_yield (bool): Whether to include dividend yield columns in the table.

    Returns:
        dict: A dictionary containing:
              - 'table_html': HTML string for the weighting table.
              - 'pie_chart_html': HTML string for the sector pie chart.
    """
    # Merge risk contribution weights with sector/industry data
    analysis_df = risk_contrib[['Weight']].merge(sector_industry_df, left_index=True, right_index=True, how='left')
    analysis_df['sector'] = analysis_df['sector'].fillna('Others')
    analysis_df['industry'] = analysis_df['industry'].fillna('Others')

    # Calculate sector weights
    sector_weights = analysis_df.groupby('sector')['Weight'].sum().sort_values(ascending=False)
    
    # Calculate industry weights within sectors
    industry_weights = analysis_df.groupby(['sector', 'industry'])['Weight'].sum().reset_index()
    industry_weights = industry_weights.sort_values(['sector', 'Weight'], ascending=[True, False])

    # Generate HTML table
    yield_header = "<th>Avg. Div Yield (%)</th>" if include_yield else ""
    table_html = f"""
    <table>
        <thead>
            <tr>
                <th>Sector / Industry</th>
                <th>Weight (%)</th>
                {yield_header}
            </tr>
        </thead>
        <tbody>
    """
    for sector, s_weight in sector_weights.items():
        # Calculate weighted average dividend yield for the sector
        sector_data = analysis_df[analysis_df['sector'] == sector]
        
        table_html += f"""
            <tr style="background-color: #eef7ff; font-weight: bold;">
                <td>{sector if sector != "Unknown" else "Others"}</td>
                <td>{s_weight*100:.2f}%</td>
        """
        if include_yield:
            # Re-normalize weights within the sector for yield calculation
            sector_yield = (sector_data['dividendYield'] * sector_data['Weight']).sum() / s_weight if s_weight > 0 else 0
            table_html += f"<td>{sector_yield*100:.2f}%</td>"
        
        table_html += "</tr>"
        
        s_industries = industry_weights[industry_weights['sector'] == sector]
        for _, row in s_industries.iterrows():
            table_html += f"""
                <tr>
                    <td style="padding-left: 20px;">&bull; {row['industry'] if row['industry'] != "Unknown" else "Others"}</td>
                    <td>{row['Weight']*100:.2f}%</td>
            """
            if include_yield:
                # Calculate weighted average dividend yield for the industry
                industry_data = analysis_df[(analysis_df['sector'] == sector) & (analysis_df['industry'] == row['industry'])]
                industry_yield = (industry_data['dividendYield'] * industry_data['Weight']).sum() / row['Weight'] if row['Weight'] > 0 else 0
                table_html += f"<td>{industry_yield*100:.2f}%</td>"
            
            table_html += "</tr>"
    table_html += "</tbody></table>"

    # Generate Sector Pie Chart
    pie = px.pie(
        values=sector_weights.values,
        names=sector_weights.index,
        height=650,
        title="Portfolio Allocation by Sector",
        hole=0.4
    )
    pie_chart_html = pie.to_html(full_html=False)

    return {
        'table_html': table_html,
        'pie_chart_html': pie_chart_html
    }


def generate_portfolio_holdings_analysis(risk_contrib, sector_industry_df, price_data, portfolio_df):
    """
    Generates a table of portfolio holdings organized by defensive then active components,
    sorted by weight from largest to lowest.
    """
    # Merge all data
    holdings = risk_contrib[['Weight']].merge(sector_industry_df, left_index=True, right_index=True, how='left')
    holdings = holdings.merge(portfolio_df[['ticker', 'type']].drop_duplicates().set_index('ticker'), left_index=True, right_index=True, how='left')
    
    # Calculate percentage growth since the start of the backtest
    # price_data columns are tickers, index are dates
    growth_pct = {}
    for ticker in holdings.index:
        if ticker in price_data.columns:
            ticker_prices = price_data[ticker].dropna()
            if not ticker_prices.empty:
                start_price = ticker_prices.iloc[0]
                end_price = ticker_prices.iloc[-1]
                growth_pct[ticker] = (end_price / start_price - 1)
            else:
                growth_pct[ticker] = 0.0
        else:
            growth_pct[ticker] = 0.0
    
    holdings['growth_pct'] = pd.Series(growth_pct)
    
    # Organize by defensive then active, then by weight
    # Map 'defensive' to 0 and 'active' to 1 for sorting
    type_map = {'defensive': 0, 'active': 1}
    holdings['type_sort'] = holdings['type'].map(type_map)
    holdings = holdings.sort_values(['type_sort', 'Weight'], ascending=[True, False])
    
    # Generate HTML Table
    table_html = """
    <table>
        <thead>
            <tr>
                <th>Component</th>
                <th>Ticker</th>
                <th>Name</th>
                <th>Sector</th>
                <th>Sub-sector/Industry</th>
                <th>Weight (%)</th>
                <th>Market Cap</th>
                <th>Forward P/E</th>
                <th>% Growth</th>
                <th>Div Yield (%)</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for ticker, row in holdings.iterrows():
        # Format Market Cap
        mc = row['marketCap']
        if mc >= 1e12:
            mc_str = f"{mc/1e12:.2f}T"
        elif mc >= 1e9:
            mc_str = f"{mc/1e9:.2f}B"
        elif mc >= 1e6:
            mc_str = f"{mc/1e6:.2f}M"
        elif mc > 0:
            mc_str = f"{mc:,.0f}"
        else:
            mc_str = "-"
        
        # Replace "Unknown" with "-"
        sector = row['sector'] if row['sector'] != "Unknown" else "Others"
        industry = row['industry'] if row['industry'] != "Unknown" else "Others"
            
        table_html += f"""
            <tr>
                <td style="font-weight: bold;">{row['type'].title()}</td>
                <td>{ticker}</td>
                <td>{row['name']}</td>
                <td>{sector}</td>
                <td>{industry}</td>
                <td>{row['Weight']*100:.2f}%</td>
                <td>{mc_str}</td>
                <td>{f"{row['forward_pe']:.2f}" if row['forward_pe'] != 0 else "-"}</td>
                <td style="color: {'green' if row['growth_pct'] > 0 else 'red' if row['growth_pct'] < 0 else 'black'}">
                    {row['growth_pct']*100:+.2f}%
                </td>
                <td>{row['dividendYield']*100:.2f}%</td>
            </tr>
        """
        
    table_html += "</tbody></table>"
    return table_html


def generate_monte_carlo_chart(mc_simulations):
    """
    Generates an interactive Plotly chart showing a subset of Monte Carlo simulation paths
    and the mean path, with Y-axis in percentage change from initial value.

    Args:
        mc_simulations (pd.DataFrame): DataFrame containing all Monte Carlo simulation paths.

    Returns:
        str: HTML string representing the generated Plotly chart.
    """
    fig = go.Figure()
    fig.update_layout(title="Monte Carlo Simulation: Portfolio Future Paths (% Change)", height=600)

    # Calculate percentage change from initial value for all paths
    initial_value_per_path = mc_simulations.iloc[0] # This will be a Series if mc_simulations has multiple columns
    # Perform element-wise division and then subtract 1 and multiply by 100
    # Ensure that initial_value_per_path is correctly broadcasted or aligned
    mc_simulations_pct = (mc_simulations.div(initial_value_per_path, axis=1) - 1) * 100

    # Select a subset of significant paths for plotting
    num_paths_to_plot = 1000
    total_simulations = mc_simulations_pct.shape[1]

    # Identify the best and worst performing paths based on their final percentage values
    final_values_pct = mc_simulations_pct.iloc[-1]
    best_path_idx = final_values_pct.idxmax()
    worst_path_idx = final_values_pct.idxmin()

    # Select a random sample of other paths
    # Exclude best and worst path indices from random selection
    other_paths_indices = [i for i in range(total_simulations) if i != best_path_idx and i != worst_path_idx]
    
    # Ensure we don't try to sample more than available, especially if total_simulations is small
    num_random_to_select = max(0, min(num_paths_to_plot - 2, len(other_paths_indices)))
    selected_other_indices = np.random.choice(other_paths_indices, num_random_to_select, replace=False)

    # Combine selected indices, ensuring uniqueness and including best/worst
    selected_indices = list(set([best_path_idx, worst_path_idx] + list(selected_other_indices)))

    # Add selected simulation paths as thin, translucent lines
    for i in selected_indices:
        fig.add_trace(go.Scatter(x=mc_simulations_pct.index, y=mc_simulations_pct[i], mode='lines', name=None, showlegend=False, line=dict(color='rgba(31, 119, 180, 0.3)', width=1), opacity=0.3))

    # Add specific traces for best and worst paths with clearer visibility
    fig.add_trace(go.Scatter(x=mc_simulations_pct.index, y=mc_simulations_pct[best_path_idx], mode='lines', name='Best Path (% Change)', line=dict(color='green', width=1.5)))
    fig.add_trace(go.Scatter(x=mc_simulations_pct.index, y=mc_simulations_pct[worst_path_idx], mode='lines', name='Worst Path (% Change)', line=dict(color='red', width=1.5)))

    # Calculate and add the mean path
    mean_path_pct = mc_simulations_pct.mean(axis=1)
    fig.add_trace(go.Scatter(x=mean_path_pct.index, y=mean_path_pct, mode='lines', name='Mean Path (% Change)', line=dict(color='darkblue', width=2, dash='dash')))

    fig.update_layout(xaxis_title='Days into Future', yaxis_title='Portfolio Value (% Change)', yaxis_tickformat=".0f%")

    return fig.to_html(full_html=False)


# =============================================================================
# HTML REPORT
# =============================================================================

def generate_html_report(metrics, charts, report_title, inception_date, include_yield=True):
    """
    Generates a comprehensive HTML report summarizing portfolio performance,
    metrics, allocation, and correlations.

    The report uses a Jinja2 template to embed dynamic data and Plotly charts.

    Args:
        metrics (dict): A dictionary containing calculated performance metrics for
                        different portfolio layers (e.g., 'defensive', 'active', 'total').
        charts (dict): A dictionary containing HTML strings of generated Plotly charts.
        report_title (str): The title to display at the top of the HTML report.
        inception_date (datetime): The start date of the backtest for display.
        include_yield (bool): Whether yield was included in performance calculations.

    Returns:
        None: The function writes the HTML report to "portfolio_report.html".
    """
    formatted_date = inception_date.strftime("%Y-%m-%d")
    template = Template("""
<html>

<head>
<style>
    body {
        font-family: Arial, sans-serif;
        margin: 20px 5%;
        padding: 0;
        background-color: #f4f4f4;
        color: #333;
    }
    h1 {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    h2 {
        color: #34495e;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    h3 {
        color: #34495e;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    pre {
        background-color: #ecf0f1;
        border: 1px solid #bdc3c7;
        padding: 15px;
        border-radius: 5px;
        overflow-x: auto;
    }
    .chart-container {
        margin-bottom: 30px;
        padding: 20px;
        background-color: #fff;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    th {
        background-color: #f2f2f2;
        color: #555;
    }
    tr:nth-child(even) {
        background-color: #f9f9f9;
    }
.side-by-side-container {
    display: flex;
    flex-wrap: wrap; /* Allows items to wrap to the next line on smaller screens */
    gap: 20px; /* Space between the items */
    margin-bottom: 30px;
}
.side-by-side-container > div {
    flex: 1; /* Each child div takes equal space */
    min-width: 300px; /* Minimum width before wrapping */
}
.charts-column {
    display: flex;
    flex-direction: column;
    justify-content: space-around; /* Distribute space evenly */
}
.charts-column .chart-container {
    flex: 1; /* Make charts grow to fill vertical space */
}
</style>
<title>Portfolio Report</title>
</head>

<body>

<h1>{{ report_title }}</h1>

<h2>Portfolio (Since {{ inception_date }})</h2>
<div class="chart-container">
    {{ charts.value | safe }}
</div>

<div class="side-by-side-container">
    <div class="metrics-column">
        <h2>Metrics</h2>
        <div class="chart-container">
            {% for layer, data in metrics.items() %}
            {% if layer != 'benchmark' %}
            <h3>{{ layer | title }} Component Metrics</h3>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Portfolio (Since {{ inception_date }})</th>
                        <th>Benchmark (Since {{ inception_date }})</th>
                        <th>Difference</th>
                    </tr>
                </thead>
                <tbody>
                    {% for metric, value in data.items() %}
                    {% if not (layer == 'total' and metric in ['MC_Expected_Drawdown_Pct', 'MC_VaR_99_Pct', 'MC_VaR_95_Pct', 'MC_Expected_Upside_95_Pct']) and not (not include_yield and metric == "Estimated Yield") %}
                    <tr>
                        <td>{{ metric }}</td>
                        {# Portfolio Column #}
                        {% if metric in ["Annualized Return", "Cumulative Return", "Volatility", "Max Drawdown", "VaR", "CVaR", "Estimated Yield"] %}
                            <td>{{ "%.2f%%" | format(value * 100) if value is number else value }}</td>
                        {% elif metric in ["Market Exposure Effect (Cum.)", "Alpha (Risk-Adj) Annualized", "Alpha (Risk-Adj) Cumulative", "Outperformance Annualized", "Outperformance Cumulative"] %}
                            <td>{{ "%.2f%%" | format(value * 100) if value is number else value }}</td>
                        {% else %}
                            <td>{{ "%.4f" | format(value) if value is number else value }}</td>
                        {% endif %}

                        {# Benchmark Column #}
                        {% if metric in ["Beta", "Market Exposure Effect (Cum.)", "Alpha (Risk-Adj) Annualized", "Alpha (Risk-Adj) Cumulative", "Outperformance Annualized", "Outperformance Cumulative", "Information Ratio"] %}
                            <td>-</td>
                            <td>-</td>
                        {% else %}
                            {% set b_val = metrics.benchmark[metric] %}
                            {% if metric in ["Annualized Return", "Cumulative Return", "Volatility", "Max Drawdown", "VaR", "CVaR", "Estimated Yield"] %}
                                <td>{{ "%.2f%%" | format(b_val * 100) if b_val is number else b_val }}</td>
                                {% set diff = (value - b_val) if (value is number and b_val is number) else 0 %}
                                <td style="color: {{ 'green' if diff > 0 else 'red' if diff < 0 else 'black' }}">
                                    {{ "%+.2f%%" | format(diff * 100) if diff != 0 else "-" }}
                                </td>
                            {% else %}
                                <td>{{ "%.4f" | format(b_val) if b_val is number else b_val }}</td>
                                {% set diff = (value - b_val) if (value is number and b_val is number) else 0 %}
                                <td style="color: {{ 'green' if diff > 0 else 'red' if diff < 0 else 'black' }}">
                                    {{ "%.4f" | format(diff) if diff != 0 else "-" }}
                                </td>
                            {% endif %}
                        {% endif %}
                    </tr>
                    {% endif %}
                    {% endfor %}
                </tbody>
            </table>
            {% endif %}
            {% endfor %}
         </div>
     </div>

     <div class="charts-column">
         <h2>Allocation</h2>
         <div class="chart-container">
             {{ charts.allocation | safe }}
         </div>

         <h2>Correlation</h2>
         <div class="chart-container">
             {{ charts.corr | safe }}
         </div>
     </div>
 </div>

<div class="chart-container">
     <h2>Sector & Industry Exposure Analysis</h2>
     <p>Insights on portfolio weighting per sector and industry to identify concentration risks and diversification levels.</p>
     <div class="side-by-side-container">
         <div style="flex: 1.2;">
             <h3>Weighting per Sector and Industry</h3>
             {{ charts.sector_table | safe }}
         </div>
         <div style="flex: 0.8;">
             <h3>Sector Distribution</h3>
             {{ charts.sector_pie | safe }}
         </div>
     </div>
</div>

<div class="chart-container">
     <h2>Portfolio Holdings Analysis</h2>
     <p>Detailed breakdown of portfolio positions, organized by component type (Defensive vs. Active) and sorted by weighting.</p>
     {{ charts.holdings_table | safe }}
</div>

<div class="chart-container">
     <h2>Monte Carlo Simulation Results (Total Portfolio)</h2>
     <table>
         <thead>
             <tr>
                 <th>Metric</th>
                 <th>Value</th>
             </tr>
         </thead>
         <tbody>
             {% if metrics["total"]["MC_Expected_Drawdown_Pct"] is defined %}
             <tr>
                 <td>Forward Expected Drawdown</td>
                 <td>{{ "%.2f%%" | format(metrics["total"]["MC_Expected_Drawdown_Pct"] * 100) if metrics["total"]["MC_Expected_Drawdown_Pct"] is number else "N/A" }}</td>
             </tr>
             {% endif %}
             {% if metrics["total"]["MC_VaR_99_Pct"] is defined %}
             <tr>
                 <td>99% VaR (1-Year)</td>
                 <td>{{ "%.2f%%" | format(metrics["total"]["MC_VaR_99_Pct"] * 100) if metrics["total"]["MC_VaR_99_Pct"] is number else "N/A" }}</td>
             </tr>
             {% endif %}
             {% if metrics["total"]["MC_VaR_95_Pct"] is defined %}
             <tr>
                 <td>95% VaR (1-Year) Percent</td>
                 <td>{{ "%.2f%%" | format(metrics["total"]["MC_VaR_95_Pct"] * 100) if metrics["total"]["MC_VaR_95_Pct"] is number else "N/A" }}</td>
             </tr>
             {% endif %}
             {% if metrics["total"]["MC_Expected_Upside_95_Pct"] is defined %}
             <tr>
                 <td>95% Expected Upside (1-Year)</td>
                 <td>{{ "%.2f%%" | format(metrics["total"]["MC_Expected_Upside_95_Pct"] * 100) if metrics["total"]["MC_Expected_Upside_95_Pct"] is number else "N/A" }}</td>
             </tr>
             {% endif %}
         </tbody>
     </table>
 </div>

{% if charts["monte_carlo"] is defined %}
<div class="chart-container">
    <h2>Monte Carlo Simulation Paths</h2>
    {{ charts.monte_carlo | safe }}
</div>
{% endif %}



</body>

</html>

""")

    # Render the template with the provided metrics, charts, and title
    html = template.render(metrics=metrics, charts=charts, report_title=report_title, inception_date=formatted_date, include_yield=include_yield)

    # Write the generated HTML content to a file
    with open("portfolio_report.html","w") as f:
        f.write(html)

    print("Report generated: portfolio_report.html")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Main function to execute the portfolio analysis workflow.

    This function orchestrates the following steps:
    1. Prompts the user for a report title and benchmark ticker.
    2. Loads portfolio data.
    3. Downloads historical price data for portfolio assets and the benchmark.
    4. Handles benchmark fallback if the primary benchmark data is not found.
    5. Builds portfolio time series data (total, defensive, active, and individual positions).
    6. Calculates daily returns for all time series.
    7. Normalizes benchmark time series for charting comparison.
    8. Calculates a comprehensive set of performance metrics (returns, volatility, Sharpe, Sortino, drawdown, VaR, CVaR, Beta, Alpha, Information Ratio) for each portfolio layer.
    9. Calculates risk contribution for individual positions.
    10. Generates various interactive charts (performance, allocation, correlation).
    11. Generates and saves an HTML report.
    12. Opens the generated HTML report in the default web browser.

    Raises:
        ValueError: If benchmark ticker (and fallback) data is not found in downloaded prices.
    """
    # Prompt user for report title
    report_title = input("Enter the portfolio report title: ")
    inception_date_str = input("Enter the inception date (YYYY-MM-DD, default: 5 years ago): ")
    print("Starting portfolio analysis...")

    # Load portfolio data from Excel files
    portfolio = load_portfolio()
    # Get unique tickers from the portfolio for data download
    tickers = portfolio["ticker"].unique().tolist()

    # Prompt user for benchmark ticker, with a default value
    benchmark_ticker_input = input("Enter the benchmark ticker (default: A200.AX): ")
    benchmark_ticker = benchmark_ticker_input if benchmark_ticker_input else "A200.AX"

    # Ask if user wants to include yield in performance
    include_yield_input = input("Include total return including yield in the report? (y/n, default: y): ")
    include_yield = include_yield_input.lower() != 'n'

    # Ask if user wants to rebalance
    rebalance_input = input("Rebalance portfolio to maintain initial weights? (y/n, default: n): ")
    rebalance = rebalance_input.lower() == 'y'

    # Explicitly ask for quantity type
    print("\nQuantity Type:")
    print("1. Percentage Allocation")
    print("2. Actual Quantity of Securities (e.g., 100 shares)")
    print("3. Auto-detect (default)")
    qty_type_input = input("Select quantity type (1/2/3): ")
    
    force_percentage = None
    percentage_format = None
    if qty_type_input == '1':
        force_percentage = True
        print("\nPercentage Format:")
        print("a. Decimal (e.g., 0.05 = 5%)")
        print("b. Whole number (e.g., 5 = 5%)")
        print("c. Auto-detect (default)")
        pct_format_input = input("Select percentage format (a/b/c): ").lower()
        if pct_format_input == 'a':
            percentage_format = 'decimal'
        elif pct_format_input == 'b':
            percentage_format = 'whole'
    elif qty_type_input == '2':
        force_percentage = False

    # Ensure benchmark ticker and a fallback ticker (^AXJO) are included in the tickers list for download
    if benchmark_ticker not in tickers:
        tickers.append(benchmark_ticker)
    if "^AXJO" not in tickers: # Always add ^AXJO as a fallback
        tickers.append("^AXJO")

    # Define the date range for historical data download
    end = datetime.now()
    if inception_date_str:
        try:
            start = datetime.strptime(inception_date_str, "%Y-%m-%d")
        except ValueError:
            print(f"Warning: Invalid date format '{inception_date_str}'. Falling back to 5 years ago.")
            start = end - timedelta(days=5*365)
    else:
        start = end - timedelta(days=5*365) # Approximately 5 years

    # Download historical price data for all identified tickers
    prices = download_price_data(tickers, start, end)

    # Final check and potential fallback for benchmark ticker from downloaded data
    if benchmark_ticker not in prices.columns:
        if "^AXJO" in prices.columns:
            print(f"Warning: Primary benchmark {benchmark_ticker} not found, falling back to ^AXJO as benchmark.")
            benchmark_ticker = "^AXJO"
        else:
            raise ValueError(f"Benchmark ticker {benchmark_ticker} (and fallback ^AXJO) not found in downloaded data. Available columns: {prices.columns.tolist()}")

    # Build time series for total portfolio, defensive, active layers, and individual positions
    ts = build_portfolio_timeseries(prices, portfolio, rebalance=rebalance, force_percentage=force_percentage, percentage_format=percentage_format)
    # Calculate daily returns for all portfolio time series
    returns = calculate_returns(ts)
    # Calculate daily returns for the confirmed benchmark
    benchmark = prices[benchmark_ticker].pct_change().dropna()

    # Calculate cumulative benchmark value, normalized to start at the same point as the portfolio
    # This ensures a fair comparison on the performance chart.
    common_index = ts["total"].index.intersection(benchmark.index)
    if not common_index.empty:
        initial_total_value = ts["total"].loc[common_index[0]]
        cumulative_benchmark_returns = (1 + benchmark.loc[common_index]).cumprod()
        # Normalize benchmark to start at the same value as the total portfolio for visual comparison
        ts["benchmark"] = cumulative_benchmark_returns * (initial_total_value / cumulative_benchmark_returns.iloc[0])
    else:
        print("Warning: No common dates between portfolio total and benchmark for charting. Benchmark chart will not be generated.")
        ts["benchmark"] = pd.Series(dtype='float64') # Assign an empty series if no common index

    # Initialize dictionary to store performance metrics for each layer
    metrics = {}

    # Fetch Sector and Industry Data (and Dividend Yield)
    print("Fetching sector, industry and dividend yield data...")
    portfolio_tickers = portfolio["ticker"].unique().tolist()
    # Include benchmark ticker for fair yield comparison
    all_tickers_for_info = list(set(portfolio_tickers + [benchmark_ticker]))
    sector_industry_df = get_sector_industry_data(all_tickers_for_info)

    # Calculate estimated yields
    # Calculate returns for individual assets to compute risk and correlation correctly
    # (Using prices instead of position values to avoid bias from rebalancing)
    asset_returns = prices[ts["positions"].columns].pct_change().dropna()

    # Calculate risk contribution for individual positions to get weights
    risk = calculate_risk_contribution(ts["positions"], ts["total"], asset_returns=asset_returns)
    
    if include_yield:
        # Merge risk (weights) with sector_industry_df (yields)
        yield_analysis_df = risk[['Weight']].merge(sector_industry_df[['dividendYield']], left_index=True, right_index=True, how='left')
        yield_analysis_df['dividendYield'] = yield_analysis_df['dividendYield'].fillna(0)
        
        # Calculate estimated portfolio yield
        portfolio_yield = (yield_analysis_df['Weight'] * yield_analysis_df['dividendYield']).sum()
        
        # Calculate yields for components (defensive and active)
        layer_yields = {"total": portfolio_yield}
        for layer in ["defensive", "active"]:
            layer_tickers = portfolio.loc[portfolio["type"] == layer, "ticker"].tolist()
            layer_risk = risk[risk.index.isin(layer_tickers)]
            layer_weight_sum = layer_risk['Weight'].sum()
            
            if layer_weight_sum > 0:
                norm_weights = layer_risk['Weight'] / layer_weight_sum
                layer_div_yields = sector_industry_df.loc[sector_industry_df.index.isin(layer_tickers), 'dividendYield'].fillna(0)
                layer_yield = (norm_weights * layer_div_yields).sum()
                layer_yields[layer] = layer_yield
            else:
                layer_yields[layer] = 0.0

        # Calculate benchmark yield
        benchmark_yield = sector_industry_df.loc[benchmark_ticker, 'dividendYield'] if benchmark_ticker in sector_industry_df.index else 0.0
        if pd.isna(benchmark_yield): benchmark_yield = 0.0
    else:
        portfolio_yield = 0.0
        layer_yields = {"total": 0.0, "defensive": 0.0, "active": 0.0}
        benchmark_yield = 0.0
        # Zero out dividend yields in sector_industry_df to ensure they don't show up in analysis tables
        # Note: We keep them for the Holdings table even if not used in performance
        pass

    # Calculate performance metrics for the benchmark itself
    benchmark_metrics = calculate_performance_metrics(benchmark, benchmark, benchmark_yield=benchmark_yield)
    benchmark_var_cvar = calculate_var_cvar(benchmark)
    benchmark_metrics["VaR"] = benchmark_var_cvar["VaR"]
    benchmark_metrics["CVaR"] = benchmark_var_cvar["CVaR"]
    metrics["benchmark"] = benchmark_metrics

    # Calculate performance metrics for defensive, active, and total portfolio layers
    for layer in ["defensive","active","total"]:
        layer_returns = returns[layer]
        performance_metrics = calculate_performance_metrics(
            layer_returns,
            benchmark,
            annual_yield=layer_yields[layer],
            benchmark_yield=benchmark_yield
        )
        # For VaR and CVaR, we use the returns INCLUDING the yield (Total Return Risk)
        daily_yield = (1 + layer_yields[layer]) ** (1/252) - 1
        adjusted_layer_returns = layer_returns + daily_yield
        var_cvar = calculate_var_cvar(adjusted_layer_returns)
        
        performance_metrics["VaR"] = var_cvar["VaR"]
        performance_metrics["CVaR"] = var_cvar["CVaR"]
        performance_metrics["Estimated Yield"] = layer_yields[layer]
        metrics[layer] = performance_metrics

    # Monte Carlo Simulation
    print("Running Monte Carlo Simulation (10,000 paths, 1-year forecast)...")
    initial_portfolio_value = ts["total"].iloc[-1] # Last known total portfolio value
    # Use total portfolio returns for simulation
    mc_simulations = run_monte_carlo_simulation(initial_portfolio_value, returns["total"], num_simulations=10000, forecast_days=252)

    # Calculate statistics from Monte Carlo simulation results
    final_mc_values = mc_simulations.iloc[-1]
    mc_mean = final_mc_values.mean()
    mc_median = final_mc_values.median()
    mc_var_95 = np.percentile(final_mc_values, 5) # 5th percentile for 95% VaR
    mc_cvar_95 = final_mc_values[final_mc_values <= mc_var_95].mean() # CVaR is the mean of the values less than or equal to VaR

    # Calculate percentage returns for each simulation path from initial value
    mc_returns_percentage = (mc_simulations.iloc[-1] / mc_simulations.iloc[0] - 1)

    # Calculate Forward Expected Drawdown (average of max drawdowns across paths)
    drawdowns = []
    for col in mc_simulations.columns:
        cumulative = mc_simulations[col]
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        drawdowns.append(drawdown.min()) # Max drawdown for this path
    mc_expected_drawdown = np.mean(drawdowns) # Average of max drawdowns

    # Calculate VaR (percentage loss)
    mc_var_95_pct = np.percentile(mc_returns_percentage, 5) # 5th percentile of returns

    # Calculate Expected Upside (percentage gain)
    mc_expected_upside_95_pct = np.percentile(mc_returns_percentage, 95) # 95th percentile of returns

    metrics["total"]["MC_Expected_Drawdown_Pct"] = mc_expected_drawdown
    metrics["total"]["MC_VaR_99_Pct"] = np.percentile(mc_returns_percentage, 1)
    metrics["total"]["MC_VaR_95_Pct"] = mc_var_95_pct
    metrics["total"]["MC_Expected_Upside_95_Pct"] = mc_expected_upside_95_pct

    # Calculate correlation matrix using asset returns
    corr = asset_returns.corr()
    # Set diagonal to NaN to be handled separately for coloring as gray
    # Set diagonal to NaN to be handled separately for coloring as gray, using pandas mask for writability
    # Set diagonal to a sentinel value (-2.0) to be colored gray
    corr = corr.mask(np.eye(len(corr), dtype=bool), -2.0)

    # Generate interactive charts
    charts = generate_charts(ts, risk, corr, benchmark_ticker, annual_yield=portfolio_yield)

    # Generate Sector and Industry Analysis
    print("Analyzing sector and industry weighting...")
    sector_analysis = generate_sector_industry_analysis(risk, sector_industry_df, include_yield=include_yield)
    charts["sector_table"] = sector_analysis["table_html"]
    charts["sector_pie"] = sector_analysis["pie_chart_html"]

    # Generate Portfolio Holdings Table
    print("Generating portfolio holdings table...")
    holdings_table_html = generate_portfolio_holdings_analysis(risk, sector_industry_df, prices, portfolio)
    charts["holdings_table"] = holdings_table_html

    # Generate Monte Carlo simulation chart
    mc_chart_html = generate_monte_carlo_chart(mc_simulations)
    charts["monte_carlo"] = mc_chart_html

    # Generate and save the HTML report
    generate_html_report(metrics, charts, report_title, start, include_yield=include_yield)

    print("Analysis complete. Check 'portfolio_report.html' for the detailed report.")

    # Open the generated HTML report in the default web browser for convenience
    report_path = os.path.abspath("portfolio_report.html")
    webbrowser.open(f"file://{report_path}")


# =============================================================================

if __name__ == "__main__":
    # Check for existence of portfolio definition files and create dummy ones if not found
    if not os.path.exists("core_portfolio.xlsx"):
        pd.DataFrame({
            "ticker":["SPY","TLT","GLD"],
            "quantity":[0.5, 0.3, 0.2] # Now using percentages to test new logic
        }).to_excel("core_portfolio.xlsx", index=False)
        print("Created dummy 'core_portfolio.xlsx' with percentage allocations.")

    if not os.path.exists("active_portfolio.xlsx"):
        pd.DataFrame({
            "ticker":["AAPL","MSFT","NVDA","TSLA"],
            "quantity":[25, 25, 25, 25] # Now using whole-number percentages to test new logic
        }).to_excel("active_portfolio.xlsx", index=False)
        print("Created dummy 'active_portfolio.xlsx' with percentage allocations.")

    # Run the main portfolio analysis
    main()