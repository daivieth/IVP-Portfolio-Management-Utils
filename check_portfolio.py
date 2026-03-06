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
        auto_adjust=True,  # Automatically adjust prices for splits and dividends
        progress=False     # Do not display download progress bar
    )

    # Check if any data was downloaded
    if data.empty:
        raise ValueError("No data downloaded for the specified tickers and date range.")

    # Handle cases where yfinance returns a MultiIndex (multiple tickers)
    # or a single-level column index (single ticker).
    if isinstance(data.columns, pd.MultiIndex):
        # Extract 'Close' prices for multiple tickers
        prices = data["Close"]
    else:
        # If only one ticker, 'data' itself contains the prices
        prices = data

    # Remove any rows with NaN values that might result from missing data for some dates
    return prices.dropna()


# =============================================================================
# BUILD PORTFOLIO TIMESERIES
# =============================================================================
def build_portfolio_timeseries(price_data, portfolio_df):
    """
    Constructs time series for portfolio position values, separating them into
    'defensive' and 'active' layers, and calculates the total portfolio value.

    Args:
        price_data (pd.DataFrame): DataFrame of historical closing prices for all tickers,
                                   with dates as index and tickers as columns.
        portfolio_df (pd.DataFrame): DataFrame containing portfolio holdings with 'ticker',
                                     'quantity', and 'type' (e.g., 'defensive', 'active') columns.

    Returns:
        dict: A dictionary containing pandas Series for:
              - 'defensive': Total value of defensive positions over time.
              - 'active': Total value of active positions over time.
              - 'total': Total portfolio value over time.
              - 'positions': DataFrame of individual position values over time.
    """
    # Initialize a DataFrame to store the value of each position over time
    pos_values = pd.DataFrame(index=price_data.index)

    # Iterate through each holding in the portfolio to calculate its value over time
    for _, row in portfolio_df.iterrows():
        ticker = row["ticker"]
        # Only include tickers for which price data was successfully downloaded
        if ticker in price_data.columns:
            pos_values[ticker] = price_data[ticker] * row["quantity"]

    # Separate tickers into core (defensive) and active lists
    core_tickers = portfolio_df.loc[portfolio_df["type"] == "defensive", "ticker"].tolist()
    active_tickers = portfolio_df.loc[portfolio_df["type"] == "active", "ticker"].tolist()

    # Filter these lists to include only tickers for which position values were calculated
    core_cols = [t for t in core_tickers if t in pos_values.columns]
    active_cols = [t for t in active_tickers if t in pos_values.columns]

    # Calculate the total value time series for each layer and the overall portfolio
    core_ts = pos_values[core_cols].sum(axis=1)
    active_ts = pos_values[active_cols].sum(axis=1)
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

def calculate_performance_metrics(returns_series, benchmark_returns, risk_free_rate=0.02):
    """
    Calculates a suite of performance metrics for a given series of returns,
    including risk-adjusted returns and drawdown metrics.

    Args:
        returns_series (pd.Series): A pandas Series of daily returns for the portfolio or asset.
        benchmark_returns (pd.Series): A pandas Series of daily returns for the benchmark.
        risk_free_rate (float): The annual risk-free rate (default is 0.02, or 2%).

    Returns:
        dict: A dictionary containing the calculated performance metrics:
              - "Annualized Return" (float)
              - "Cumulative Return" (float)
              - "Volatility" (float): Annualized standard deviation of returns.
              - "Sharpe Ratio" (float)
              - "Sortino Ratio" (float)
              - "Max Drawdown" (float)
              - "Beta" (float or NaN): Sensitivity to benchmark, NaN if insufficient common data.
              - "Alpha" (float or NaN): Excess return relative to benchmark, NaN if insufficient common data.
              - "Information Ratio" (float or NaN): Excess return per unit of tracking error, NaN if insufficient common data or zero tracking error.
    """
    # Convert annual risk-free rate to a daily rate
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    # Calculate cumulative return
    cum_ret = (1 + returns_series).prod() - 1
    n_days = len(returns_series)
    # Calculate annualized return
    ann_ret = (1 + cum_ret) ** (252/n_days) - 1
    # Calculate annualized volatility
    vol = returns_series.std() * np.sqrt(252)
    # Calculate Sharpe Ratio
    sharpe = ((returns_series.mean() - daily_rf) / returns_series.std()) * np.sqrt(252)
    # Isolate downside returns for Sortino Ratio calculation
    downside = returns_series[returns_series < 0]
    # Calculate Sortino Ratio
    sortino = (ann_ret - risk_free_rate) / (downside.std() * np.sqrt(252))
    # Calculate cumulative product for drawdown
    cumulative = (1 + returns_series).cumprod()
    # Find the running maximum for drawdown calculation
    peak = cumulative.cummax()
    # Calculate drawdown
    drawdown = (cumulative - peak) / peak
    # Determine the maximum drawdown
    max_dd = drawdown.min()
    # Find common dates between portfolio and benchmark returns for relative metrics
    common = returns_series.index.intersection(benchmark_returns.index)

    # Calculate Beta, Alpha, and Information Ratio only if sufficient common data exists
    if len(common) > 30: # A reasonable threshold for statistical significance
        y = returns_series.loc[common]
        x = benchmark_returns.loc[common]
        # Perform linear regression to get beta (slope)
        slope, intercept, r, p, stderr = stats.linregress(x, y)
        beta = slope
        # Calculate annualized benchmark return
        benchmark_ann = (1 + x.mean())**252 - 1
        # Calculate Alpha (Jensen's Alpha)
        alpha = ann_ret - (risk_free_rate + beta * (benchmark_ann - risk_free_rate))
        # Calculate excess returns relative to the benchmark
        excess_returns = returns_series.loc[common] - benchmark_returns.loc[common]
        # Calculate Information Ratio, handle division by zero for standard deviation
        information_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() != 0 else np.nan
    else:
        # If not enough common data, set benchmark-relative metrics to NaN
        beta = np.nan
        alpha = np.nan
        information_ratio = np.nan

    return {
        "Annualized Return": ann_ret,
        "Cumulative Return": cum_ret,
        "Volatility": vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_dd,
        "Beta": beta,
        "Alpha": alpha,
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

def calculate_risk_contribution(pos_values, total_ts):
    """
    Calculates the risk contribution of each position to the total portfolio volatility.

    Args:
        pos_values (pd.DataFrame): DataFrame of individual position values over time.
        total_ts (pd.Series): Series of total portfolio value over time.

    Returns:
        pd.DataFrame: A DataFrame containing:
                      - "Weight": The current weight of each position in the portfolio.
                      - "Risk Contribution": The contribution of each position to the total portfolio risk.
                      - "% Risk Contribution": The percentage contribution of each position to the total portfolio risk.
    """
    # Calculate daily returns for individual positions and drop any NaN values
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

def generate_charts(ts_data, risk_contrib, corr_matrix, benchmark_ticker):
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

    # Add portfolio performance trace
    performance_total = ((ts_data["total"] / initial_reference_value) - 1) * 100
    fig.add_trace(go.Scatter(x=performance_total.index, y=performance_total, name="Portfolio % Performance", mode='lines'))

    # Add benchmark performance trace if available
    if not ts_data["benchmark"].empty:
        performance_benchmark = ((ts_data["benchmark"] / initial_reference_value) - 1) * 100
        fig.add_trace(go.Scatter(x=performance_benchmark.index, y=performance_benchmark, name=f"Benchmark ({benchmark_ticker}) % Performance", mode='lines'))

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

def generate_html_report(metrics, charts, report_title):
    """
    Generates a comprehensive HTML report summarizing portfolio performance,
    metrics, allocation, and correlations.

    The report uses a Jinja2 template to embed dynamic data and Plotly charts.

    Args:
        metrics (dict): A dictionary containing calculated performance metrics for
                        different portfolio layers (e.g., 'defensive', 'active', 'total').
        charts (dict): A dictionary containing HTML strings of generated Plotly charts.
        report_title (str): The title to display at the top of the HTML report.

    Returns:
        None: The function writes the HTML report to "portfolio_report.html".
    """
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

<h2>Portfolio 5-Year Backtest</h2>
<div class="chart-container">
    {{ charts.value | safe }}
</div>

<div class="side-by-side-container">
    <div class="metrics-column">
        <h2>Metrics</h2>
        <div class="chart-container">
            {% for layer, data in metrics.items() %}
            <h3>{{ layer | title }} Component Metrics</h3>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    {% for metric, value in data.items() %}
                    {% if not (layer == 'total' and metric in ['MC_Expected_Drawdown_Pct', 'MC_VaR_99_Pct', 'MC_VaR_95_Pct', 'MC_Expected_Upside_95_Pct']) %}
                    <tr>
                        <td>{{ metric }}</td>
                        {% if metric in ["Annualized Return", "Cumulative Return", "Volatility", "Max Drawdown", "VaR", "CVaR"] %}
                            <td>{{ "%.2f%%" | format(value * 100) if value is number else value }}</td>
                        {% else %}
                            <td>{{ "%.4f" | format(value) if value is number else value }}</td>
                        {% endif %}
                    </tr>
                    {% endif %}
                    {% endfor %}
                </tbody>
            </table>
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
    html = template.render(metrics=metrics, charts=charts, report_title=report_title)

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
    print("Starting portfolio analysis...")

    # Load portfolio data from Excel files
    portfolio = load_portfolio()
    # Get unique tickers from the portfolio for data download
    tickers = portfolio["ticker"].unique().tolist()

    # Prompt user for benchmark ticker, with a default value
    benchmark_ticker_input = input("Enter the benchmark ticker (default: A200.AX): ")
    benchmark_ticker = benchmark_ticker_input if benchmark_ticker_input else "A200.AX"

    # Ensure benchmark ticker and a fallback ticker (^AXJO) are included in the tickers list for download
    if benchmark_ticker not in tickers:
        tickers.append(benchmark_ticker)
    if "^AXJO" not in tickers: # Always add ^AXJO as a fallback
        tickers.append("^AXJO")

    # Define the date range for historical data download (last 5 years)
    end = datetime.now()
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
    ts = build_portfolio_timeseries(prices, portfolio)
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

    # Calculate performance metrics for defensive, active, and total portfolio layers
    for layer in ["defensive","active","total"]:
        layer_returns = returns[layer]
        performance_metrics = calculate_performance_metrics(
            layer_returns,
            benchmark
        )
        var_cvar = calculate_var_cvar(layer_returns)
        performance_metrics["VaR"] = var_cvar["VaR"]
        performance_metrics["CVaR"] = var_cvar["CVaR"]
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

    # Calculate risk contribution for individual positions
    risk = calculate_risk_contribution(ts["positions"], ts["total"])

    # Calculate returns for individual positions to compute correlation matrix
    pos_returns = ts["positions"].pct_change().dropna()
    corr = pos_returns.corr()
    # Set diagonal to NaN to be handled separately for coloring as gray
    # Set diagonal to NaN to be handled separately for coloring as gray, using pandas mask for writability
    # Set diagonal to a sentinel value (-2.0) to be colored gray
    corr = corr.mask(np.eye(len(corr), dtype=bool), -2.0)

    # Generate interactive charts
    charts = generate_charts(ts, risk, corr, benchmark_ticker)

    # Generate Monte Carlo simulation chart
    mc_chart_html = generate_monte_carlo_chart(mc_simulations)
    charts["monte_carlo"] = mc_chart_html

    # Generate and save the HTML report
    generate_html_report(metrics, charts, report_title)

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
            "quantity":[100,80,50]
        }).to_excel("core_portfolio.xlsx", index=False)
        print("Created dummy 'core_portfolio.xlsx'.")

    if not os.path.exists("active_portfolio.xlsx"):
        pd.DataFrame({
            "ticker":["AAPL","MSFT","NVDA","TSLA"],
            "quantity":[10,5,20,15]
        }).to_excel("active_portfolio.xlsx", index=False)
        print("Created dummy 'active_portfolio.xlsx'.")

    # Run the main portfolio analysis
    main()