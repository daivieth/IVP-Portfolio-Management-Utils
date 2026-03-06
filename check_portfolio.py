import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from jinja2 import Template
import os
import webbrowser
from datetime import datetime, timedelta

"""
Portfolio Analysis and Reporting Tool

This script performs a comprehensive analysis of a stock portfolio, including:
- Loading portfolio data from Excel files (core/defensive and active components).
- Downloading historical market data using `yfinance`.
- Building portfolio value time series for different layers (defensive, active, total).
- Calculating various performance metrics such as annualized return, volatility, Sharpe ratio, Sortino ratio,
  maximum drawdown, Beta, Alpha, and Information Ratio.
- Estimating Value at Risk (VaR) and Conditional Value at Risk (CVaR).
- Determining individual asset risk contributions.
- Generating interactive financial charts (portfolio value, asset allocation, correlation matrix).
- Compiling all analysis results into a single, interactive HTML report.

The script also includes a `main` function to orchestrate the analysis flow, prompt user inputs,
and automatically open the generated report in a web browser.
"""

# =============================================================================
# LOAD PORTFOLIO
# =============================================================================

def load_portfolio(core_file='core_portfolio.xlsx', active_file='active_portfolio.xlsx'):
    """
    Loads portfolio data from core (defensive) and active Excel files.

    Args:
        core_file (str): Path to the Excel file containing core portfolio holdings.
        active_file (str): Path to the Excel file containing active portfolio holdings.

    Returns:
        pandas.DataFrame: A consolidated DataFrame containing all portfolio positions,
                          with an added 'type' column to distinguish between 'defensive' and 'active' assets.
    """
    # Load the core portfolio data, assign 'defensive' type
    core_df = pd.read_excel(core_file)
    core_df['type'] = 'defensive'

    # Load the active portfolio data, assign 'active' type
    active_df = pd.read_excel(active_file)
    active_df['type'] = 'active'

    # Concatenate both dataframes into a single portfolio dataframe
    portfolio = pd.concat([core_df, active_df], ignore_index=True)

    # Ensure the 'ticker' column is treated as string type to prevent issues with mixed types
    portfolio['ticker'] = portfolio['ticker'].astype(str)

    return portfolio


# =============================================================================
# DOWNLOAD MARKET DATA
# =============================================================================

def download_price_data(tickers, start_date, end_date):
    """
    Downloads historical adjusted close price data for a list of tickers from Yahoo Finance.

    Args:
        tickers (list): A list of stock tickers (strings) to download data for.
        start_date (datetime): The start date for the historical data.
        end_date (datetime): The end date for the historical data.

    Returns:
        pandas.DataFrame: A DataFrame where columns are tickers and the index is the date,
                          containing the adjusted close prices.

    Raises:
        ValueError: If no data is downloaded for the given tickers and date range.
    """
    # Use yfinance to download data for the specified tickers and date range
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,  # Automatically adjust prices for dividends and splits
        progress=False     # Do not display download progress bar
    )

    # Check if the downloaded data is empty, indicating a potential issue with tickers or date range
    if data.empty:
        raise ValueError("No data downloaded for the specified tickers and date range.")

    # yfinance returns a MultiIndex DataFrame if multiple tickers are downloaded, or a Series for a single ticker.
    # Extract 'Close' prices in case of MultiIndex.
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        # If only one ticker, data is already the prices (or a single column DataFrame)
        prices = data.copy() # Use .copy() to avoid SettingWithCopyWarning

    # Drop any rows with NaN values that might result from missing data for certain dates/tickers
    return prices.dropna(axis=0, how='all') # Drop rows where all values are NaN


# =============================================================================
# BUILD PORTFOLIO TIMESERIES
# =============================================================================
def build_portfolio_timeseries(price_data, portfolio_df):
    """
    Constructs time series of portfolio values for different layers (defensive, active, total)
    and individual positions.

    Args:
        price_data (pandas.DataFrame): DataFrame of historical adjusted close prices.
        portfolio_df (pandas.DataFrame): DataFrame containing portfolio holdings with 'ticker' and 'quantity'.

    Returns:
        dict: A dictionary containing pandas Series for 'defensive', 'active', 'total' portfolio values,
              and a DataFrame for 'positions' values over time.
    """
    # Initialize an empty DataFrame to store the value of each position over time
    pos_values = pd.DataFrame(index=price_data.index)

    # Iterate through each row in the portfolio DataFrame to calculate position values
    for _, row in portfolio_df.iterrows():
        ticker = str(row["ticker"]) # Ensure ticker is string for lookup
        quantity = row["quantity"]

        # If the ticker's price data is available, calculate its value over time
        if ticker in price_data.columns:
            pos_values[ticker] = price_data[ticker] * quantity

    # Identify tickers for defensive and active components from the portfolio DataFrame
    # Convert to list so we can safely filter columns that exist in price_data
    core_tickers = portfolio_df.loc[portfolio_df["type"] == "defensive", "ticker"].tolist()
    active_tickers = portfolio_df.loc[portfolio_df["type"] == "active", "ticker"].tolist()

    # Filter to keep only tickers for which we actually have price data (and thus position values)
    core_cols = [t for t in core_tickers if t in pos_values.columns]
    active_cols = [t for t in active_tickers if t in pos_values.columns]

    # Calculate the total value for the defensive and active layers by summing their respective position values
    core_ts = pos_values[core_cols].sum(axis=1)
    active_ts = pos_values[active_cols].sum(axis=1)

    # Calculate the total portfolio value by summing all position values
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
    Calculates daily percentage returns for each time series in the input dictionary.

    Args:
        timeseries_dict (dict): A dictionary where keys are time series identifiers (e.g., 'total', 'defensive')
                                and values are pandas Series or DataFrames representing asset values over time.

    Returns:
        dict: A dictionary with the same keys as `timeseries_dict`, but values are pandas Series or DataFrames
              containing the daily percentage returns.
    """
    returns = {}

    # Iterate through each time series in the dictionary
    for k, ts in timeseries_dict.items():
        # Ensure the item is a pandas Series or DataFrame before calculating returns
        if isinstance(ts, pd.Series) or isinstance(ts, pd.DataFrame):
            # Calculate percentage change and drop the first NaN value
            returns[k] = ts.pct_change().dropna()

    return returns


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

def calculate_performance_metrics(returns_series, benchmark_returns, risk_free_rate=0.02):
    """
    Calculates various financial performance metrics for a given returns series.

    Args:
        returns_series (pandas.Series): A Series of daily percentage returns for the portfolio or asset.
        benchmark_returns (pandas.Series): A Series of daily percentage returns for the benchmark.
        risk_free_rate (float): The annual risk-free rate (default is 0.02 or 2%).

    Returns:
        dict: A dictionary containing calculated performance metrics:
              'Annualized Return', 'Cumulative Return', 'Volatility', 'Sharpe Ratio',
              'Sortino Ratio', 'Max Drawdown', 'Beta', 'Alpha', 'Information Ratio'.
    """
    # Convert annual risk-free rate to daily
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1

    # Cumulative Return: Total return over the period
    cum_ret = (1 + returns_series).prod() - 1
    n_days = len(returns_series) # Number of trading days in the series

    # Annualized Return: Return normalized to a one-year period (assuming 252 trading days)
    ann_ret = (1 + cum_ret) ** (252/n_days) - 1 if n_days > 0 else np.nan

    # Volatility (Annualized Standard Deviation): Measure of risk
    vol = returns_series.std() * np.sqrt(252) if n_days > 1 else np.nan

    # Sharpe Ratio: Risk-adjusted return (excess return per unit of volatility)
    sharpe = ((returns_series.mean() - daily_rf) / returns_series.std()) * np.sqrt(252) if returns_series.std() != 0 else np.nan

    # Downside Deviation: Standard deviation of negative returns only
    downside = returns_series[returns_series < 0]

    # Sortino Ratio: Risk-adjusted return using downside deviation
    sortino = (ann_ret - risk_free_rate) / (downside.std() * np.sqrt(252)) if downside.std() != 0 else np.nan

    # Drawdown Calculation:
    # 1. Cumulative returns
    cumulative = (1 + returns_series).cumprod()
    # 2. Peak value up to each point
    peak = cumulative.cummax()
    # 3. Drawdown: (Current value - Peak value) / Peak value
    drawdown = (cumulative - peak) / peak
    # 4. Maximum Drawdown: Largest percentage drop from a peak to a trough
    max_dd = drawdown.min() if not drawdown.empty else np.nan

    # Identify common dates between portfolio and benchmark returns for relative metrics
    common = returns_series.index.intersection(benchmark_returns.index)

    # Calculate Beta, Alpha, and Information Ratio only if sufficient common data exists (e.g., > 30 days)
    if len(common) > 30:
        y = returns_series.loc[common] # Portfolio returns on common dates
        x = benchmark_returns.loc[common] # Benchmark returns on common dates

        # Perform linear regression to find Beta (slope) and Alpha (intercept adjusted)
        slope, intercept, r, p, stderr = stats.linregress(x, y)
        beta = slope

        # Annualized benchmark return for Alpha calculation
        benchmark_ann = (1 + x.mean())**252 - 1 if len(x) > 0 else np.nan

        # Jensen's Alpha: Measures excess return relative to the return predicted by the CAPM
        alpha = ann_ret - (risk_free_rate + beta * (benchmark_ann - risk_free_rate))

        # Excess returns over benchmark
        excess_returns = returns_series.loc[common] - benchmark_returns.loc[common]

        # Information Ratio: Measures the consistency of outperformance over a benchmark
        information_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() != 0 else np.nan
    else:
        # If not enough common data, set these metrics to NaN
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
    Calculates Value at Risk (VaR) and Conditional Value at Risk (CVaR) for a series of returns.

    Args:
        returns (pandas.Series): A Series of daily percentage returns.
        confidence (float): The confidence level for VaR/CVaR calculation (e.g., 0.95 for 95% confidence).

    Returns:
        dict: A dictionary containing 'VaR' (Value at Risk) and 'CVaR' (Conditional Value at Risk).
    """
    # VaR: The (1-confidence) percentile of the returns distribution.
    # For 95% confidence, we look at the 5th percentile (1-0.95 = 0.05). If returns are sorted ascending, this is the value below which 5% of returns fall.
    var = np.percentile(returns, (1 - confidence) * 100)

    # CVaR: The expected return of the worst (1-confidence)% of cases.
    # This is the average of all returns that are less than or equal to the VaR.
    cvar = returns[returns <= var].mean()

    return {"VaR": var, "CVaR": cvar}


# =============================================================================
# RISK CONTRIBUTION
# =============================================================================

def calculate_risk_contribution(pos_values, total_ts):
    """
    Calculates the absolute and percentage risk contribution of each position to the total portfolio volatility.

    Args:
        pos_values (pandas.DataFrame): DataFrame of individual position values over time.
        total_ts (pandas.Series): Series of total portfolio values over time.

    Returns:
        pandas.DataFrame: A DataFrame containing 'Weight', 'Risk Contribution', and '% Risk Contribution'
                          for each asset in the portfolio.
    """
    # Calculate daily returns for each position
    returns = pos_values.pct_change().dropna()

    # Calculate the current weight of each position in the portfolio
    # Weights are based on the last available value in the time series
    weights = pos_values.iloc[-1] / total_ts.iloc[-1]

    # Calculate the annualized covariance matrix of position returns
    cov = returns.cov() * 252

    # Calculate total portfolio volatility
    # Formula: sqrt(w' * Cov * w), where w is weights vector, Cov is covariance matrix
    port_vol = np.sqrt(weights.T @ cov @ weights)

    # Marginal Contribution to Risk (MCR): How much portfolio volatility changes if position weight changes slightly
    # Formula: (Cov * w) / Port_Vol
    mcr = (cov @ weights) / port_vol

    # Component Contribution to Risk (CCR): Absolute risk contribution of each asset
    # Formula: w * MCR
    ccr = weights * mcr

    # Percentage Contribution to Risk: Each asset's contribution as a percentage of total portfolio volatility
    pct = ccr / port_vol

    # Combine results into a DataFrame
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
    Generates interactive Plotly charts for portfolio value, asset allocation, and correlation matrix.

    Args:
        ts_data (dict): Dictionary containing time series data, including 'total' portfolio value and 'benchmark'.
        risk_contrib (pandas.DataFrame): DataFrame from `calculate_risk_contribution` containing asset weights.
        corr_matrix (pandas.DataFrame): Correlation matrix of asset returns.
        benchmark_ticker (str): The ticker symbol of the benchmark asset.

    Returns:
        dict: A dictionary where keys are chart names ('value', 'allocation', 'corr') and values are
              HTML strings of the generated Plotly charts (full_html=False).
    """
    charts = {}

    # --- Portfolio Value Chart ---
    fig = go.Figure()
    fig.update_layout(height=600, title_text='Portfolio vs. Benchmark Cumulative Value')
    # Add portfolio total value trace
    fig.add_trace(go.Scatter(x=ts_data["total"].index, y=ts_data["total"], name="Portfolio", mode='lines'))
    # Add benchmark value trace if available and not empty
    if not ts_data["benchmark"].empty:
        fig.add_trace(go.Scatter(x=ts_data["benchmark"].index, y=ts_data["benchmark"], name=f"Benchmark ({benchmark_ticker})", mode='lines'))
    charts["value"] = fig.to_html(full_html=False, include_plotlyjs='cdn') # Use 'cdn' for Plotly JS to keep HTML light

    # --- Asset Allocation Pie Chart ---
    weights = risk_contrib["Weight"]
    # Filter out assets with zero or NaN weights for cleaner pie chart
    weights = weights[weights.fillna(0) > 0] # Use fillna(0) to handle NaN before comparison
    pie = px.pie(values=weights, names=weights.index, height=550, title='Current Portfolio Allocation by Weight')
    charts["allocation"] = pie.to_html(full_html=False, include_plotlyjs='cdn')

    # --- Correlation Heatmap ---
    heat = px.imshow(corr_matrix, text_auto=True, aspect="auto", height=550, title='Asset Return Correlation Matrix',
                     color_continuous_scale=px.colors.sequential.Viridis)
    charts["corr"] = heat.to_html(full_html=False, include_plotlyjs='cdn')

    return charts


# =============================================================================
# HTML REPORT
# =============================================================================

def generate_html_report(metrics, charts, report_title):
    """
    Generates a standalone HTML report file incorporating portfolio performance metrics and interactive charts.

    Args:
        metrics (dict): A dictionary containing performance metrics for different portfolio layers.
        charts (dict): A dictionary containing HTML strings of Plotly charts.
        report_title (str): The main title for the HTML report.

    Returns:
        None: The function writes the HTML report to a file named 'portfolio_report.html'.
    """
    # Jinja2 template for the HTML report structure and styling
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
                        <tr>
                            <td>{{ metric }}</td>
                            {% if metric in ["Annualized Return", "Cumulative Return", "Volatility", "Max Drawdown", "VaR", "CVaR"] %}
                                <td>{{ "%.2f%%" | format(value * 100) if value is number else value }}</td>
                            {% else %}
                                <td>{{ "%.4f" | format(value) if value is number else value }}</td>
                            {% endif %}
                        </tr>
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
    Main function to run the portfolio analysis workflow.
    It prompts the user for inputs, downloads data, performs calculations,
    generates charts, and creates an HTML report which is then opened in a browser.
    """
    # Prompt user for the report title
    report_title = input("Enter the portfolio report title: ")
    print("Starting portfolio analysis")

    # Load portfolio holdings from Excel files
    portfolio = load_portfolio()
    # Get a unique list of all tickers in the portfolio
    tickers = portfolio["ticker"].unique().tolist()

    # Prompt user for a benchmark ticker, with a default value
    benchmark_ticker_input = input("Enter the benchmark ticker (default: A200.AX): ")
    benchmark_ticker = benchmark_ticker_input if benchmark_ticker_input else "A200.AX"

    # Ensure benchmark ticker and a fallback (^AXJO for Australian market) are included in the tickers list
    # so their data is downloaded along with portfolio assets.
    if benchmark_ticker not in tickers:
        tickers.append(benchmark_ticker)
    if benchmark_ticker != "^AXJO" and "^AXJO" not in tickers:
        tickers.append("^AXJO")

    # Define the date range for data download (last 5 years)
    end = datetime.now()
    start = end - timedelta(days=5*365) # Approximately 5 years

    # Download historical price data for all identified tickers
    prices = download_price_data(tickers, start, end)

    # Final check for the benchmark ticker in the downloaded data
    # If the primary benchmark is not found, try to use ^AXJO as a fallback.
    if benchmark_ticker not in prices.columns:
        if "^AXJO" in prices.columns:
            print(f"Warning: Primary benchmark {benchmark_ticker} not found, falling back to ^AXJO as benchmark.")
            benchmark_ticker = "^AXJO"
        else:
            # If neither primary nor fallback benchmark is found, raise an error.
            raise ValueError(f"Benchmark ticker {benchmark_ticker} (and fallback ^AXJO) not found in downloaded data. Available columns: {prices.columns.tolist()}")

    # Build time series of portfolio values (defensive, active, total, and individual positions)
    ts = build_portfolio_timeseries(prices, portfolio)

    # Calculate daily returns for each portfolio layer
    returns = calculate_returns(ts)

    # Calculate daily returns for the chosen benchmark
    benchmark = prices[benchmark_ticker].pct_change().dropna()

    # Calculate cumulative benchmark value, normalized to start at the same point as the portfolio total value.
    # This ensures a fair comparison on the charts.
    common_index = ts["total"].index.intersection(benchmark.index)
    if not common_index.empty:
        initial_total_value = ts["total"].loc[common_index[0]] # Portfolio starting value
        cumulative_benchmark_returns = (1 + benchmark.loc[common_index]).cumprod() # Cumulative returns of benchmark
        # Normalize benchmark to start at the same value as the total portfolio for visual comparison
        ts["benchmark"] = cumulative_benchmark_returns * (initial_total_value / cumulative_benchmark_returns.iloc[0])
    else:
        print("Warning: No common dates between portfolio total and benchmark for charting. Benchmark will not be shown.")
        ts["benchmark"] = pd.Series(dtype='float64') # Create an empty series if no common index

    metrics = {}
    # Calculate performance metrics for each portfolio layer (defensive, active, total)
    for layer in ["defensive","active","total"]:
        layer_returns = returns[layer]
        performance_metrics = calculate_performance_metrics(
            layer_returns,
            benchmark
        )
        # Calculate VaR and CVaR for the current layer
        var_cvar = calculate_var_cvar(layer_returns)
        performance_metrics["VaR"] = var_cvar["VaR"]
        performance_metrics["CVaR"] = var_cvar["CVaR"]
        metrics[layer] = performance_metrics

    # Calculate risk contribution of individual positions
    risk = calculate_risk_contribution(ts["positions"], ts["total"])

    # Calculate correlation matrix of position returns
    pos_returns = ts["positions"].pct_change().dropna()
    corr = pos_returns.corr()

    # Generate interactive charts
    charts = generate_charts(ts, risk, corr, benchmark_ticker)

    # Generate the final HTML report
    generate_html_report(metrics, charts, report_title)

    print("Analysis complete.")

    # Open the generated HTML report in the default web browser
    report_path = os.path.abspath("portfolio_report.html")
    webbrowser.open(f"file://{report_path}")


# =============================================================================
# Script Entry Point
# =============================================================================

if __name__ == "__main__":
    # This block ensures that sample portfolio files exist if they are not already present.
    # This makes the script runnable out-of-the-box for first-time users.

    # Create a dummy core_portfolio.xlsx if it doesn't exist
    if not os.path.exists("core_portfolio.xlsx"):
        print("Creating sample core_portfolio.xlsx...")
        pd.DataFrame({
            "ticker":["SPY","TLT","GLD"],
            "quantity":[100,80,50]
        }).to_excel("core_portfolio.xlsx", index=False)

    # Create a dummy active_portfolio.xlsx if it doesn't exist
    if not os.path.exists("active_portfolio.xlsx"):
        print("Creating sample active_portfolio.xlsx...")
        pd.DataFrame({
            "ticker":["AAPL","MSFT","NVDA","TSLA"],
            "quantity":[10,5,20,15]
        }).to_excel("active_portfolio.xlsx", index=False)

    # Run the main portfolio analysis function
    main()
