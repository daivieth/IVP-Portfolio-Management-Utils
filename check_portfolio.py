import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from jinja2 import Template
import os
from datetime import datetime, timedelta


# =============================================================================
# LOAD PORTFOLIO
# =============================================================================

def load_portfolio(core_file='core_portfolio.xlsx', active_file='active_portfolio.xlsx'):

    core_df = pd.read_excel(core_file)
    core_df['type'] = 'core'

    active_df = pd.read_excel(active_file)
    active_df['type'] = 'active'

    portfolio = pd.concat([core_df, active_df], ignore_index=True)

    portfolio['ticker'] = portfolio['ticker'].astype(str)

    return portfolio


# =============================================================================
# DOWNLOAD MARKET DATA
# =============================================================================

def download_price_data(tickers, start_date, end_date):

    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError("No data downloaded")

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data

    return prices.dropna()


# =============================================================================
# BUILD PORTFOLIO TIMESERIES
# =============================================================================
def build_portfolio_timeseries(price_data, portfolio_df):

    pos_values = pd.DataFrame(index=price_data.index)

    # Build position value time series
    for _, row in portfolio_df.iterrows():

        ticker = row["ticker"]

        if ticker in price_data.columns:
            pos_values[ticker] = price_data[ticker] * row["quantity"]

    # Convert to list so we can safely filter columns
    core_tickers = portfolio_df.loc[portfolio_df["type"] == "core", "ticker"].tolist()
    active_tickers = portfolio_df.loc[portfolio_df["type"] == "active", "ticker"].tolist()

    # Keep only tickers that exist in downloaded price data
    core_cols = [t for t in core_tickers if t in pos_values.columns]
    active_cols = [t for t in active_tickers if t in pos_values.columns]

    # Calculate layer values
    core_ts = pos_values[core_cols].sum(axis=1)
    active_ts = pos_values[active_cols].sum(axis=1)

    total_ts = pos_values.sum(axis=1)

    return {
        "core": core_ts,
        "active": active_ts,
        "total": total_ts,
        "positions": pos_values
    }


# =============================================================================
# RETURNS
# =============================================================================

def calculate_returns(timeseries_dict):

    returns = {}

    for k, ts in timeseries_dict.items():

        if isinstance(ts, pd.Series) or isinstance(ts, pd.DataFrame):
            returns[k] = ts.pct_change().dropna()

    return returns


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

def calculate_performance_metrics(returns_series, benchmark_returns, risk_free_rate=0.02):

    daily_rf = (1 + risk_free_rate) ** (1/252) - 1

    cum_ret = (1 + returns_series).prod() - 1

    n_days = len(returns_series)

    ann_ret = (1 + cum_ret) ** (252/n_days) - 1

    vol = returns_series.std() * np.sqrt(252)

    sharpe = ((returns_series.mean() - daily_rf) / returns_series.std()) * np.sqrt(252)

    downside = returns_series[returns_series < 0]

    sortino = (ann_ret - risk_free_rate) / (downside.std() * np.sqrt(252))

    cumulative = (1 + returns_series).cumprod()

    peak = cumulative.cummax()

    drawdown = (cumulative - peak) / peak

    max_dd = drawdown.min()

    common = returns_series.index.intersection(benchmark_returns.index)

    if len(common) > 30:

        y = returns_series.loc[common]
        x = benchmark_returns.loc[common]

        slope, intercept, r, p, stderr = stats.linregress(x, y)

        beta = slope

        benchmark_ann = (1 + x.mean())**252 - 1

        alpha = ann_ret - (risk_free_rate + beta * (benchmark_ann - risk_free_rate))

    else:

        beta = np.nan
        alpha = np.nan

    return {
        "Annualized Return": ann_ret,
        "Cumulative Return": cum_ret,
        "Volatility": vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_dd,
        "Beta": beta,
        "Alpha": alpha
    }


# =============================================================================
# VAR / CVAR
# =============================================================================

def calculate_var_cvar(returns, confidence=0.95):

    var = np.percentile(returns, (1-confidence)*100)

    cvar = returns[returns <= var].mean()

    return {"VaR": var, "CVaR": cvar}


# =============================================================================
# RISK CONTRIBUTION
# =============================================================================

def calculate_risk_contribution(pos_values, total_ts):

    returns = pos_values.pct_change().dropna()

    weights = pos_values.iloc[-1] / total_ts.iloc[-1]

    cov = returns.cov() * 252

    port_vol = np.sqrt(weights.T @ cov @ weights)

    mcr = cov @ weights / port_vol

    ccr = weights * mcr

    pct = ccr / port_vol

    return pd.DataFrame({
        "Weight": weights,
        "Risk Contribution": ccr,
        "% Risk Contribution": pct
    })


# =============================================================================
# CHARTS
# =============================================================================

def generate_charts(ts_data, risk_contrib, corr_matrix):

    charts = {}

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=ts_data["total"].index, y=ts_data["total"], name="Portfolio"))

    if not ts_data["benchmark"].empty:
        fig.add_trace(go.Scatter(x=ts_data["benchmark"].index, y=ts_data["benchmark"], name="Benchmark"))

    charts["value"] = fig.to_html(full_html=False)

    weights = risk_contrib["Weight"]

    pie = px.pie(values=weights, names=weights.index)

    charts["allocation"] = pie.to_html(full_html=False)

    heat = px.imshow(corr_matrix)

    charts["corr"] = heat.to_html(full_html=False)

    return charts


# =============================================================================
# HTML REPORT
# =============================================================================

def generate_html_report(metrics, charts):

    template = Template("""

    <html>

    <head>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
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
    </style>
    <title>Portfolio Report</title>
    </head>

    <body>

    <h1>Portfolio Analytics</h1>

    <h2>Metrics</h2>
    <div class="chart-container">
        {% for layer, data in metrics.items() %}
        <h3>{{ layer | title }} Portfolio Metrics</h3>
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
                    <td>{{ "%.4f" | format(value) if value is number else value }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endfor %}
    </div>

    <h2>Portfolio Value</h2>
    <div class="chart-container">
        {{ charts.value | safe }}
    </div>

    <h2>Allocation</h2>
    <div class="chart-container">
        {{ charts.allocation | safe }}
    </div>

    <h2>Correlation</h2>
    <div class="chart-container">
        {{ charts.corr | safe }}
    </div>

    </body>

    </html>

    """)

    html = template.render(metrics=metrics, charts=charts)

    with open("portfolio_report.html","w") as f:
        f.write(html)

    print("Report generated: portfolio_report.html")


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("Starting portfolio analysis")

    portfolio = load_portfolio()

    tickers = portfolio["ticker"].unique().tolist()

    if "A200.AX" not in tickers:
        tickers.append("A200.AX")
    if "^AXJO" not in tickers:
        tickers.append("^AXJO")

    end = datetime.now()

    start = end - timedelta(days=5*365)

    prices = download_price_data(tickers, start, end)

    # Ensure benchmark ticker is in the downloaded prices
    benchmark_ticker = "A200.AX"
    if benchmark_ticker not in prices.columns:
        # Fallback to ^AXJO if A200.AX is not found
        if "^AXJO" in prices.columns:
            print(f"Warning: {benchmark_ticker} not found, falling back to ^AXJO as benchmark.")
            benchmark_ticker = "^AXJO"
        else:
            raise ValueError(f"Benchmark ticker {benchmark_ticker} (and fallback ^AXJO) not found in downloaded data. Available columns: {prices.columns.tolist()}")

    ts = build_portfolio_timeseries(prices, portfolio)

    returns = calculate_returns(ts)

    benchmark = prices[benchmark_ticker].pct_change().dropna()

    # Calculate cumulative benchmark value, normalized to start at the same point as the portfolio
    # Ensure benchmark and total portfolio start at the same date for comparison
    common_index = ts["total"].index.intersection(benchmark.index)
    if not common_index.empty:
        initial_total_value = ts["total"].loc[common_index[0]]
        cumulative_benchmark_returns = (1 + benchmark.loc[common_index]).cumprod()
        # Normalize benchmark to start at the same value as the total portfolio
        ts["benchmark"] = cumulative_benchmark_returns * (initial_total_value / cumulative_benchmark_returns.iloc[0])
    else:
        print("Warning: No common dates between portfolio total and benchmark for charting.")
        ts["benchmark"] = pd.Series(dtype='float64') # Empty series if no common index

    metrics = {}

    for layer in ["core","active","total"]:

        metrics[layer] = calculate_performance_metrics(
            returns[layer],
            benchmark
        )

    risk = calculate_risk_contribution(ts["positions"], ts["total"])

    pos_returns = ts["positions"].pct_change().dropna()

    corr = pos_returns.corr()

    charts = generate_charts(ts, risk, corr)

    generate_html_report(metrics, charts)

    print("Analysis complete")


# =============================================================================

if __name__ == "__main__":

    if not os.path.exists("core_portfolio.xlsx"):

        pd.DataFrame({
            "ticker":["SPY","TLT","GLD"],
            "quantity":[100,80,50]
        }).to_excel("core_portfolio.xlsx", index=False)

    if not os.path.exists("active_portfolio.xlsx"):

        pd.DataFrame({
            "ticker":["AAPL","MSFT","NVDA","TSLA"],
            "quantity":[10,5,20,15]
        }).to_excel("active_portfolio.xlsx", index=False)

    main()