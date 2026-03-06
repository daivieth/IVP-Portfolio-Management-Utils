# IVP-Portfolio-Management-Utils

## Project Overview

This project provides a Python-based portfolio analytics system designed to analyze and visualize investment portfolios. It allows users to load core and active portfolio holdings, download historical price data, calculate various performance and risk metrics, generate automated insights, and visualize key aspects of the portfolio through interactive charts. A comprehensive HTML report is generated to summarize the analysis.

## Getting Started

To set up and run this project, follow these steps:

### Prerequisites

Ensure you have Python 3.x installed. The project dependencies are listed in `requirements.txt`.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd IVP-Portfolio-Management-Utils
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  **Prepare your portfolio data:**
    Place your core and active portfolio holdings in Excel files named `core_portfolio.xlsx` and `active_portfolio.xlsx` in the project root directory. These files should contain at least `ticker` and `quantity` columns.
    
    Example `core_portfolio.xlsx`:
    | ticker | quantity |
    |--------|----------|
    | SPY    | 100      |
    | TLT    | 80       |
    | GLD    | 50       |

    Example `active_portfolio.xlsx`:
    | ticker | quantity |
    |--------|----------|
    | AAPL   | 10       |
    | MSFT   | 5        |
    | NVDA   | 20       |
    | TSLA   | 15       |

2.  **Run the analysis script:**
    ```bash
    python check_portfolio.py
    ```
    The script will download historical data, perform the analysis, and generate an HTML report named `portfolio_report.html` in the project root directory.

3.  **View the report:**
    Open `portfolio_report.html` in your web browser to view the detailed portfolio analysis.

## Project Structure

-   [`check_portfolio.py`](check_portfolio.py): The main Python script containing all the logic for portfolio loading, data downloading, metric calculation, insight generation, chart creation, and HTML report generation.
-   [`requirements.txt`](requirements.txt): Lists all Python library dependencies required to run the project.
-   `core_portfolio.xlsx`: (Example/Input) Excel file for core portfolio holdings.
-   `active_portfolio.xlsx`: (Example/Input) Excel file for active portfolio holdings.
-   `portfolio_report.html`: (Output) The generated HTML report with detailed analytics and visualizations.

## Key Features

-   **Flexible Portfolio Loading:** Supports loading core and active portfolio components from Excel files.
-   **Historical Data Download:** Utilizes `yfinance` to fetch historical adjusted close prices for specified tickers.
-   **Comprehensive Performance Metrics:** Calculates annualized returns, cumulative returns, volatility, Sharpe Ratio, Sortino Ratio, Max Drawdown, Calmar Ratio, Beta, Alpha, Tracking Error, Information Ratio, and Hit Ratio.
-   **Risk Attribution:** Determines the contribution of individual positions to overall portfolio risk.
-   **Diversification Analysis:** Provides Diversification Ratio and Effective Number of Bets.
-   **Automated Insights:** Generates qualitative insights based on quantitative metrics.
-   **Interactive Visualizations:** Creates dynamic charts using Plotly for portfolio value evolution, drawdown, asset allocation, risk contribution, and correlation matrix.
-   **Professional HTML Report:** Consolidates all analysis, charts, and insights into a single, easy-to-read HTML report.

## Dependencies

The project relies on the following Python libraries:

-   `pandas`: For data manipulation and analysis.
-   `numpy`: For numerical operations.
-   `yfinance`: For downloading financial market data.
-   `scipy`: For scientific computing, particularly statistical functions.
-   `plotly`: For generating interactive charts and visualizations.
-   `jinja2`: For templating the HTML reports.

These can be installed using `pip` as described in the [Installation](#installation) section.
