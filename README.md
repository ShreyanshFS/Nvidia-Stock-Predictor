<div align="center">

# NVIDIA Stock Predictor

*A machine-learning experiment for estimating NVDA's approximately three-month return.*

[![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-FF6F00?logo=scikit-learn&logoColor=white)](#)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![Last commit](https://img.shields.io/github/last-commit/ShreyanshFS/NvidiaStockPredict)](https://github.com/ShreyanshFS/NvidiaStockPredict/commits/main)
[![Repository size](https://img.shields.io/github/repo-size/ShreyanshFS/NvidiaStockPredict)](https://github.com/ShreyanshFS/NvidiaStockPredict)

</div>

> [!WARNING]
> This is an educational forecasting project, not investment advice. Model forecasts are uncertain and should not be used as the sole basis for trading or investment decisions.

## Overview

The project downloads daily adjusted data for NVIDIA (`NVDA`) and the S&P 500 ETF (`SPY`), derives technical and relative-performance features, compares regression models, and optionally displays diagnostic charts.

## What it does

`main.py` performs the complete workflow:

1. Gets up to 10 years of daily, adjusted NVDA and SPY market data from Alpaca's IEX feed. Downloaded data is cached as CSV files in `outputs/`.
2. Builds features from price, volume, momentum, volatility, drawdown, RSI, moving averages, MACD, and NVDA performance relative to SPY.
3. Uses the return from each date to the date 63 business days later as the prediction target.
4. Splits the labeled observations chronologically: the earlier 80% is used for training and the most recent 20% for testing. It leaves a 63-row gap at the training boundary.
5. Evaluates a no-change baseline, a scaled linear regression model, and histogram gradient boosting with mean absolute error (MAE), reported in percentage points.
6. Refits the learned models on all labeled data and prints a forecast based on the latest available feature row.
7. If requested, displays charts for predicted versus actual prices, test MAE, and the forecast prices.

## Requirements

- Python 3
- An Alpaca Market Data API key and secret, required only when no cached CSV data is available
- Python packages: `alpaca-py`, `numpy`, `pandas`, `python-dotenv`, `scikit-learn`, and `matplotlib`

Install the dependencies in your active virtual environment:

```powershell
python -m pip install alpaca-py numpy pandas python-dotenv scikit-learn matplotlib
```

## Setup

Create a `.env` file in the project root containing your Alpaca credentials:

```dotenv
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
```

Do not commit this file. It is already ignored by Git.

The script first checks for these cache files:

- `outputs/NVDA_bars.csv`
- `outputs/SPY_bars.csv`

If both are present, it uses them and does not call Alpaca. To download fresh data, delete the relevant cache file and run the script again. The `outputs/` directory is ignored by Git.

## Run

```powershell
python main.py
```

The program prints the dataset summary, the latest three days of feature values, test-set MAE, a roughly three-month forecast, and the most recent three gradient-boosting return predictions. At the prompt, enter `y` to show the Matplotlib chart window or `n` to finish without charts.

> [!TIP]
> With the included cache files, you can run the project without API credentials. Credentials are needed only after deleting a cache file or when starting without cached data.

## Feature set

The models use these inputs derived from the daily bars:

- NVDA and SPY returns over 5, 21, and 63 days
- 21-day NVDA-versus-SPY relative return
- 21- and 63-day return volatility
- 63-day drawdown
- 14-day RSI
- Distance from the 50-day moving average
- MACD-style 12/26-day exponential-moving-average difference
- Current volume relative to its 21-day average

The target is the future 63-day NVDA return. The converted price forecast is the latest closing price multiplied by one plus the predicted return.

## Project layout

```text
main.py       # Data acquisition, features, training, evaluation, forecasting, charts
outputs/      # Local market-data cache created/used by the script
.env          # Local Alpaca credentials (not tracked)
```

## Fresh-download caveat

The checked-in cache files let the current script run with existing data. For a fresh data download, there is a cache-path typo in `fetch_bars`: `out/F"{symbol}_bars.csv"` should be `out/f"{symbol}_bars.csv"`. With the current uppercase `F`, deleting a cache file causes a `NameError` before Alpaca data can be fetched. This README documents the existing behavior and does not change the application code.

## Limitations

- The data source is Alpaca's IEX feed, so coverage and values depend on that feed and on the time of download.
- The evaluation is one chronological holdout rather than walk-forward validation.
- The forecast date is estimated with 63 business days, which is not an exact trading-calendar calculation.
- No model persistence, automated tests, command-line configuration, or dependency lockfile is currently included.
