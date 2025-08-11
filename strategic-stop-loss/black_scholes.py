import os
import requests
import numpy as np
import pandas as pd
import finnhub
from matplotlib import ticker
from scipy.stats import norm
from tabulate import tabulate
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


class StockData:
    """A helper class to fetch data from the Alpha Vantage API."""

    def __init__(self):
        self.api_key = os.environ.get('STOCKDATA_API_KEY')
        if not self.api_key:
            raise ValueError(
                "STOCKDATA_API_KEY environment variable not set.")

        self.base_url = "https://api.stockdata.org/v1/data/"

    def string_to_timestamp(self, date, format='%Y-%m-%d'):
        return int(
            datetime.strptime(date, format)
            .replace(tzinfo=ZoneInfo('US/Eastern'))
            .timestamp()
        )

    def timestamp_to_string(self, timestamp, format='%Y-%m-%d'):
        return datetime.fromtimestamp(timestamp).strftime(format)

    def hit_api(self, url):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()['data']
        except requests.exceptions.HTTPError as http_err:
            raise (f"HTTP error occurred: {http_err}\n",
                   f"Status Code: {response.status_code}\n",
                   f"Response Text: {response.text}"

                   )

    def get_price_data(self, symbol, lookback_period=20):

        base_url = self.base_url + (
            f"eod?symbols={symbol}&api_token"
            f"={self.api_key}"
        )

        _data = self.hit_api(base_url)
        data = [
            {
                "close": i['close'],
                "low": i['low'],
                "open": i['open'],
                "high": i['high'],
                "volume": i['volume'],
                "date": i['date'].split('T')[0]
            } for i in _data
        ]

        data = sorted(
            data,
            key=lambda i: datetime.strptime(i['date'], '%Y-%m-%d'),
            reverse=True
        )

        return data[:lookback_period]


def calculate_nav_erosion(prices:list):
    if len(prices) < 2:
        raise ValueError(
            "At least two prices are required to calculate NAV erosion.")

    # Convert to numpy array for efficiency
    prices = np.array(prices)

    # Calculate total log return over the period
    total_log_return = np.log(prices[-1] / prices[0])

    # Number of trading days in the period
    num_days = len(prices) - 1

    # Annualized compound return rate (negative = erosion)
    if num_days == 0:
        return 0.0
    annualized_rate = np.exp(total_log_return * (252 / num_days)) - 1

    return annualized_rate


def estimate_volatility_from_ohlcv_dicts(ohlcv_list, method='garman_klass'):
    """
    Estimate daily volatility from a list of OHLCV dictionaries.

    Parameters:
    - ohlcv_list: List of dicts with keys 'open', 'high', 'low', 'close' (in chronological order).
    - method: 'garman_klass' (default) or 'parkinson'.

    Returns:
    - daily_volatility: Float (decimal, e.g., 0.0102 for 1.02%).
    """
    if len(ohlcv_list) < 2:
        raise ValueError("At least two OHLCV entries are required.")

    # Extract OHLC arrays
    opens = np.array([d['open'] for d in ohlcv_list])
    highs = np.array([d['high'] for d in ohlcv_list])
    lows = np.array([d['low'] for d in ohlcv_list])
    closes = np.array([d['close'] for d in ohlcv_list])

    if method == 'parkinson':
        hl_log = np.log(highs / lows)
        volatility = np.sqrt((1 / (4 * np.log(2))) * np.mean(hl_log ** 2))
    elif method == 'garman_klass':
        hl_log = np.log(highs / lows)
        co_log = np.log(closes / opens)
        variance = 0.5 * (hl_log ** 2) - (2 * np.log(2) - 1) * (co_log ** 2)
        volatility = np.sqrt(np.mean(variance))
    else:
        raise ValueError("Method must be 'garman_klass' or 'parkinson'.")

    return volatility


def calculate_daily_drift_dicts(ohlcv_list, risk_free_rate_annual=0.0,
                                nav_erosion_annual=-0.44, use_historical=True,
                                use_theoretical=True):
    """
    Calculate daily drift from OHLCV dicts, with optional theoretical adjustment.

    Parameters:
    - ohlcv_list: List of dicts with 'close' key (optional for historical).
    - risk_free_rate_annual: Annual risk-free rate (default: 0.0 as per your preference).
    - nav_erosion_annual: Annual NAV erosion (default: -0.44 for ULTY).
    - use_historical/use_theoretical: Flags to include methods.

    Returns:
    - daily_drift: Float (decimal).
    - results: Dict with breakdown.
    """
    results = {}

    # Historical drift
    historical_drift = None
    if use_historical and ohlcv_list is not None:
        if len(ohlcv_list) < 2:
            raise ValueError("At least two OHLCV entries for historical drift.")
        closes = np.array([d['close'] for d in ohlcv_list])
        log_returns = np.log(closes[1:] / closes[:-1])
        historical_drift = np.mean(log_returns)
        results["historical_drift"] = historical_drift

    # Theoretical drift
    theoretical_drift = None
    if use_theoretical:
        daily_risk_free = risk_free_rate_annual / 252
        daily_nav_erosion = nav_erosion_annual / 252
        theoretical_drift = daily_risk_free + daily_nav_erosion
        results["theoretical_drift"] = theoretical_drift

    # Combine or select
    if use_historical and use_theoretical and historical_drift is not None and theoretical_drift is not None:
        daily_drift = (historical_drift + theoretical_drift) / 2
        results["combined_drift"] = daily_drift
    elif use_historical and historical_drift is not None:
        daily_drift = historical_drift
        results["historical_drift"] = daily_drift
    elif use_theoretical and theoretical_drift is not None:
        daily_drift = theoretical_drift
        results["theoretical_drift"] = daily_drift
    else:
        raise ValueError("No valid drift calculation method or data.")

    return daily_drift, results


def monte_carlo_stop_loss_probability(
        stop_loss,
        time_horizon_days,
        distribution_per_share,
        distribution_frequency_days,
        n_simulations=10000,
        volatility_method='garman_klass',
        intraday_steps=0
):
    """
    Monte Carlo simulation using OHLCV dicts for stop-loss probability.

    Parameters:
    - current_price: Current ETF price (e.g., 6.01).
    - stop_loss: Stop-loss level (e.g., 5.50).
    - ohlcv_list: List of dicts with 'date', 'open', 'high', 'low', 'close', 'volume'.
    - time_horizon_days: Simulation period (e.g., 1, 5, 20).
    - distribution_per_share: Weekly distribution (e.g., 0.10).
    - distribution_frequency_days: Days between distributions (e.g., 5).
    - n_simulations: Number of paths (default: 10000).
    - volatility_method: 'garman_klass' or 'parkinson'.
    - intraday_steps: If >0, simulate intraday for OHLC output (e.g., 390 for a trading day).

    Returns:
    - probability: % chance of hitting stop-loss.
    - stats: Dict with price distribution stats.
    - ohlc_paths: List of simulated OHLC arrays (if intraday_steps > 0).
    """

    stock_helper = StockData()
    historical_data = stock_helper.get_price_data(symbol="ULTY")
    current_price = historical_data[0]['close']

    # Estimate parameters
    daily_volatility = estimate_volatility_from_ohlcv_dicts(historical_data,
                                                            method=volatility_method)

    nav_erosion = calculate_nav_erosion([i['close'] for i in historical_data])

    daily_drift, _ = calculate_daily_drift_dicts(
        historical_data,
        risk_free_rate_annual=0.0,
        nav_erosion_annual=nav_erosion,
        use_historical=True,
        use_theoretical=True
    )

    all_data = list()

    for _stop_loss in stop_loss:
        for _time in time_horizon_days:
            # Time step
            dt = 1 / 252
            hit_stop_loss = np.zeros(n_simulations, dtype=bool)
            final_prices = np.zeros(n_simulations)
            ohlc_paths = [] if intraday_steps > 0 else None

            for i in range(n_simulations):
                price = current_price
                path_ohlc = [] if intraday_steps > 0 else None
                for t in range(_time):
                    if intraday_steps > 0:
                        intraday_prices = [price]
                        for _ in range(intraday_steps - 1):
                            z = np.random.normal(0, 1)
                            intraday_prices.append(intraday_prices[-1] * np.exp(
                                (daily_drift - 0.5 * daily_volatility ** 2) * (
                                            dt / intraday_steps) +
                                daily_volatility * np.sqrt(dt / intraday_steps) * z
                            ))
                        o = intraday_prices[0]
                        h = np.max(intraday_prices)
                        l = np.min(intraday_prices)
                        price = intraday_prices[-1]
                        path_ohlc.append([o, h, l, price])
                    else:
                        z = np.random.normal(0, 1)
                        price *= np.exp(
                            (daily_drift - 0.5 * daily_volatility ** 2) * dt +
                            daily_volatility * np.sqrt(dt) * z)

                    if (t + 1) % distribution_frequency_days == 0:
                        price = max(price - distribution_per_share, 0)

                    if price <= _stop_loss:
                        hit_stop_loss[i] = True
                        break

                final_prices[i] = price

                if ohlc_paths is not None:
                    ohlc_paths.append(np.array(path_ohlc))

            probability = np.mean(hit_stop_loss) * 100

            stats = {
                "stop_loss": _stop_loss,
                "time_horizon": _time,
                "probability": probability,
                "current_price": current_price,
                "mean_price": np.mean(final_prices),
                "median_price": np.median(final_prices),
                "25th_percentile": np.percentile(final_prices, 25),
                "75th_percentile": np.percentile(final_prices, 75),
                "volatility_used": daily_volatility,
                "drift_used": daily_drift
            }

            all_data.append(stats)

    return all_data, ohlc_paths
