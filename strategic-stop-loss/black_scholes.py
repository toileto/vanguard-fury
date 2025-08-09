import os
import requests
import numpy as np
import pandas as pd
from scipy.stats import norm
from tabulate import tabulate


class AVHelper:
    """A helper class to fetch data from the Alpha Vantage API."""

    def __init__(self):
        self.api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            raise ValueError(
                "ALPHA_VANTAGE_API_KEY environment variable not set.")
        self.base_url = 'https://www.alphavantage.co/query'

    def get_price_data(self, ticker, lookback_period=20):
        """
        Fetches daily price data for a given stock ticker.

        Args:
            ticker (str): The stock symbol (e.g., 'ULTY').
            lookback_period (int): The number of recent trading days to fetch.

        Returns:
            list: A list of dictionaries containing daily price data, or None on failure.
        """
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "apikey": self.api_key,
            "outputsize": "compact"
        }
        try:
            r = requests.get(self.base_url, params=params, timeout=10)
            r.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            data = r.json()
            if "Time Series (Daily)" not in data:
                print(
                    f"Error: Could not find time series data for {ticker}. API response: {data}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Network or API error occurred: {e}")
            return None

        time_series = data.get('Time Series (Daily)', {})

        # Convert dictionary to list and slice to the lookback period
        result = [
            {'date': date, 'close': float(values['4. close'])}
            for date, values in time_series.items()
        ]
        return result[:lookback_period]


def calculate_price_probability(prices: list, ticker: str, lookback_days: int,
                                horizon_days: int):
    """
    Calculates the probability of a stock's price falling below given stop-loss levels.

    Args:
        prices (list): A list of stop-loss price points to evaluate.
        ticker (str): The stock ticker symbol.
        lookback_days (int): How many past days of data to use for volatility calculation.
        horizon_days (int): The future time frame (in days) for the probability calculation.
    """
    alphav = AVHelper()
    data = alphav.get_price_data(ticker, lookback_period=lookback_days)

    if not data or len(data) < 2:
        print(
            "Could not retrieve sufficient price data to calculate probability.")
        return

    df = pd.DataFrame(data)

    # --- 1. Calculate Historical Volatility ---
    # Use logarithmic returns for a more accurate volatility measure
    log_returns = np.log(df["close"] / df["close"].shift(
        -1))  # Shift -1 because data is newest first
    volatility = log_returns.std() * np.sqrt(252)  # Annualized volatility

    current_price = df["close"].iloc[0]
    time_in_years = horizon_days / 252.0

    # --- 2. Calculate Probability for Each Price ---
    # This formula, derived from the lognormal distribution of stock prices, calculates
    # the probability of the price being below a certain level. It's related to the
    # 'd2' term in the Black-Scholes options pricing model.
    # We assume a drift of 0 to focus purely on volatility.

    results_data = []
    for stop_price in prices:
        # Calculate d2 from the Black-Scholes model
        d2 = (np.log(current_price / stop_price)) / (
                    volatility * np.sqrt(time_in_years))

        # The CDF of d2 gives the probability of the price being *above* the stop price.
        prob_above = norm.cdf(d2)
        prob_below = 1 - prob_above

        results_data.append({
            "Stop-Loss Price": f"${stop_price:.2f}",
            f"Prob. Above (in {horizon_days} days)": f"{prob_above:.2%}",
            f"Prob. Below (in {horizon_days} days)": f"{prob_below:.2%}"
        })

    # --- 3. Display Results ---
    print(f"Current {ticker} Price: ${current_price:,.2f} (from latest data)")
    print(
        f"Annualized Volatility: {volatility:.2%} (from last {len(data)} days)")
    print(tabulate(results_data, headers="keys", tablefmt="fancy_grid"))
