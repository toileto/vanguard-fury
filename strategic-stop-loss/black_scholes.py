from zoneinfo import ZoneInfo
from scipy.stats import norm
from tabulate import tabulate
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import os
import requests


def unix_timestamp_from_date(date, format='%Y-%m-%d'):
    '''Transform human readable date string to UNIX timestamp'''
    return int(
        datetime.strptime(date, format)
        .replace(tzinfo=ZoneInfo('US/Eastern'))
        .timestamp()
    )


class AVHelper:
    def __init__(self):
        self.api_key = os.environ['ALPHA_VANTAGE_API_KEY']
        pass

    def get_price_data(self, ticker, function="TIME_SERIES_DAILY",
                       lookback_period=20):
        url = (f'https://www.alphavantage.co/query?function={function}'
               f'&symbol={ticker}&apikey={self.api_key}')
        r = requests.get(url)
        data = r.json().get('Time Series (Daily)')
        result = []
        i = 0
        for date, values in data.items():
            if i != lookback_period:
                entry = {
                    'date': date,
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': float(values['5. volume'])
                }
                result.append(entry)
                i += 1
            else:
                break

        return result


def calculate_price_probability(stop_loss_prices: list):
    alphav = AVHelper()
    lookback_period = 10
    forward_horizon_days = 5
    # stop_loss_prices = [5.98, 5.81, 5.69, 5.52, 5.35]
    data = alphav.get_price_data(
        "ULTY",
        lookback_period=lookback_period
    )

    df = pd.DataFrame(data)
    
    # Calculate daily logarithmic returns
    log_returns = np.log(df["close"] / df["close"].shift(1))

    # Calculate annualized volatility (sigma) from historical data
    volatility = log_returns.std() * np.sqrt(252)  # 252 trading days in a year

    # Get the current stock price (S)
    current_price = df["close"].iloc[0]

    # Convert time horizon to years for the formula
    time_in_years = forward_horizon_days / 252.0

    # 2. Calculate the probability
    # We use a formula derived from the lognormal distribution properties.
    # It calculates the probability of the price being below the stop_loss_price.
    # Here, we assume the expected return (drift) is zero to focus purely on volatility.

    # This term is similar to 'd2' in the Black-Scholes model
    all_data = list()
    for stop_loss_price in stop_loss_prices:
        d2 = (np.log(
            current_price / stop_loss_price) - 0.5 * volatility ** 2 * time_in_years) / (
                         volatility * np.sqrt(time_in_years))

        # The cumulative distribution function (CDF) of d2 gives us the probability
        # of the price ending up *above* the stop-loss.
        prob_above = norm.cdf(d2)

        # Therefore, 1 minus that is the probability of it being *below* the stop-loss.
        prob_below = 1 - prob_above

        all_data.append({
            "stop_loss_price": f"${stop_loss_price:.2f}",
            "prob_above": f"{prob_above:.2%}",
            "prob_below": f"{prob_below:.2%}"
        })

    # 3. Display the results
    print(f"Current Stock Price [from data]:      ${current_price:,.2f}")
    print(f"Calculated Volatility [from data]:    {volatility * 100:.2f}%")

    print(tabulate(all_data, headers="keys", tablefmt="fancy_grid"))

    print("-" * 42)
