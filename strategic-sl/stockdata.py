import os
import requests
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
