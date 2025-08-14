import os
from datetime import datetime
from typing import List, Dict
import requests
from zoneinfo import ZoneInfo


class StockData:
    """Fetches stock price data from the Alpha Vantage API."""

    BASE_URL = "https://api.stockdata.org/v1/data/"

    def __init__(self):
        """
        Initialize StockData with API key from environment variable.

        Raises:
            ValueError: If STOCKDATA_API_KEY environment variable is not set
        """
        self.api_key = os.environ.get('STOCKDATA_API_KEY')
        if not self.api_key:
            raise ValueError("STOCKDATA_API_KEY environment variable not set")
        self.base_url = self.BASE_URL

    def string_to_timestamp(self, date: str, format: str = '%Y-%m-%d') -> int:
        """
        Convert a date string to a Unix timestamp.

        Args:
            date: Date string in specified format
            format: Date format (default: '%Y-%m-%d')

        Returns:
            Integer Unix timestamp

        Raises:
            ValueError: If date string format is invalid
        """
        try:
            return int(
                datetime.strptime(date, format)
                .replace(tzinfo=ZoneInfo('US/Eastern'))
                .timestamp()
            )
        except ValueError as e:
            raise ValueError(f"Invalid date format: {date}. Expected format: {format}") from e

    def timestamp_to_string(self, timestamp: float, format: str = '%Y-%m-%d') -> str:
        """
        Convert a Unix timestamp to a date string.

        Args:
            timestamp: Unix timestamp
            format: Desired output format (default: '%Y-%m-%d')

        Returns:
            Formatted date string

        Raises:
            ValueError: If timestamp is invalid
        """
        try:
            return datetime.fromtimestamp(timestamp).strftime(format)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid timestamp: {timestamp}") from e

    def hit_api(self, url: str) -> List[Dict]:
        """
        Make an API request to fetch data.

        Args:
            url: API endpoint URL

        Returns:
            List of dictionaries containing API response data

        Raises:
            ValueError: If the API request fails or response is invalid
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json().get('data')
            if not data:
                raise ValueError("No data returned from API")
            return data
        except requests.exceptions.HTTPError as http_err:
            raise ValueError(
                f"HTTP error occurred: {http_err}\n"
                f"Status Code: {response.status_code}\n"
                f"Response Text: {response.text}"
            ) from http_err
        except requests.exceptions.RequestException as req_err:
            raise ValueError(f"API request failed: {req_err}") from req_err

    def get_price_data(self, symbol: str, lookback_period: int = 20) -> List[Dict]:
        """
        Fetch historical OHLCV data for a given stock symbol.

        Args:
            symbol: Stock ticker symbol
            lookback_period: Number of days to fetch (default: 20)

        Returns:
            List of dictionaries with OHLCV data, sorted by date (newest first)

        Raises:
            ValueError: If symbol is invalid, lookback_period is non-positive, or API request fails
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        if lookback_period <= 0:
            raise ValueError("Lookback period must be positive")

        url = f"{self.base_url}eod?symbols={symbol}&api_token={self.api_key}"
        raw_data = self.hit_api(url)

        # Transform and validate data
        required_keys = {'close', 'low', 'open', 'high', 'volume', 'date'}
        data = []
        for item in raw_data:
            if not all(key in item for key in required_keys):
                raise ValueError(f"API data missing required keys: {required_keys}")
            data.append({
                "close": float(item['close']),
                "low": float(item['low']),
                "open": float(item['open']),
                "high": float(item['high']),
                "volume": int(item['volume']),
                "date": item['date'].split('T')[0]
            })

        # Sort by date in descending order
        data.sort(
            key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'),
            reverse=True
        )

        return data[:lookback_period]
