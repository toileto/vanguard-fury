import numpy as np
import pandas as pd
import mplfinance as mpf
import gurobipy as gp
import os
import requests

from gurobipy import GRB
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from tabulate import tabulate


class FinSor:
    """Financial Advisor class for stock data fetching, Monte Carlo simulations, and stop-loss optimization."""

    BASE_URL = "https://api.stockdata.org/v1/data/"

    def __init__(self, ticker: str, api_key: Optional[str] = None):
        """
        Initialize FinSor with a stock ticker and API key.

        Args:
            ticker: Stock ticker symbol
            api_key: Alpha Vantage API key (defaults to STOCKDATA_API_KEY environment variable)

        Raises:
            ValueError: If ticker is invalid or API key is not provided
        """
        if not ticker or not isinstance(ticker, str):
            raise ValueError("Ticker must be a non-empty string")
        self.ticker = ticker
        self.api_key = api_key or os.environ.get('STOCKDATA_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided or set as STOCKDATA_API_KEY environment variable")
        self.historical_data = self.stockdata_get_price_data(ticker)

    # --- Stock Data Methods ---
    def stockdata_string_to_timestamp(self, date: str, format: str = '%Y-%m-%d') -> int:
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

    def stockdata_timestamp_to_string(self, timestamp: float, format: str = '%Y-%m-%d') -> str:
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

    def stockdata_hit_api(self, url: str) -> List[Dict]:
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

    def stockdata_get_price_data(self, symbol: str, lookback_period: int = 20) -> List[Dict]:
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

        url = f"{self.BASE_URL}eod?symbols={symbol}&api_token={self.api_key}"
        raw_data = self.stockdata_hit_api(url)

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

        data.sort(
            key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'),
            reverse=True
        )
        return data[:lookback_period]

    # --- Monte Carlo Methods ---
    def montecarlo_dict_to_list(self, data: List[Dict], key: str = "close") -> List[float]:
        """
        Extract values for a specific key from a list of dictionaries.

        Args:
            data: List of dictionaries containing price data
            key: Key to extract from each dictionary (default: 'close')

        Returns:
            List of values for the specified key

        Raises:
            KeyError: If the specified key is not found in dictionaries
        """
        try:
            return [item[key] for item in data]
        except KeyError as e:
            raise KeyError(f"Key '{key}' not found in data dictionaries") from e

    def montecarlo_calculate_nav_erosion(self, prices: List[float]) -> float:
        """
        Calculate annualized NAV erosion from a list of prices.

        Args:
            prices: List of prices in chronological order

        Returns:
            Annualized NAV erosion rate (decimal)

        Raises:
            ValueError: If fewer than two prices are provided
        """
        if len(prices) < 2:
            raise ValueError("At least two prices are required to calculate NAV erosion")

        prices = np.array(prices)
        total_log_return = np.log(prices[-1] / prices[0])
        num_days = len(prices) - 1
        return np.exp(total_log_return * (252 / num_days)) - 1 if num_days > 0 else 0.0

    def montecarlo_estimate_volatility(self, price_data: List[Dict], method: str = 'garman_klass') -> float:
        """
        Estimate daily volatility from OHLCV data.

        Args:
            price_data: List of dictionaries with 'open', 'high', 'low', 'close' keys
            method: Volatility estimation method ('garman_klass' or 'parkinson', default: 'garman_klass')

        Returns:
            Daily volatility (decimal)

        Raises:
            ValueError: If fewer than two entries or invalid method
            KeyError: If required OHLC keys are missing
        """
        if len(price_data) < 2:
            raise ValueError("At least two OHLCV entries are required")

        required_keys = {'open', 'high', 'low', 'close'}
        if not all(key in price_data[0] for key in required_keys):
            raise KeyError(f"OHLCV data must contain keys: {required_keys}")

        opens = np.array([d['open'] for d in price_data])
        highs = np.array([d['high'] for d in price_data])
        lows = np.array([d['low'] for d in price_data])
        closes = np.array([d['close'] for d in price_data])

        if method == 'parkinson':
            hl_log = np.log(highs / lows)
            volatility = np.sqrt(np.mean(hl_log ** 2) / (4 * np.log(2)))
        elif method == 'garman_klass':
            hl_log = np.log(highs / lows)
            co_log = np.log(closes / opens)
            variance = 0.5 * (hl_log ** 2) - (2 * np.log(2) - 1) * (co_log ** 2)
            volatility = np.sqrt(np.mean(variance))
        else:
            raise ValueError("Method must be 'garman_klass' or 'parkinson'")

        return volatility

    def montecarlo_calculate_daily_drift(self, price_data: List[Dict], risk_free_rate_annual: float = 0.0,
                                        use_historical: bool = True, use_theoretical: bool = True) -> Tuple[float, Dict]:
        """
        Calculate daily drift from OHLCV data.

        Args:
            price_data: List of dictionaries with 'close' key
            risk_free_rate_annual: Annual risk-free rate (default: 0.0)
            use_historical: Include historical drift calculation (default: True)
            use_theoretical: Include theoretical drift calculation (default: True)

        Returns:
            Tuple containing:
            - Daily drift (decimal)
            - Dictionary with calculation breakdown

        Raises:
            ValueError: If no valid drift calculation method or insufficient data
            KeyError: If 'close' key is missing
        """
        if not price_data:
            raise ValueError("Price data cannot be empty")
        if 'close' not in price_data[0]:
            raise KeyError("OHLCV data must contain 'close' key")

        results = {}
        daily_drift = None

        if use_historical:
            if len(price_data) < 2:
                raise ValueError("At least two OHLCV entries required for historical drift")
            closes = np.array([d['close'] for d in price_data])
            log_returns = np.log(closes[1:] / closes[:-1])
            historical_drift = np.mean(log_returns)
            results["historical_drift"] = historical_drift
            daily_drift = historical_drift

        if use_theoretical:
            daily_risk_free = risk_free_rate_annual / 252
            daily_nav_erosion = self.montecarlo_calculate_nav_erosion(self.montecarlo_dict_to_list(price_data)) / 252
            theoretical_drift = daily_risk_free + daily_nav_erosion
            results["theoretical_drift"] = theoretical_drift
            daily_drift = theoretical_drift

        if use_historical and use_theoretical and historical_drift is not None and theoretical_drift is not None:
            daily_drift = (historical_drift + theoretical_drift) / 2
            results["combined_drift"] = daily_drift

        if daily_drift is None:
            raise ValueError("No valid drift calculation method or data available")

        return daily_drift, results

    def montecarlo_analyse_stop_loss_probability(
            self,
            price: List[Dict],
            stop_loss: List[float],
            time_horizon_days: List[int],
            distribution_per_share: float,
            distribution_frequency_days: int,
            n_simulations: int = 10000,
            volatility_method: str = 'garman_klass',
            intraday_steps: int = 0,
            result_mode: str = "compact"
    ) -> List[List[Dict], List[Dict], Optional[List[np.ndarray]]]:
        """
        Perform Monte Carlo simulation to estimate stop-loss trigger probabilities.

        Args:
            price: List of dictionaries with OHLCV data
            stop_loss: List of stop-loss prices
            time_horizon_days: List of time horizons in days (e.g., [5, 10, 20])
            distribution_per_share: Distribution amount per share
            distribution_frequency_days: Days between distributions
            n_simulations: Number of simulation paths (default: 10000)
            volatility_method: Volatility estimation method ('garman_klass' or 'parkinson', default: 'garman_klass')
            intraday_steps: Number of intraday steps for OHLC simulation (default: 0)
            result_mode: Output format ('compact' or 'detailed', default: 'compact')

        Returns:
            List containing:
            - List of compact probability statistics
            - List of detailed probability statistics
            - List of simulated OHLC paths (if intraday_steps > 0, else None)

        Raises:
            ValueError: If inputs are invalid or insufficient data
            KeyError: If required OHLC keys are missing
        """
        if not price or 'close' not in price[0]:
            raise KeyError("Price data must contain 'close' key")
        if not stop_loss or not time_horizon_days:
            raise ValueError("Stop-loss prices and time horizons cannot be empty")
        if distribution_per_share < 0 or distribution_frequency_days <= 0:
            raise ValueError("Distribution parameters must be positive")
        if n_simulations <= 0:
            raise ValueError("Number of simulations must be positive")

        current_price = price[0]['close']
        daily_volatility = self.montecarlo_estimate_volatility(price, method=volatility_method)
        daily_drift, _ = self.montecarlo_calculate_daily_drift(price, use_historical=True, use_theoretical=True)

        all_data = []
        compact_stats = []

        for stop_price in stop_loss:
            compact_stat = {"stop_loss": stop_price, "current_price": current_price}

            for time_horizon in time_horizon_days:
                hit_stop_loss = np.zeros(n_simulations, dtype=bool)
                final_prices = np.zeros(n_simulations)
                ohlc_paths = [] if intraday_steps > 0 else None

                for i in range(n_simulations):
                    price = current_price
                    path_ohlc = [] if intraday_steps > 0 else None

                    for t in range(time_horizon):
                        if intraday_steps > 0:
                            intraday_prices = np.zeros(intraday_steps)
                            intraday_prices[0] = price
                            for j in range(1, intraday_steps):
                                z = np.random.normal(0, 1)
                                intraday_prices[j] = intraday_prices[j-1] * np.exp(
                                    (daily_drift - 0.5 * daily_volatility ** 2) +
                                    daily_volatility * z
                                )
                            path_ohlc.append([
                                intraday_prices[0],
                                np.max(intraday_prices),
                                np.min(intraday_prices),
                                intraday_prices[-1]
                            ])
                            price = intraday_prices[-1]
                        else:
                            z = np.random.normal(0, 1)
                            price *= np.exp(
                                (daily_drift - 0.5 * daily_volatility ** 2) +
                                daily_volatility * np.sqrt(1) * z
                            )

                        if (t + 1) % distribution_frequency_days == 0:
                            price = max(price - distribution_per_share, 0)

                        if price <= stop_price:
                            hit_stop_loss[i] = True
                            break

                    final_prices[i] = price
                    if path_ohlc:
                        ohlc_paths.append(np.array(path_ohlc))

                probability = np.mean(hit_stop_loss) * 100
                compact_stat[f"hit_in_{time_horizon}_days"] = probability

                stats = {
                    "stop_loss": stop_price,
                    "time_horizon": time_horizon,
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

            compact_stats.append(compact_stat)

        return [compact_stats, all_data, ohlc_paths]

    def montecarlo_plot_ohlc_paths(self, ohlc_paths: List[np.ndarray], stop_loss: List[float], time_horizon: int,
                                   num_paths: int = 5, title: str = "Simulated OHLC Paths") -> None:
        """
        Plot candlestick charts for simulated OHLC paths with stop-loss levels.

        Args:
            ohlc_paths: List of NumPy arrays with OHLC data [open, high, low, close]
            stop_loss: List of stop-loss prices to plot as horizontal lines
            time_horizon: Number of days in the simulation for x-axis
            num_paths: Number of paths to plot (default: 5)
            title: Plot title (default: 'Simulated OHLC Paths')

        Raises:
            ValueError: If ohlc_paths is empty, num_paths is invalid, or OHLC data is malformed
        """
        if not ohlc_paths:
            raise ValueError("OHLC paths cannot be empty")
        if num_paths <= 0 or num_paths > len(ohlc_paths):
            raise ValueError(f"num_paths must be between 1 and {len(ohlc_paths)}")
        if not all(path.shape[1] == 4 for path in ohlc_paths):
            raise ValueError("Each OHLC path must contain exactly 4 columns (open, high, low, close)")

        paths_to_plot = ohlc_paths[:min(num_paths, len(ohlc_paths))]
        fig, axes = plt.subplots(nrows=len(paths_to_plot), ncols=1, figsize=(12, 4 * len(paths_to_plot)), sharex=True)

        if len(paths_to_plot) == 1:
            axes = [axes]

        start_date = datetime.now()
        dates = [start_date + timedelta(days=i) for i in range(time_horizon)]

        for idx, (path, ax) in enumerate(zip(paths_to_plot, axes)):
            df = pd.DataFrame({
                'Date': dates[:len(path)],
                'Open': path[:, 0],
                'High': path[:, 1],
                'Low': path[:, 2],
                'Close': path[:, 3]
            })
            df.set_index('Date', inplace=True)

            mpf.plot(
                df,
                type='candle',
                ax=ax,
                style='yahoo',
                title=f"Path {idx + 1}" if len(paths_to_plot) > 1 else title,
                ylabel='Price' if idx == 0 else '',
                show_nontrading=False
            )

            for sl in stop_loss:
                ax.axhline(y=sl, color='blue', linestyle='--', alpha=0.7, label=f'Stop Loss: ${sl:.2f}')

            if idx == 0:
                ax.legend()

        fig.suptitle(title if len(paths_to_plot) > 1 else '', y=0.98)
        plt.xlabel('Date')
        plt.tight_layout()
        plt.show()

    # --- Stop-Loss Optimization Methods ---
    def stoploss_calculate_solution_probability(self, layers: List[str], model: gp.Model) -> Tuple[List[Dict], List]:
        """
        Calculate probabilities for stop-loss triggers using Monte Carlo simulation.

        Args:
            layers: List of layer names (e.g., ['A', 'B', 'C', 'D'])
            model: Solved Gurobi model

        Returns:
            Tuple containing:
            - List of probability statistics
            - List of OHLC paths from Monte Carlo simulation

        Raises:
            ValueError: If model is not solved or layers are invalid
        """
        if model.Status != GRB.OPTIMAL:
            raise ValueError("Model must be solved optimally before probability calculation")

        share_vars = model.getVars()[:len(layers)]
        price_vars = model.getVars()[len(layers):]
        solution_data = {
            f"Layer {layer}": {"shares": round(s_var.X), "price": round(p_var.X, 2)}
            for layer, s_var, p_var in zip(layers, share_vars, price_vars)
        }
        stop_loss_prices = [data["price"] for data in solution_data.values()]

        stats_data, all_data, ohlc_paths = self.montecarlo_analyse_stop_loss_probability(
            price=self.historical_data,
            stop_loss=stop_loss_prices,
            time_horizon_days=[5, 10, 20],
            distribution_per_share=0.10,
            distribution_frequency_days=5
        )
        return stats_data, ohlc_paths

    def stoploss_build_and_solve_model(self, config: Dict) -> gp.Model:
        """
        Build and solve a Gurobi optimization model for stop-loss strategy.

        Args:
            config: Configuration dictionary with parameters:
                - total_shares: Total number of shares
                - avg_price: Average purchase price
                - target_loss_percentage: Target loss as a percentage (default: 0.1)
                - layers: List of layer names (default: ['A', 'B', 'C', 'D'])
                - capital_preservation: Enable capital preservation (default: True)
                - capital_preservation_layer: Layer for capital preservation (default: second-to-last)
                - capital_preservation_target: Capital preservation target (default: 0.65)
                - max_loss_per_layer: Maximum loss per layer (default: 1000)
                - method: Layer sizing method ('sizing_up' or 'sizing_down', default: 'sizing_up')
                - price_upper_bound: Maximum price for first layer (default: 0.95 * avg_price)
                - layer_steps: Price steps between layers (default: [0.2, 0.4, 0.3])
                - minimum_shares_per_layer: Minimum shares per layer (default: 0.15 * total_shares)

        Returns:
            Solved Gurobi model

        Raises:
            ValueError: If configuration is invalid or solver fails
        """
        total_shares = config.get("total_shares")
        avg_price = config.get("avg_price")
        if not total_shares or not avg_price or total_shares <= 0 or avg_price <= 0:
            raise ValueError("total_shares and avg_price must be positive numbers")

        initial_capital = total_shares * avg_price
        target_loss_percentage = config.get("target_loss_percentage", 0.1)
        layers = config.get("layers", ["A", "B", "C", "D"])
        capital_preservation = config.get("capital_preservation", True)
        capital_preservation_layer = config.get("capital_preservation_layer", layers[-2])
        capital_preservation_target = config.get("capital_preservation_target", 0.65)
        max_loss_per_layer = config.get("max_loss_per_layer", 1000)
        method = config.get("method", "sizing_up")
        price_upper_bound = config.get("price_upper_bound", 0.95 * avg_price)
        layer_steps = config.get("layer_steps", [0.2, 0.4, 0.3])
        minimum_shares_per_layer = config.get("minimum_shares_per_layer", 0.15 * total_shares)

        target_loss = target_loss_percentage * initial_capital
        if not layers or len(layers) < 2:
            raise ValueError("At least two layers are required")

        model = gp.Model("StopLossStrategy")
        model.setParam("OutputFlag", 0)

        shares = model.addVars(layers, vtype=GRB.INTEGER, name="shares", lb=minimum_shares_per_layer)
        prices = model.addVars(layers, vtype=GRB.CONTINUOUS, name="prices", lb=0.01)

        model.setObjective(
            gp.quicksum(shares[layer] * prices[layer] for layer in layers),
            sense=GRB.MAXIMIZE
        )

        model.addConstr(
            gp.quicksum(shares[layer] for layer in layers) == total_shares,
            name="TotalShares"
        )

        if method in ["sizing_up", "sizing_down"]:
            for i in range(1, len(layers)):
                if method == "sizing_up":
                    model.addConstr(
                        shares[layers[i]] >= shares[layers[i-1]] + 1,
                        name=f"Hierarchy_{layers[i]}"
                    )
                else:
                    model.addConstr(
                        shares[layers[i]] <= shares[layers[i-1]] + 1,
                        name=f"Hierarchy_{layers[i]}"
                    )

        model.addConstr(
            gp.quicksum(shares[layer] * (avg_price - prices[layer]) for layer in layers) == target_loss,
            name="CumulativeLoss"
        )

        if capital_preservation:
            if capital_preservation_layer not in layers:
                raise ValueError("capital_preservation_layer must be one of the layers")
            layer_index = layers.index(capital_preservation_layer) + 1
            layers_to_sum = layers[:layer_index]
            model.addConstr(
                gp.quicksum(shares[layer] * prices[layer] for layer in layers_to_sum) >=
                capital_preservation_target * initial_capital,
                name="CapitalPreservation"
            )

        if price_upper_bound:
            model.addConstr(
                prices[layers[0]] <= price_upper_bound,
                name="PriceUpperBound_A"
            )

        if max_loss_per_layer:
            for layer in layers:
                model.addConstr(
                    shares[layer] * (avg_price - prices[layer]) <= max_loss_per_layer,
                    name=f"MaxLoss_{layer}"
                )

        if layer_steps and len(layer_steps) >= len(layers) - 1:
            for i in range(1, len(layers)):
                model.addConstr(
                    prices[layers[i]] <= prices[layers[i-1]] * (1 - layer_steps[i-1]),
                    name=f"PriceGap_{layers[i]}{layers[i-1]}"
                )

        model.setParam("NonConvex", 2)
        model.optimize()

        if model.Status != GRB.OPTIMAL:
            raise ValueError(f"Solver did not find an optimal solution. Status code: {model.Status}")

        print(f"\n{'=' * 20} Optimal Stop-Loss Strategy for {self.ticker} {'=' * 20}\n")
        self.stoploss_display_solution(total_shares, layers, avg_price, model)

        probability, ohlc_paths = self.stoploss_calculate_solution_probability(layers, model)
        print(f"\n{'=' * 20} PROBABILITY ANALYSIS {'=' * 20}")
        print(tabulate(probability, headers="keys", tablefmt="fancy_grid"))

        return model

    def stoploss_display_solution(self, total_shares: int, layers: List[str], avg_price: float, model: gp.Model) -> None:
        """
        Display the optimization results in a tabulated format.

        Args:
            total_shares: Total number of shares
            layers: List of layer names
            avg_price: Average purchase price
            model: Solved Gurobi model

        Raises:
            ValueError: If model is not solved
        """
        if model.Status != GRB.OPTIMAL:
            raise ValueError("Model must be solved optimally before displaying solution")

        table_data = []
        total_loss_check = 0
        remaining_shares = total_shares
        preserved_capital_cumulative = 0

        share_vars = model.getVars()[:len(layers)]
        price_vars = model.getVars()[len(layers):]
        solution_data = {
            f"Layer {layer}": {"shares": round(s_var.X), "price": round(p_var.X, 2)}
            for layer, s_var, p_var in zip(layers, share_vars, price_vars)
        }

        for layer_name in layers:
            shares = solution_data[f"Layer {layer_name}"]["shares"]
            price = solution_data[f"Layer {layer_name}"]["price"]
            preserved_capital_cumulative += shares * price
            loss_for_layer = shares * (avg_price - price)
            total_loss_check += loss_for_layer
            remaining_shares -= shares

            table_data.append({
                "Layer": layer_name,
                "Total Shares": shares,
                "Trigger Price": f"${price:.2f}",
                "Loss For Layer": f"${loss_for_layer:,.2f}",
                "Preserved Capital": f"${preserved_capital_cumulative:,.2f}",
                "Remaining Shares": remaining_shares,
                "Market Value": f"${remaining_shares * price:,.2f}"
            })

        print(tabulate(table_data, headers="keys", tablefmt="fancy_grid"))
