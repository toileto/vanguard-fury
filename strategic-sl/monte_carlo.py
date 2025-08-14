import numpy as np
from typing import List, Dict, Tuple, Optional


class MonteCarlo:
    """Performs Monte Carlo simulations for stop-loss probability analysis."""

    def __init__(self):
        """Initialize MonteCarlo instance."""
        pass

    def dict_to_list(self, data: List[Dict], key: str = "close") -> List[float]:
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

    def calculate_nav_erosion(self, prices: List[float]) -> float:
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

    def estimate_volatility(self, price_data: List[Dict], method: str = 'garman_klass') -> float:
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

    def calculate_daily_drift(self, price_data: List[Dict], risk_free_rate_annual: float = 0.0,
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

        # Historical drift
        if use_historical:
            if len(price_data) < 2:
                raise ValueError("At least two OHLCV entries required for historical drift")
            closes = np.array([d['close'] for d in price_data])
            log_returns = np.log(closes[1:] / closes[:-1])
            historical_drift = np.mean(log_returns)
            results["historical_drift"] = historical_drift
            daily_drift = historical_drift

        # Theoretical drift
        if use_theoretical:
            daily_risk_free = risk_free_rate_annual / 252
            daily_nav_erosion = self.calculate_nav_erosion(self.dict_to_list(price_data)) / 252
            theoretical_drift = daily_risk_free + daily_nav_erosion
            results["theoretical_drift"] = theoretical_drift
            daily_drift = theoretical_drift

        # Combine drifts if both are used
        if use_historical and use_theoretical and historical_drift is not None and theoretical_drift is not None:
            daily_drift = (historical_drift + theoretical_drift) / 2
            results["combined_drift"] = daily_drift

        if daily_drift is None:
            raise ValueError("No valid drift calculation method or data available")

        return daily_drift, results

    def analyse_stop_loss_probability(
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
    ) -> Tuple[List[Dict], Optional[List[np.ndarray]]]:
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
            Tuple containing:
            - List of probability statistics (compact or detailed format)
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
        daily_volatility = self.estimate_volatility(price, method=volatility_method)
        daily_drift, _ = self.calculate_daily_drift(price, use_historical=True, use_theoretical=True)

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
