import numpy as np


class MonteCarlo:

    def __init__(self):
        pass

    def dict_to_list(self, list_of_dict:list, key="close"):
        return [i[key] for i in list_of_dict]

    def calculate_nav_erosion(self, prices: list):
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

    def calculate_volatility(self, price: list, method='garman_klass'):
        """
        Estimate daily volatility from a list of OHLCV dictionaries.

        Parameters:
        - price: List of dicts with keys 'open', 'high', 'low', 'close'
        (in chronological order).
        - method: 'garman_klass' (default) or 'parkinson'.

        Returns:
        - daily_volatility: Float (decimal, e.g., 0.0102 for 1.02%).
        """
        if len(price) < 2:
            raise ValueError("At least two OHLCV entries are required.")

        # Extract OHLC arrays
        opens = np.array([d['open'] for d in price])
        highs = np.array([d['high'] for d in price])
        lows = np.array([d['low'] for d in price])
        closes = np.array([d['close'] for d in price])

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

    def calculate_daily_drift(self, price: list, risk_free_rate_annual=0.0,
                              use_historical=True, use_theoretical=True):
        """
        Calculate daily drift from OHLCV dicts, with optional theoretical
        adjustment.

        Parameters:
        - price: List of dicts with 'close' key (optional for historical).
        - risk_free_rate_annual: Annual risk-free rate (default: 0.0 assume
        conservative).
        - nav_erosion_annual: Annual NAV erosion.
        - use_historical/use_theoretical: Flags to include methods.

        Returns:
        - daily_drift: Float (decimal).
        - results: Dict with breakdown.
        """
        results = {}

        # Historical drift
        historical_drift = None
        if use_historical and price is not None:
            if len(price) < 2:
                raise ValueError(
                    "At least two OHLCV entries for historical drift.")
            closes = np.array([d['close'] for d in price])
            log_returns = np.log(closes[1:] / closes[:-1])
            historical_drift = np.mean(log_returns)
            results["historical_drift"] = historical_drift

        # Theoretical drift
        theoretical_drift = None
        if use_theoretical:
            daily_risk_free = risk_free_rate_annual / 252
            daily_nav_erosion = self.calculate_nav_erosion(self.dict_to_list(
                price)) / 252
            theoretical_drift = daily_risk_free + daily_nav_erosion
            results["theoretical_drift"] = theoretical_drift

        # Combine or select
        if (use_historical and use_theoretical and historical_drift is not None
                and theoretical_drift is not None):
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

    def estimate_volatility(self, price: list, method='garman_klass'):
        """
        Estimate daily volatility from a list of OHLCV dictionaries.

        Parameters:
        - price: List of dicts with keys 'open', 'high', 'low', 'close' (in
        chronological order).
        - method: 'garman_klass' (default) or 'parkinson'.

        Returns:
        - daily_volatility: Float (decimal, e.g., 0.0102 for 1.02%).
        """
        if len(price) < 2:
            raise ValueError("At least two OHLCV entries are required.")

        # Extract OHLC arrays
        opens = np.array([d['open'] for d in price])
        highs = np.array([d['high'] for d in price])
        lows = np.array([d['low'] for d in price])
        closes = np.array([d['close'] for d in price])

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

    def analyse_stop_loss_probability(
            self,
            price: list,
            stop_loss: list,
            time_horizon_days: list,
            distribution_per_share: float,
            distribution_frequency_days: int,
            n_simulations=10000,
            volatility_method='garman_klass',
            intraday_steps=0,
            result_mode="compact"
    ):
        """
        Monte Carlo simulation using OHLCV dicts for stop-loss probability.

        Parameters:
        - current_price: Current ETF price (e.g., 6.01).
        - stop_loss: Stop-loss level (e.g., 5.50).
        - ohlcv_list: List of dicts with 'date', 'open', 'high',
        'low', 'close', 'volume'.
        - time_horizon_days: Simulation period (e.g., 1, 5, 20).
        - distribution_per_share: Weekly distribution (e.g., 0.10).
        - distribution_frequency_days: Days between distributions (e.g., 5).
        - n_simulations: Number of paths (default: 10000).
        - volatility_method: 'garman_klass' or 'parkinson'.
        - intraday_steps: If >0, simulate intraday for OHLC output
         (e.g., 390 for a trading day).

        Returns:
        - probability: % chance of hitting stop-loss.
        - stats: Dict with price distribution stats.
        - ohlc_paths: List of simulated OHLC arrays (if intraday_steps > 0).
        """

        current_price = price[0]['close']

        # Estimate parameters
        daily_volatility = self.estimate_volatility(price,
                                                    method=volatility_method)

        daily_drift, _ = self.calculate_daily_drift(
            price,
            risk_free_rate_annual=0.0,
            use_historical=True,
            use_theoretical=True
        )

        all_data = list()
        compact_stats = list()

        for _stop_loss in stop_loss:
            _compact_stats = {
                "stop_loss": _stop_loss,
                "current_price": current_price,
            }
            for _time in time_horizon_days:

                # Time step
                dt = 1
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
                                intraday_prices.append(
                                    intraday_prices[-1] * np.exp(
                                        (daily_drift - 0.5 * daily_volatility
                                         ** 2) + daily_volatility * z
                                    )
                                )
                            o = intraday_prices[0]
                            h = np.max(intraday_prices)
                            l = np.min(intraday_prices)
                            price = intraday_prices[-1]
                            path_ohlc.append([o, h, l, price])
                        else:
                            z = np.random.normal(0, 1)
                            price *= np.exp(
                                (daily_drift - 0.5 * daily_volatility ** 2)
                                * dt + daily_volatility * np.sqrt(dt) * z
                            )

                        if (t + 1) % distribution_frequency_days == 0:
                            price = max(price - distribution_per_share, 0)

                        if price <= _stop_loss:
                            hit_stop_loss[i] = True
                            break

                    final_prices[i] = price

                    if ohlc_paths is not None:
                        ohlc_paths.append(np.array(path_ohlc))

                probability = np.mean(hit_stop_loss) * 100

                _compact_stats[f"hit_in_{_time}_days"] = probability

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

            compact_stats.append(_compact_stats)

        if result_mode != "compact":
            return all_data, ohlc_paths
        else:
            return compact_stats, ohlc_paths
