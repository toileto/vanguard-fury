import os
import pyomo.environ as pyo
import numpy as np
import requests

from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta

class VanguardFury:

    RED_TERM = "\033[91m"

    def __init__(self, ticker, historical_price):
        self.ticker = ticker
        self.api_key = os.environ['STOCKDATA_API_KEY']
        if not self.api_key:
            raise ValueError("STOCKDATA_API_KEY environment variable not set")
        self.stockdata_base_url = "https://api.stockdata.org/v1/data/"
        # self.price_data = self.get_price_data()
        self.price_data = historical_price
        pass

    def hit_stockdata_api(self, url: str) -> List[Dict]:
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

    def get_price_data(self, lookback_period: int = 20) -> List[Dict]:
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

        if lookback_period <= 0:
            raise ValueError("Lookback period must be positive")

        url = (
            f"{self.stockdata_base_url}eod?symbols={self.ticker}"
            f"&api_token={self.api_key}"
        )
        raw_data = self.hit_stockdata_api(url)

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

    def optimise_stop_loss(self, total_shares: int, avg_price: float,
                          target_loss_percentage: float = 0.1,
                          num_layers: int = 4,
                          capital_preservation: bool = True,
                          capital_preservation_layer: int = 3,
                          capital_preservation_target: float = 0.65,
                          max_loss_per_layer: int = None,
                          price_upper_bound: float = None,
                          method: str = "sizing_up",
                          layer_steps: List[float] = [0.02, 0.04, 0.03],
                          minimum_shares_per_layer: float = None
                          ):

        # Make up parameters
        initial_capital = avg_price * total_shares

        layers = list()
        for i in range(num_layers):
            layers.append(str(i+1))

        if minimum_shares_per_layer is None:
            minimum_shares_per_layer = 0.15 * total_shares

        if price_upper_bound is None:
            price_upper_bound = 0.95 * avg_price


        # Validate input
        if not max_loss_per_layer:
            raise ValueError("max_loss_per_layer can not be empty...")

        if not total_shares:
            raise ValueError("Total shares cannot be empty...")

        if not avg_price:
            raise ValueError("Avg price cannot be empty...")

        if capital_preservation_layer > num_layers:
            raise ValueError(
                "Capital preservation layer cannot be "
                "greater than number of layers..."
            )
        else:
            capital_preservation_layer = layers[capital_preservation_layer]

        if capital_preservation_target >= 1:
            raise ValueError("Capital preservation target cannot be greater "
                             "than or equal to 1...")

        if method not in ['sizing_up', 'sizing_down']:
            raise ValueError(
                "Method must be either sizing_up or sizing_down...")

        if len(layer_steps) != (num_layers - 1):
            raise ValueError("Length of layer_step must"
                             " be equal to number of layers - 1...")

        target_loss = target_loss_percentage * initial_capital
        if not layers or len(layers) < 2:
            raise ValueError("At least two layers are required")

        # --- 2. Initialize Pyomo model ---
        model = pyo.ConcreteModel("StopLossStrategy")

        # --- 3. Define Sets and Decision Variables ---
        model.LAYERS = pyo.Set(initialize=layers, ordered=True)

        model.shares = pyo.Var(model.LAYERS, domain=pyo.NonNegativeIntegers,
                               bounds=(minimum_shares_per_layer, total_shares),
                               name="shares")
        model.prices = pyo.Var(model.LAYERS, domain=pyo.NonNegativeReals,
                               bounds=(0.01, None),
                               name="prices")

        # --- 4. Define the Objective Function ---
        objective_expr = sum(
            model.shares[layer] * model.prices[layer] for layer in model.LAYERS)
        model.objective = pyo.Objective(expr=objective_expr, sense=pyo.maximize)

        # --- 5. Define Constraints (same as original) ---
        # Total Shares Constraint
        model.total_shares_constr = pyo.Constraint(
            expr=sum(
                model.shares[layer] for layer in model.LAYERS) == total_shares
        )

        # Layer Hierarchy Constraints
        model.hierarchy_constr = pyo.ConstraintList()
        ordered_layers = list(model.LAYERS)
        if method in ["sizing_up", "sizing_down"]:
            for i in range(1, len(ordered_layers)):
                prev_layer = ordered_layers[i - 1]
                curr_layer = ordered_layers[i]
                if method == "sizing_up":
                    model.hierarchy_constr.add(
                        model.shares[curr_layer] >= model.shares[prev_layer] + 1
                    )
                else:  # sizing_down
                    model.hierarchy_constr.add(
                        model.shares[curr_layer] <= model.shares[prev_layer] + 1
                    )

        # Cumulative Loss Constraint
        model.cumulative_loss_constr = pyo.Constraint(
            expr=sum(model.shares[layer] * (avg_price - model.prices[layer]) for
                     layer in model.LAYERS) == target_loss
        )

        # Capital Preservation Constraint
        if capital_preservation:
            if capital_preservation_layer not in layers:
                raise ValueError(
                    "capital_preservation_layer must be one of the layers")
            layer_index = ordered_layers.index(capital_preservation_layer) + 1
            layers_to_sum = ordered_layers[:layer_index]
            model.capital_preservation_constr = pyo.Constraint(
                expr=sum(model.shares[layer] * model.prices[layer] for layer in
                         layers_to_sum) >=
                     capital_preservation_target * initial_capital
            )

        # Price Upper Bound Constraint
        if price_upper_bound:
            first_layer = ordered_layers[0]
            model.price_upper_bound_constr = pyo.Constraint(
                expr=model.prices[first_layer] <= price_upper_bound
            )

        # Max Loss Per Layer Constraint
        def max_loss_rule(model, layer):
            return model.shares[layer] * (
                    avg_price - model.prices[layer]) <= max_loss_per_layer

        if max_loss_per_layer:
            model.max_loss_constr = pyo.Constraint(model.LAYERS,
                                                   rule=max_loss_rule)

        # Price Gap Constraints
        model.price_gap_constr = pyo.ConstraintList()
        if layer_steps and len(layer_steps) >= len(layers) - 1:
            for i in range(1, len(ordered_layers)):
                prev_layer = ordered_layers[i - 1]
                curr_layer = ordered_layers[i]
                model.price_gap_constr.add(
                    model.prices[curr_layer] <= model.prices[prev_layer] * (
                            1 - layer_steps[i - 1])
                )

        # --- 6. Solve the Model ---
        solver = SolverFactory('gurobi')
        solver.options['NonConvex'] = 2

        print("\nSolving the Pyomo model with Gurobi...")
        results = solver.solve(model, tee=False)

        # --- 7. Check Solution, Display, and Format Results ---
        if (results.solver.status == SolverStatus.ok) and (
                results.solver.termination_condition == TerminationCondition.optimal):
            print("Optimal solution found!")

            solution_list = []
            for layer_name in model.LAYERS:
                solution_list.append({
                    'layer': layer_name,
                    'shares': int(round(pyo.value(model.shares[layer_name]))),
                    'price': round(pyo.value(model.prices[layer_name]),
                                           2)
                })

            return solution_list

        else:
            print(f"Solver did not find an optimal solution.")
            print(f"Solver Status: {results.solver.status}")
            print(
                f"Termination Condition: {results.solver.termination_condition}")
            # Raise an error or return an empty list if no solution is found
            print(f"{self.RED_TERM}Problems can not be solved with such "
                  f"condition...")

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

    def calculate_target_price_probability(
            self,
            target_price: float,
            distribution_per_share: float,
            distribution_frequency_days: int,
            time_horizon_days: int = 5,
            n_simulations: int = 10000,
            volatility_method: str = 'garman_klass',
            intraday_steps: int = 0,
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

        price = self.price_data

        # if not price or 'close' not in price[0]:
        #     raise KeyError("Price data must contain 'close' key")
        if not target_price or not time_horizon_days:
            raise ValueError("Stop-loss prices and time horizons cannot be empty")
        if distribution_per_share < 0 or distribution_frequency_days <= 0:
            raise ValueError("Distribution parameters must be positive")
        if n_simulations <= 0:
            raise ValueError("Number of simulations must be positive")

        current_price = price[0]['close']
        daily_volatility = self.estimate_volatility(price,
                                                    method=volatility_method)
        daily_drift, _ = self.calculate_daily_drift(price,
                                                    use_historical=True,
                                                    use_theoretical=True)

        hit_stop_loss = np.zeros(n_simulations, dtype=bool)
        final_prices = np.zeros(n_simulations)
        ohlc_paths = [] if intraday_steps > 0 else None

        for i in range(n_simulations):
            price = current_price
            path_ohlc = [] if intraday_steps > 0 else None

            for t in range(time_horizon_days):
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

                if price <= target_price:
                    hit_stop_loss[i] = True
                    break

            final_prices[i] = price
            if path_ohlc:
                ohlc_paths.append(np.array(path_ohlc))

        probability = np.mean(hit_stop_loss) * 100

        stats = {
            "target_price": target_price,
            "time_horizon": time_horizon_days,
            "current_price": current_price,
            "percent_probability": float(probability),
            "mean_price": float(np.mean(final_prices)),
            "median_price": float(np.median(final_prices)),
            "25th_percentile": float(np.percentile(final_prices, 25)),
            "75th_percentile": float(np.percentile(final_prices, 75)),
            "volatility_used": daily_volatility,
            "drift_used": float(daily_drift)
        }

        return stats

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
