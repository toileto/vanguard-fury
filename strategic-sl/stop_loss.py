import gurobipy as gp
from gurobipy import GRB
from typing import Dict, List, Tuple
from tabulate import tabulate
from stockdata import StockData
from monte_carlo import MonteCarlo


class StopLossOptimizer:
    """
    Optimizes stop-loss strategies using Gurobi and Monte Carlo simulations.
    """

    def __init__(self, ticker: str):
        """
        Initialize the StopLossOptimizer.

        Args:
            ticker: Stock ticker symbol

        Raises:
            ValueError: If ticker is empty or invalid
        """
        if not ticker or not isinstance(ticker, str):
            raise ValueError("Ticker must be a non-empty string")
        self.ticker = ticker
        self.stockdata_client = StockData()
        self.monte_carlo = MonteCarlo()
        self.historical_data = self.stockdata_client.get_price_data(ticker)
        self.solution = None

    def calculate_solution_probability(self,
                                       layers: List[str],
                                       model: gp.Model
                                       ) -> Tuple[List[Dict], List]:
        """
        Calculate probabilities for stop-loss triggers using Monte Carlo
        simulation.

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
            raise ValueError(
                "Model must be solved optimally before probability calculation")

        share_vars = model.getVars()[:len(layers)]
        price_vars = model.getVars()[len(layers):]
        solution_data = {
            f"Layer {layer}": {"shares": round(s_var.X), "price": round(p_var.X, 2)}
            for layer, s_var, p_var in zip(layers, share_vars, price_vars)
        }
        stop_loss_prices = [data["price"] for data in solution_data.values()]

        result = self.monte_carlo.analyse_stop_loss_probability(
            price=self.historical_data,
            stop_loss=stop_loss_prices,
            time_horizon_days=[5, 10, 20],
            distribution_per_share=0.10,
            distribution_frequency_days=5
        )
        return result

    def build_and_solve_model(self, config: Dict) -> gp.Model:
        """
        Build and solve a Gurobi optimization model for stop-loss strategy.

        Args:
            config: Configuration dictionary with parameters:
                - total_shares: Total number of shares
                - avg_price: Average purchase price
                - target_loss_percentage: Target loss as a
                    percentage (default: 0.1)
                - layers: List of layer names (default: ['A', 'B', 'C', 'D'])
                - capital_preservation: Enable capital preservation
                    (default: True)
                - capital_preservation_layer: Layer for capital preservation
                    (default: second-to-last)
                - capital_preservation_target: Capital preservation target
                    (default: 0.65)
                - max_loss_per_layer: Maximum loss per layer (default: 1000)
                - method: Layer sizing method ('sizing_up' or 'sizing_down',
                    default: 'sizing_up')
                - price_upper_bound: Maximum price for first layer
                    (default: 0.95 * avg_price)
                - layer_steps: Price steps between layers
                    (default: [0.2, 0.4, 0.3])
                - minimum_shares_per_layer: Minimum shares per layer
                    (default: 0.15 * total_shares)

        Returns:
            Solved Gurobi model

        Raises:
            ValueError: If configuration is invalid or solver fails
        """
        # Extract and validate configuration
        total_shares = config.get("total_shares")
        avg_price = config.get("avg_price")

        if (not total_shares or not avg_price or total_shares <= 0
                or avg_price <= 0):
            raise ValueError(
                "total_shares and avg_price must be positive numbers")

        initial_capital = total_shares * avg_price
        target_loss_percentage = config.get("target_loss_percentage", 0.1)
        layers = config.get("layers", ["A", "B", "C", "D"])
        capital_preservation = config.get("capital_preservation", True)
        capital_preservation_layer = config.get(
            "capital_preservation_layer", layers[-2])
        capital_preservation_target = config.get(
            "capital_preservation_target", 0.65)
        max_loss_per_layer = config.get("max_loss_per_layer", 1000)
        method = config.get("method", "sizing_up")
        price_upper_bound = config.get("price_upper_bound", 0.95 * avg_price)
        layer_steps = config.get("layer_steps", [0.2, 0.4, 0.3])
        minimum_shares_per_layer = config.get(
            "minimum_shares_per_layer", 0.15 * total_shares)

        target_loss = target_loss_percentage * initial_capital
        if not layers or len(layers) < 2:
            raise ValueError("At least two layers are required")

        # Initialize Gurobi model
        model = gp.Model("StopLossStrategy")
        # Suppress Gurobi output
        model.setParam("OutputFlag", 0)

        # Decision variables
        shares = model.addVars(layers, vtype=GRB.INTEGER, name="shares", lb=minimum_shares_per_layer)
        prices = model.addVars(layers, vtype=GRB.CONTINUOUS, name="prices", lb=0.01)

        # Objective: Maximize total revenue
        model.setObjective(
            gp.quicksum(shares[layer] * prices[layer] for layer in layers),
            sense=GRB.MAXIMIZE
        )

        # Constraints
        model.addConstr(
            gp.quicksum(shares[layer] for layer in layers) == total_shares,
            name="TotalShares"
        )

        # Layer hierarchy constraints
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

        # Loss constraint
        model.addConstr(
            gp.quicksum(shares[layer] * (avg_price - prices[layer]) for layer in layers) == target_loss,
            name="CumulativeLoss"
        )

        # Capital preservation constraint
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

        # Price upper bound constraint
        if price_upper_bound:
            model.addConstr(
                prices[layers[0]] <= price_upper_bound,
                name="PriceUpperBound_A"
            )

        # Max loss per layer constraint
        if max_loss_per_layer:
            for layer in layers:
                model.addConstr(
                    shares[layer] * (avg_price - prices[layer]) <= max_loss_per_layer,
                    name=f"MaxLoss_{layer}"
                )

        # Price gap constraints
        if layer_steps and len(layer_steps) >= len(layers) - 1:
            for i in range(1, len(layers)):
                model.addConstr(
                    prices[layers[i]] <= prices[layers[i-1]] * (1 - layer_steps[i-1]),
                    name=f"PriceGap_{layers[i]}{layers[i-1]}"
                )

        # Solve model
        model.setParam("NonConvex", 2)
        model.optimize()

        if model.Status != GRB.OPTIMAL:
            raise ValueError(f"Solver did not find an optimal solution. Status code: {model.Status}")

        # Display results
        print(f"\n{'=' * 20} Optimal Stop-Loss Strategy for {self.ticker} {'=' * 20}\n")
        self.display_solution(total_shares, layers, avg_price, model)

        # Calculate and display probability
        probability_result = (
            self.calculate_solution_probability(layers, model)
        )

        compact = probability_result[0]
        all_data = probability_result[1]
        ohlc_path = probability_result[2]

        print(f"\n{'=' * 20} PROBABILITY ANALYSIS {'=' * 20}")
        print("\n" + ("=" * 10) + " SUMMARY " + ("=" * 10))
        print(tabulate(compact, headers="keys", tablefmt="fancy_grid"))
        print("\n" + ("=" * 10) + " FULL ANALYSIS " + ("=" * 10))
        print(tabulate(all_data, headers="keys", tablefmt="fancy_grid"))

        self.solution = model
        return model

    def display_solution(self, total_shares: int, layers: List[str], avg_price: float, model: gp.Model) -> None:
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
