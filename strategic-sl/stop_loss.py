import gurobipy as gp
from gurobipy import GRB
from tabulate import tabulate
from stockdata import StockData
from monte_carlo import MonteCarlo

class StopLossOptimizer:

    def __init__(self, ticker):
        self.stockdata_client = StockData()
        self.monte_carlo = MonteCarlo()
        self.ticker = ticker
        self.historical_data = self.stockdata_client.get_price_data(
            self.ticker)
        self.solution = None
        pass

    def calculate_solution_probability(self, layers, model):
        s = model.getVars()[0:len(layers)]  # Share variables
        p = model.getVars()[len(layers):]  # Price variables
        solution_data = {f"Layer {layer}": {'shares': round(s_var.X),
                                            'price': float(f"{p_var.X:.2f}")}
                         for layer, s_var, p_var in zip(layers, s, p)}
        stop_loss_prices = [val['price'] for val in solution_data.values()]
        stats_data, ohlc_paths = self.monte_carlo.analyse_stop_loss_probability(
            price=self.historical_data,
            stop_loss=stop_loss_prices,
            time_horizon_days=[5, 10, 20],
            distribution_per_share=0.10,
            distribution_frequency_days=5
        )

        return stats_data, ohlc_paths



    def build_and_solve_model(self,
                              _config:dict):
        """
        Builds and solves the Gurobi optimization model based on the provided configuration.

        Args:
            config (dict):
                A dictionary containing all necessary parameters for the model.
                - price_upper_bound: The minimum price of the stop-loss.
                - method: sizing_up or sizing_down

        Returns:
            gurobipy.Model: The solved Gurobi model object.
        """

        total_shares = _config.get('total_shares')
        avg_price = _config.get('avg_price')
        initial_capital = total_shares * avg_price
        target_loss_percentage = _config.get('target_loss_percentage') or 0.1
        layers = _config.get('layers') or ['A', 'B', 'C', 'D']
        capital_preservation = _config.get('capital_preservation') or True
        capital_preservation_layer = \
            _config.get('capital_preservation_layer') or layers[-2]
        capital_preservation_target = \
            _config.get('capital_preservation_target') or 0.65
        max_loss_per_layer = _config.get("max_loss_per_layer") or 1000
        method = _config.get('method') or "sizing_up"
        price_upper_bound = \
            _config.get('price_upper_bound') or (0.95 * avg_price)
        layer_steps = _config.get('layer_steps') or [0.2, 0.4, 0.3]
        target_loss = target_loss_percentage * initial_capital
        minimum_shares_per_layer = (
            _config.get('minimum_shares_per_layer')
            or 0.15 * total_shares
        )

        print(initial_capital)

        m = gp.Model("StopLossStrategy")

        # --- Decision Variables ---
        s = m.addVars(
            layers, vtype=GRB.INTEGER, name="s", lb=minimum_shares_per_layer)
        p = m.addVars(layers, vtype=GRB.CONTINUOUS, lb=0.01, name="p")

        # --- Objective Function: Maximize total revenue from sales ---
        m.setObjective(gp.quicksum(s[i] * p[i] for i in layers),
                       sense=GRB.MAXIMIZE)

        m.addConstr(
            gp.quicksum(s[i] for i in layers) == total_shares,"TotalShares"
        )

        if method == "sizing_up":
            for i in range(1, len(layers)):
                m.addConstr(s[layers[i]] >= s[layers[i-1]] + 1,
                            f"Hierarchy_{s[layers[i]]}")
        elif method == "sizing_down":
            for i in range(1, len(layers)):
                m.addConstr(s[layers[i]] <= s[layers[i-1]] + 1,
                            f"Hierarchy_{s[layers[i]]}")

        m.addConstr(
            gp.quicksum(s[i] * (avg_price - p[i]) for i in layers) ==
            target_loss, "CumulativeLoss"
        )

        if capital_preservation:
            if capital_preservation_layer not in layers:
                raise("capital_perservation_layer must be one of the layers.")

            layer_index = layers.index(capital_preservation_layer) + 1
            layers_to_sum = layers[:layer_index]

            m.addConstr(
                gp.quicksum(s[i] * p[i] for i in layers_to_sum) >=
                capital_preservation_target * initial_capital,
                "CapitalPreservation"
            )

        if price_upper_bound:
            m.addConstr(
                p[layers[0]] <= price_upper_bound,"PriceUpperBound_A"
            )

        if max_loss_per_layer:
            for i in layers:
                m.addConstr(
                    s[i] * (avg_price - p[i]) <= max_loss_per_layer,
                    f"MaxLoss_{i}")

        if layer_steps:
            for i in range(1, len(layers)):
                print((1 - layer_steps[i-1]))
                m.addConstr(
                    p[layers[i]] <= p[layers[i-1]] * (1 - layer_steps[i-1]),
                f"PriceGap_{layers[i]}{layers[i-1]}"
                )

        # --- Solve the model ---
        m.setParam('NonConvex', 2)
        m.optimize()

        if m.Status != GRB.OPTIMAL:
            raise(
                "\nSolver did not find an optimal solution. "
                f"Status code: {m.Status}"
            )

        print(
            "\n" +
            ("=" * 20) +
            f" Optimal Stop-Loss Strategy for {self.ticker} " +
            ("=" * 20) +
            "\n"
        )
        self.display_solution(total_shares, layers, avg_price, m)

        # Calculate Probability
        probability, ohlc_path = self.calculate_solution_probability(
            model=m,
            layers=layers
        )
        print("\n" + "=" * 20 + " PROBABILITY ANALYSIS " + "=" * 20)
        print(tabulate(probability, headers="keys", tablefmt="fancy_grid"))


        return m

    def display_solution(self, total_shares, layers, avg_price, solution):
        # --- Prepare data for tabular display ---
        table_data = []
        total_loss_check = 0
        remaining_shares = total_shares
        preserved_capital_cumulative = 0

        share = solution.getVars()[0:len(layers)]  # Share variables
        price = solution.getVars()[len(layers):]  # Price variables

        solution_data = {f"Layer {layer}": {'shares': round(s_var.X),
                                            'price': float(f"{p_var.X:.2f}")}
                         for layer, s_var, p_var in zip(layers, share, price)}

        for i, layer_name in enumerate(layers):
            shares = solution_data[f"Layer {layer_name}"]['shares']
            price = solution_data[f"Layer {layer_name}"]['price']

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
