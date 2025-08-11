import gurobipy as gp
from gurobipy import GRB
from tabulate import tabulate
from black_scholes import monte_carlo_stop_loss_probability
from solution_visualization import visualize_stop_loss_strategy


def build_and_solve_model(config):
    """
    Builds and solves the Gurobi optimization model based on the provided configuration.

    Args:
        config (dict): A dictionary containing all necessary parameters for the model.

    Returns:
        gurobipy.Model: The solved Gurobi model object.
    """
    m = gp.Model("StopLossStrategy")

    # --- Decision Variables ---
    layers = config['layers']
    s = m.addVars(layers, vtype=GRB.INTEGER, name="s", lb=100)
    p = m.addVars(layers, vtype=GRB.CONTINUOUS, lb=0.01, name="p")

    # --- Objective Function: Maximize total revenue from sales ---
    m.setObjective(gp.quicksum(s[i] * p[i] for i in layers), sense=GRB.MAXIMIZE)

    # --- Constraints ---
    m.addConstr(gp.quicksum(s[i] for i in layers) == config['total_shares'],
                "TotalShares")
    m.addConstr(s['B'] >= s['A'] + 1, "Hierarchy_B")
    m.addConstr(s['C'] >= s['B'] + 1, "Hierarchy_C")
    m.addConstr(s['D'] >= s['C'] + 1, "Hierarchy_D")

    m.addConstr(
        gp.quicksum(s[i] * (config['avg_price'] - p[i]) for i in layers) ==
        config['target_loss'], "CumulativeLoss")

    m.addConstr(
        (s['A'] * p['A']) + (s['B'] * p['B']) + (s['C'] * p['C']) >= config[
            'target_at_c_layer'] * config['initial_capital'],
        "CapitalPreservation")

    m.addConstr(p['A'] <= 5.85, "PriceUpperBound_A")

    for i in layers:
        m.addConstr(
            s[i] * (config['avg_price'] - p[i]) <= config['max_loss_per_layer'],
            f"MaxLoss_{i}")

    m.addConstr(p['B'] <= p['A'] * 0.98, "PriceGap_AB")
    m.addConstr(p['C'] <= p['B'] * 0.96, "PriceGap_BC")
    m.addConstr(p['D'] <= p['C'] * 0.97, "PriceGap_CD")

    # --- Solve the model ---
    m.setParam('NonConvex', 2)
    m.optimize()

    return m


def process_and_display_results(model, config):
    """
    Processes the optimal solution from the model and displays the results.

    Args:
        model (gurobipy.Model): The solved Gurobi model object.
        config (dict): The configuration dictionary.
    """
    if model.Status != GRB.OPTIMAL:
        print(
            f"\nSolver did not find an optimal solution. Status code: {model.Status}")
        return

    print("\n--- Optimal Stop-Loss Strategy ---\n")

    layers = config['layers']
    avg_price = config['avg_price']
    total_shares = config['total_shares']
    initial_capital = config['initial_capital']

    s = model.getVars()[0:len(layers)]  # Share variables
    p = model.getVars()[len(layers):]  # Price variables

    solution_data = {f"Layer {layer}": {'shares': round(s_var.X),
                                        'price': float(f"{p_var.X:.2f}")}
                     for layer, s_var, p_var in zip(layers, s, p)}

    # --- Prepare data for tabular display ---
    table_data = []
    total_loss_check = 0
    remaining_shares = total_shares
    preserved_capital_cumulative = 0

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

    # --- Display Probability and Verification ---
    print("\n" + "=" * 20 + " PROBABILITY ANALYSIS " + "=" * 20)
    stop_loss_prices = [val['price'] for val in solution_data.values()]
    stats_data, ohlc_paths = monte_carlo_stop_loss_probability(
        stop_loss=stop_loss_prices,
        time_horizon_days=[5, 10, 20],
        distribution_per_share=0.10,
        distribution_frequency_days=5,
        n_simulations=10000,
        volatility_method='garman_klass',
        intraday_steps=0  # Set to 390 for OHLC output
    )

    # Print results

    if stats_data:
        print(tabulate(stats_data, headers='keys', tablefmt='fancy_grid'))

    if ohlc_paths:
        print("Sample OHLC path (first simulation, first day):",
              ohlc_paths[0][0])

    print("\n" + "=" * 24 + " VERIFICATION " + "=" * 24)
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(
        f"Target Total Loss ({config['target_loss_percentage']:.0%}): ${config['target_loss']:,.2f}")
    print(f"Calculated Total Loss from Solution: ${total_loss_check:,.2f}")

    capital_at_c = sum(
        solution_data[f'Layer {l}']['shares'] * solution_data[f'Layer {l}'][
            'price'] for l in ['A', 'B', 'C'])
    print(
        f"\nCapital at C trigger: ${capital_at_c:,.2f} (must be >= ${config['initial_capital'] * config['target_at_c_layer']:.2f})")

    total_preserved_capital = sum(
        val['shares'] * val['price'] for val in solution_data.values())
    print(f"Total Preserved Capital: ${total_preserved_capital:,.2f}")
    print("\n" + "=" * 62)

    # --- Trigger Visualization ---
    visualize_stop_loss_strategy(solution_data, avg_price)


if __name__ == '__main__':
    # --- Configuration ---
    config = {
        'total_shares': 3280,
        'avg_price': 6.22,
        'target_loss_percentage': 0.10,
        'layers': ['A', 'B', 'C', 'D'],
        'target_at_c_layer': 0.65,
        'max_loss_per_layer': 1000,
        'ticker': 'ULTY'  # Ticker symbol for probability analysis
    }
    config['initial_capital'] = config['total_shares'] * config['avg_price']
    config['target_loss'] = config['initial_capital'] * config[
        'target_loss_percentage']

    # --- Run the process ---
    model = build_and_solve_model(config)
    process_and_display_results(model, config)
