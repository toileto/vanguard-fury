import gurobipy as gp
from tabulate import tabulate
from gurobipy import GRB
from black_scholes import calculate_price_probability
from solution_visualization import visualize_stop_loss_strategy

# --- Parameters ---
total_shares = 3280
avg_price = 6.22
initial_capital = total_shares * avg_price
target_loss_percentage = 0.10
target_loss = initial_capital * target_loss_percentage
layers = ['A', 'B', 'C', 'D']
target_at_c_layer = 0.65

# --- Create Gurobi Model ---
m = gp.Model("StopLossStrategy")

# --- Decision Variables ---
# s: shares per layer (Integer), with a lower bound of 1
s = m.addVars(layers, vtype=GRB.INTEGER, name="s", lb=100)
# p: price per layer (Continuous)
p = m.addVars(layers, vtype=GRB.CONTINUOUS, lb=0.01, name="p")

# --- Objective Function: Maximize total revenue from sales ---
m.setObjective(
    gp.quicksum(s[i] * p[i] for i in layers),
    sense=GRB.MAXIMIZE
)

# --- Constraints ---
m.addConstr(
    gp.quicksum(s[i] for i in layers) == total_shares, "TotalShares")
m.addConstr(s['B'] >= s['A'] + 1, "Hierarchy_B")
m.addConstr(s['C'] >= s['B'] + 1, "Hierarchy_C")
m.addConstr(s['D'] >= s['C'] + 1, "Hierarchy_D")
m.addConstr(
    gp.quicksum(s[i] * (avg_price - p[i]) for i in layers) == target_loss,
    "CumulativeLoss")

m.addConstr((s['A'] * p['A']) + (s['B'] * p['B']) +
            (s['C'] * p['C']) >= target_at_c_layer * initial_capital,
            "CapitalPreservation")

m.addConstr(p['A'] <= 5.85, "PriceUpperBound_A")

max_loss_per_layer = 1000
for i in layers:
    m.addConstr(
        s[i] * (avg_price - p[i]) <= max_loss_per_layer,
        f"MaxLoss_{i}"
    )

    # m.addConstr(p[i] <= 5.85, f"MinimumLimit_{i}")

# # Enforce p_B is at least 5% lower than p_A
m.addConstr(p['B'] <= p['A'] * 0.98, "PriceGap_AB_5pct")
#
# # Enforce p_C is at least 5% lower than p_B
m.addConstr(p['C'] <= p['B'] * 0.96, "PriceGap_BC_5pct")
#
# # Enforce p_D is at least 5% lower than p_C (THIS IS THE CORRECTED LINE)
m.addConstr(p['D'] <= p['C'] * 0.97, "PriceGap_CD_5pct")

# --- Solve the model ---
m.setParam('NonConvex', 2)
# With these tighter constraints, we can turn off presolve if needed, but it's usually fine
# m.setParam('Presolve', 0)
m.optimize()

# --- Print the results and Verification ---
if m.Status == GRB.OPTIMAL:
    print("\n--- Optimal Stop-Loss Strategy ---\n")

    total_loss_check = 0
    all_data = list()
    remaining_shares = total_shares
    preserve_capital = 0
    final = dict()
    for i in layers:
        shares = round(s[i].X)
        price = float("{:.2f}".format(p[i].X))
        preserve_capital += (shares * price)
        loss_for_layer = shares * (avg_price - price)
        total_loss_check += loss_for_layer
        remaining_shares -= shares

        _new_data = {
            "Layer": i,
            "TotalShares": shares,
            "TriggerPrice": price,
            "LossForLayer": loss_for_layer,
            "PreservedCapital": preserve_capital,
            "RemainingShares": remaining_shares,
            "MarketValue": remaining_shares * price
        }

        final[f"Layer {i}"] = {
            "shares": shares,
            "price": price
        }

        all_data.append(_new_data)

    print(tabulate(all_data, headers="keys", tablefmt="fancy_grid"))

    print("\n" + "=" * 20 + " PROBABILITY " + "=" * 20)

    calculate_price_probability([float(final[i]['price']) for i in final])
    visualize_stop_loss_strategy(final)

    print("\n" + "=" * 20 + " VERIFICATION " + "=" * 20)
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Target Total Loss (10%): ${target_loss:.2f}")
    print(f"Calculated Total Loss from Solution: ${total_loss_check:.2f}")

    print("\n--- Price Gaps ---")
    p_A = p['A'].X
    p_B = p['B'].X
    p_C = p['C'].X
    p_D = p['D'].X
    gap_A = (p_A - avg_price) / avg_price
    gap_B = (p_B - avg_price) / avg_price
    gap_C = (p_C - avg_price) / avg_price
    gap_D = (p_D - avg_price) / avg_price
    print(f"Price Gap A: {gap_A:.2%}")
    print(f"Price Gap B: {gap_B:.2%}")
    print(f"Price Gap C: {gap_C:.2%}")
    print(f"Price Gap D: {gap_D:.2%}")

    capital_at_C = (
            s['A'].X * p['A'].X +
            s['B'].X * p['B'].X +
            s['C'].X * p['C'].X
    )
    print(
        f"\nCapital at C trigger: ${capital_at_C:.2f}"
        f" (must be >= ${initial_capital * target_at_c_layer:.2f})")

    print("\n" + "=" * 54)

    preserved_capital = (
            s['A'].X * p['A'].X +
            s['B'].X * p['B'].X +
            s['C'].X * p['C'].X +
            s['D'].X * p['D'].X
    )

    print(f"Preserved Capital: ${preserved_capital:.2f}")

else:
    print("\nSolver did not find an optimal solution.")
    print(f"Gurobi optimization status code: {m.Status}")
