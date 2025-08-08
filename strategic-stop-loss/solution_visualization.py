import matplotlib.pyplot as plt
import numpy as np

def visualize_stop_loss_strategy(solution_data):
    """
    Visualizes the stop-loss strategy problem and solution.

    NOTE: The solution data is hardcoded based on a plausible outcome
    from your Gurobi script. You should replace the values in the
    'solution_data' dictionary with the actual output from your script.
    """
    # --- 1. Parameters and Solution Data ---

    # Initial Problem Parameters (from your gurobi script)
    avg_price = 6.22
    total_shares = 3280
    initial_capital = avg_price * total_shares

    # --- Replace this with your actual Gurobi output ---
    # Plausible Solution Data
    # solution_data = {
    #     'Layer A': {'shares': 818, 'price': 5.81},
    #     'Layer B': {'shares': 819, 'price': 5.69},
    #     'Layer C': {'shares': 821, 'price': 5.52},
    #     'Layer D': {'shares': 822, 'price': 5.35}
    # }
    # ----------------------------------------------------

    # --- 2. Process Data for Plotting ---
    layers = list(solution_data.keys())
    shares = np.array([d['shares'] for d in solution_data.values()])
    prices = np.array([d['price'] for d in solution_data.values()])
    losses = shares * (avg_price - prices)
    total_loss = np.sum(losses)

    # Data for the capital decay waterfall chart
    capital_stages = [initial_capital]
    remaining_shares_at_stage = total_shares
    for i in range(len(layers)):
        cash_from_sales = np.sum(shares[:i+1] * prices[:i+1])
        remaining_shares_at_stage -= shares[i]
        value_of_remaining_shares = remaining_shares_at_stage * prices[i]
        capital_stages.append(cash_from_sales + value_of_remaining_shares)

    # --- 3. Create Visualization Dashboard ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(2, 2, figsize=(17, 13))
    fig.suptitle('Visualization of the Strategic Stop-Loss Solution', fontsize=22, y=0.98)

    # --- PLOT 1: Share Allocation ---
    bars = axs[0, 0].bar(layers, shares, color='#4682B4', alpha=0.8)
    axs[0, 0].set_title('Share Allocation per Layer', fontsize=14)
    axs[0, 0].set_ylabel('Number of Shares Sold', fontsize=12)
    axs[0, 0].bar_label(bars, fmt='%d')

    # --- PLOT 2: Trigger Prices (MODIFIED TO WATERFALL/STEP PLOT) ---
    price_points = [avg_price] + list(prices)
    price_labels = ['Avg. Purchase'] + layers
    # Using a step plot creates the waterfall visualization
    axs[0, 1].step(price_labels, price_points, where='post', color='#FF6347', linewidth=2)
    axs[0, 1].plot(price_labels, price_points, 'o', color='#FF6347', markerfacecolor='white', markersize=8, mew=2) # Overlay markers
    axs[0, 1].set_title('Trigger Price Waterfall', fontsize=14)
    axs[0, 1].set_ylabel('Price ($)', fontsize=12)
    axs[0, 1].grid(True, which='both', linestyle='--', linewidth=0.5)
    for i, price in enumerate(price_points):
        axs[0, 1].text(i, price + 0.03, f'${price:.2f}', ha='center')
    axs[0, 1].set_xlim(left=-0.2, right=len(layers) + 0.2) # Add some padding

    # --- PLOT 3: Loss Contribution ---
    wedges, texts, autotexts = axs[1, 0].pie(
        losses,
        labels=layers,
        autopct=lambda p: f'${p/100.*total_loss:.2f}\n({p:.1f}%)',
        startangle=140,
        colors=['#FFDDC1', '#FFC2B4', '#FFB3A6', '#FF9F99']
    )
    axs[1, 0].set_title('Contribution to Total Loss', fontsize=14)
    plt.setp(autotexts, size=10, weight="bold")


    # --- PLOT 4: Capital Decay Waterfall ---
    waterfall_labels = ['Initial'] + layers
    axs[1, 1].step(waterfall_labels, capital_stages, where='post', marker='o', color='#3CB371', linewidth=2)
    axs[1, 1].set_title('Portfolio Value as Strategy Executes', fontsize=14)
    axs[1, 1].set_ylabel('Portfolio Value ($)', fontsize=12)
    axs[1, 1].grid(True, which='both', linestyle='--', linewidth=0.5)
    for i, val in enumerate(capital_stages):
        axs[1, 1].text(i, val + 200, f'${val:,.2f}', ha='center')
    axs[1, 1].set_ylim(bottom=min(capital_stages) * 0.95)

    # Show the plots
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Run the visualization function
if __name__ == '__main__':
    visualize_stop_loss_strategy()
