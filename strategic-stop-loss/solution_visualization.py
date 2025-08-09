import matplotlib.pyplot as plt
import numpy as np


def _plot_share_allocation(ax, layers, shares):
    """Plots the bar chart for share allocation."""
    bars = ax.bar(layers, shares, color='#4682B4', alpha=0.8)
    ax.set_title('Share Allocation per Layer', fontsize=14)
    ax.set_ylabel('Number of Shares Sold', fontsize=12)
    ax.bar_label(bars, fmt='%d')
    ax.grid(axis='y', linestyle='--', alpha=0.7)


def _plot_trigger_prices(ax, layers, prices, avg_price):
    """Plots the step chart for trigger prices."""
    price_points = [avg_price] + list(prices)
    price_labels = ['Avg. Purchase'] + layers
    ax.step(price_labels, price_points, where='post', color='#FF6347',
            linewidth=2)
    ax.plot(price_labels, price_points, 'o', color='#FF6347',
            markerfacecolor='white', markersize=8, mew=2)
    ax.set_title('Trigger Price Waterfall', fontsize=14)
    ax.set_ylabel('Price ($)', fontsize=12)
    for i, price in enumerate(price_points):
        ax.text(i, price + 0.03, f'${price:.2f}', ha='center')
    ax.set_xlim(left=-0.2, right=len(layers) + 0.2)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)


def _plot_loss_contribution(ax, layers, losses):
    """Plots the pie chart for loss contribution."""
    total_loss = np.sum(losses)
    ax.pie(
        losses,
        labels=layers,
        autopct=lambda p: f'${p / 100. * total_loss:.2f}\n({p:.1f}%)',
        startangle=140,
        colors=['#FFDDC1', '#FFC2B4', '#FFB3A6', '#FF9F99']
    )
    ax.set_title('Contribution to Total Loss', fontsize=14)


def _plot_capital_decay(ax, layers, capital_stages):
    """Plots the step chart for portfolio value decay."""
    waterfall_labels = ['Initial'] + layers
    ax.step(waterfall_labels, capital_stages, where='post', marker='o',
            color='#3CB371', linewidth=2)
    ax.set_title('Portfolio Value as Strategy Executes', fontsize=14)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    for i, val in enumerate(capital_stages):
        ax.text(i, val + 200, f'${val:,.2f}', ha='center')
    ax.set_ylim(bottom=min(capital_stages) * 0.95)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)


def visualize_stop_loss_strategy(solution_data, avg_price):
    """
    Visualizes the stop-loss strategy problem and solution.

    Args:
        solution_data (dict): The solution output from the Gurobi script.
        avg_price (float): The average purchase price of the shares.
    """
    # --- 1. Process Data for Plotting ---
    layers = [key.replace('Layer ', '') for key in solution_data.keys()]
    shares = np.array([d['shares'] for d in solution_data.values()])
    prices = np.array([d['price'] for d in solution_data.values()])
    total_shares = np.sum(shares)
    initial_capital = avg_price * total_shares

    losses = shares * (avg_price - prices)

    # Calculate the portfolio value at each stage of the strategy
    capital_stages = [initial_capital]
    cumulative_cash = 0
    remaining_shares_at_stage = total_shares
    for i in range(len(layers)):
        cumulative_cash += shares[i] * prices[i]
        remaining_shares_at_stage -= shares[i]
        value_of_remaining_shares = remaining_shares_at_stage * prices[i]
        capital_stages.append(cumulative_cash + value_of_remaining_shares)

    # --- 2. Create Visualization Dashboard ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(2, 2, figsize=(17, 13))
    fig.suptitle('Visualization of the Strategic Stop-Loss Solution',
                 fontsize=22, y=0.98)

    # --- 3. Call Plotting Functions ---
    _plot_share_allocation(axs[0, 0], layers, shares)
    _plot_trigger_prices(axs[0, 1], layers, prices, avg_price)
    _plot_loss_contribution(axs[1, 0], layers, losses)
    _plot_capital_decay(axs[1, 1], layers, capital_stages)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# This block allows you to test the visualization independently
if __name__ == '__main__':
    # Define plausible sample data for direct testing
    sample_solution_data = {
        'Layer A': {'shares': 818, 'price': 5.81},
        'Layer B': {'shares': 819, 'price': 5.69},
        'Layer C': {'shares': 821, 'price': 5.52},
        'Layer D': {'shares': 822, 'price': 5.35}
    }
    sample_avg_price = 6.22

    print("Running visualization with sample data...")
    visualize_stop_loss_strategy(sample_solution_data, sample_avg_price)
