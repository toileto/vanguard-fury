from stop_loss import StopLossOptimizer

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

    config = {
        'ticker': "ULTY",
        'total_shares': 3300,
        'avg_price': 6.22,
        'target_loss_percentage': 0.10,
        'layers': ['A', 'B', 'C', 'D'],
        'capital_preservation': True,
        'capital_preservation_layer': "C",
        'capital_preservation_target': 0.65,
        'max_loss_per_layer': 1000,
        'minimum_shares_per_layer': 100,
        'price_upper_bound': 5.85,
        'layer_steps': [0.02, 0.04, 0.03],
    }

    model = StopLossOptimizer("ULTY")
    solution = model.build_and_solve_model(config)
