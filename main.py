from vanguard import VanguardFury
import csv
from typing import List, Dict, Any, Optional


def csv_to_list_of_dicts(csv_file_path: str) -> Optional[List[Dict[str, Any]]]:
    """
    Reads a CSV file and converts it into a list of dictionaries.
    - All keys (headers) are converted to lowercase.
    - All values are converted to float, except for keys 'date' and 'volume'.
    - If a value cannot be converted to float, it is set to None.

    Args:
        csv_file_path (str): The path to the input CSV file.

    Returns:
        Optional[List[Dict[str, Any]]]: A list of dictionaries representing
        the CSV data with appropriate types, or None if an error occurs.
    """
    data = []
    # Define the set of keys to exclude from float conversion (in lowercase)
    excluded_keys = {'date', 'volume'}

    try:
        with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)

            for row in csv_reader:
                processed_row = {}
                for key, value in row.items():
                    lower_key = key.lower()

                    # If the key is in our exclusion set, keep the value as a string
                    if lower_key in excluded_keys:
                        processed_row[lower_key] = value
                    # Otherwise, try to convert the value to a float
                    else:
                        try:
                            # Attempt the conversion
                            processed_row[lower_key] = float(value)
                        except (ValueError, TypeError):
                            # If conversion fails (e.g., empty string), set to None
                            processed_row[lower_key] = None

                data.append(processed_row)

    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    return data

historical_data = csv_to_list_of_dicts("price.csv")

fury = VanguardFury(ticker="ULTY", historical_price=historical_data)

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

stop_loss = fury.optimise_stop_loss(
    total_shares=3300,
    avg_price=6.22,
    target_loss_percentage=0.11,
    max_loss_per_layer=1000,
    price_upper_bound=5.80,
    minimum_shares_per_layer=100
)

print(stop_loss)
