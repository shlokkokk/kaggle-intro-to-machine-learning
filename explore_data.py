import pandas as pd

# Step 1: Load data
melbourne_file_path = 'data/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

# Exploratory Data Analysis (EDA)
print("--- Columns in Dataset ---")
print(melbourne_data.columns)

print("\n--- Summary Statistics ---")
print(melbourne_data.describe())
