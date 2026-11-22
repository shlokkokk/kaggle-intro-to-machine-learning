import pandas as pd

# Step 1: Load data
melbourne_file_path = 'data/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

# Step 2: Clean missing values (dropna)
filtered_melbourne_data = melbourne_data.dropna(axis=0)

# Choose target variable (y = what we want to predict)
y = filtered_melbourne_data.Price

# Choose feature columns (X = clues to make predictions)
melbourne_features = [
    'Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
    'YearBuilt', 'Lattitude', 'Longtitude'
]
X = filtered_melbourne_data[melbourne_features]

print("Features selected for X:")
print(X.head())
print("\nTarget y (Price):")
print(y.head())
