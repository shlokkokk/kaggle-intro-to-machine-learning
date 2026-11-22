import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Step 1: Load data
melbourne_file_path = 'data/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

# Step 2: Clean missing values (dropna)
filtered_melbourne_data = melbourne_data.dropna(axis=0)

# Target y and Features X
y = filtered_melbourne_data.Price
melbourne_features = [
    'Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
    'YearBuilt', 'Lattitude', 'Longtitude'
]
X = filtered_melbourne_data[melbourne_features]

# Step 3: Build first Decision Tree Model
melbourne_model = DecisionTreeRegressor(random_state=1)
melbourne_model.fit(X, y)

# In-sample predictions
in_sample_predictions = melbourne_model.predict(X)
in_sample_mae = mean_absolute_error(y, in_sample_predictions)

print("=== In-Sample Predictions vs Actual (First 5) ===")
print("Actual:   ", [f"${p:,.0f}" for p in y.head().values])
print("Predicted:", [f"${p:,.0f}" for p in in_sample_predictions[:5]])
print(f"In-Sample MAE: ${in_sample_mae:,.2f} (Artificially low due to memorization!)")
