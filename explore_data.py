import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Step 1: Load data
melbourne_file_path = 'data/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

# Step 2: Clean missing values
filtered_melbourne_data = melbourne_data.dropna(axis=0)

# Target y and Features X
y = filtered_melbourne_data.Price
melbourne_features = [
    'Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
    'YearBuilt', 'Lattitude', 'Longtitude'
]
X = filtered_melbourne_data[melbourne_features]

# Step 4: Model Validation (Train / Test Split)
# Split data into training (75%) and validation (25%)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Train ONLY on the training data
melbourne_model = DecisionTreeRegressor(random_state=0)
melbourne_model.fit(train_X, train_y)

# Test on the hidden validation data
val_predictions = melbourne_model.predict(val_X)
val_mae = mean_absolute_error(val_y, val_predictions)

print(f"Validation MAE: ${val_mae:,.2f}")
print("-> Out-of-sample error is ~$265k compared to in-sample ~$434 due to overfitting!")
