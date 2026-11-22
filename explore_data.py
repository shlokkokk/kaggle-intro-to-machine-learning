import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# =====================================================================
# STEP 1: LOAD DATA
# =====================================================================
melbourne_file_path = 'data/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

# =====================================================================
# STEP 2: CLEAN DATA & SELECT FEATURES
# =====================================================================
# Filter out rows with missing values (dropna)
filtered_melbourne_data = melbourne_data.dropna(axis=0)

# Target y (what we want to predict)
y = filtered_melbourne_data.Price

# Features X (clues used to predict y)
melbourne_features = [
    'Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
    'YearBuilt', 'Lattitude', 'Longtitude'
]
X = filtered_melbourne_data[melbourne_features]

# =====================================================================
# STEP 3 & 4: SPLIT DATA INTO TRAIN AND VALIDATION SETS
# =====================================================================
# 75% for training (studying), 25% for validation (the test)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# =====================================================================
# STEP 5: TUNE TREE SIZE (UNDERFITTING VS OVERFITTING)
# =====================================================================
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    """Builds, trains, and evaluates a decision tree with a given leaf limit."""
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae

print("--- Comparing Candidate Tree Sizes ---")
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
scores = {}
for max_leaves in candidate_max_leaf_nodes:
    my_mae = get_mae(max_leaves, train_X, val_X, train_y, val_y)
    scores[max_leaves] = my_mae
    print(f"Max Leaves: {max_leaves:<5} | Validation MAE: ${my_mae:,.2f}")

# Find the leaf size with the lowest MAE
best_tree_size = min(scores, key=scores.get)
print(f"\nWinning Tree Size: {best_tree_size} leaves (Lowest MAE: ${scores[best_tree_size]:,.2f})")

# =====================================================================
# FINAL STEP: TRAIN FINAL MODEL ON ALL DATA (FOR PRODUCTION)
# =====================================================================
# Once the optimal tree size is known, retrain on 100% of the data
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)
final_model.fit(X, y)
print("\nFinal production model successfully trained on 100% of dataset!")
