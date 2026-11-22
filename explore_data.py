import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
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
filtered_melbourne_data = melbourne_data.dropna(axis=0)

y = filtered_melbourne_data.Price
melbourne_features = [
    'Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
    'YearBuilt', 'Lattitude', 'Longtitude'
]
X = filtered_melbourne_data[melbourne_features]

# =====================================================================
# STEP 3 & 4: TRAIN / TEST SPLIT
# =====================================================================
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# =====================================================================
# STEP 5: SINGLE DECISION TREE (BEST TUNED: 500 LEAVES)
# =====================================================================
tree_model = DecisionTreeRegressor(max_leaf_nodes=500, random_state=0)
tree_model.fit(train_X, train_y)
tree_preds = tree_model.predict(val_X)
tree_mae = mean_absolute_error(val_y, tree_preds)

# =====================================================================
# STEP 6: RANDOM FOREST (ENSEMBLE OF 100 TREES)
# =====================================================================
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
forest_preds = forest_model.predict(val_X)
forest_mae = mean_absolute_error(val_y, forest_preds)

print("================ MODEL COMPARISON ================")
print(f"Single Decision Tree (500 leaves) MAE : ${tree_mae:,.2f}")
print(f"Random Forest (100 trees) MAE         : ${forest_mae:,.2f}")
print(f"Improvement                           : ${tree_mae - forest_mae:,.2f} less error!")
print("==================================================")
