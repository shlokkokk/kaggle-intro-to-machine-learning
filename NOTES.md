# Machine Learning Course - Concept Notes & Knowledge Base

This document contains a structured, concept-by-concept record of everything covered so far in the Kaggle Machine Learning track.

---

## 1. Exploratory Data Analysis (EDA) & Pandas Basics

* **Loading Data**: `pd.read_csv('filepath.csv')` reads CSV files into a 2-dimensional tabular structure called a **DataFrame**.
* **Understanding `.describe()`**:
  * **`count`**: Number of valid, non-missing values.
  * **`mean`**: Mathematical average ($\sum x / N$).
  * **`std`**: Standard deviation (measures how widely spread the values are around the mean).
  * **`min` / `max`**: Smallest and largest values.
  * **`25%`, `50%`, `75%`**: Percentiles (1st Quartile, Median, 3rd Quartile).
* **Scientific Notation**:
  * `1.075684e+06` means $1.075684 \times 10^6 = \$1,075,684$.
* **Missing Data**:
  * If `BuildingArea` has 7,130 rows while total rows are 13,580, missing values exist.
  * `df.dropna(axis=0)` drops rows that have missing values (`axis=0` means rows).

---

## 2. Choosing the Prediction Target (`y`) and Features (`X`)

In machine learning and statistics:
* **`y` (Target / Lowercase 1D Series)**:
  * The single variable you want the model to learn to predict.
  * Example: `y = melbourne_data.Price`
* **`X` (Features / Uppercase 2D DataFrame)**:
  * The table of clues/attributes used to make the prediction.
  * Example: `melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'YearBuilt', ...]`
  * `X = melbourne_data[melbourne_features]`

---

## 3. Building Your First Model (The 4 Universal Steps)

Scikit-learn (`sklearn`) follows the same pattern across all algorithms:

1. **Define**: Choose the algorithm and hyperparameters.
   ```python
   from sklearn.tree import DecisionTreeRegressor
   model = DecisionTreeRegressor(random_state=1)
   ```
   * *What is `random_state`?* Sets the random seed so that tie-breaking splits remain identical across runs, ensuring reproducible results.
2. **Fit**: The model studies the historical data and learns patterns.
   ```python
   model.fit(X, y)
   ```
3. **Predict**: Feed in house features to receive predicted prices.
   ```python
   predictions = model.predict(X)
   ```
4. **Evaluate**: Measure the prediction error.

---

## 4. Model Validation & The "In-Sample" Trap

* **MAE (Mean Absolute Error)**:
  $$\text{MAE} = \frac{1}{N} \sum |\text{Actual} - \text{Predicted}|$$
  *Plain English:* On average, how many dollars off is our guess?
* **The "In-Sample" Fallacy**:
  * Testing a model on the same data it learned from gives an artificially low error (e.g. \$434 MAE).
  * The model didn't learn general truth; it just **memorized** the answer key.
* **Train / Test Split (`train_test_split`)**:
  * To measure real-world performance, split the dataset before training:
  ```python
  from sklearn.model_selection import train_test_split
  train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
  ```
  * `train_X, train_y`: 75% of data used to train the model.
  * `val_X, val_y`: 25% hidden data used solely for the final exam.
  * Out-of-sample MAE jumped to **\$265,345**, revealing true performance on unseen homes.

---

## 5. Underfitting vs. Overfitting (Tree Depth Optimization)

* **Tree Depth and Leaves**:
  * A decision tree splits data repeatedly into branches.
  * The terminal endpoints are **leaves** (where final predictions are made).
* **Underfitting**:
  * Tree is too shallow (e.g., 5 leaves).
  * Groups vastly different homes together; fails to learn real patterns.
  * High error on both training and validation data.
* **Overfitting**:
  * Tree is too deep (e.g., 5,000 leaves).
  * Each leaf has 1-2 homes; memorizes random noise (e.g., door color).
  * Perfect on training data, terrible on validation data.
* **The Sweet Spot (`max_leaf_nodes`)**:
  * We use a loop over candidate leaf counts `[5, 25, 50, 100, 250, 500]` to test validation MAE.
  * The sweet spot occurs at **100 leaves** (Iowa data) / **500 leaves** (Melbourne data).
* **Final Model Retraining**:
  * After determining the optimal leaf count, retrain on **100% of data (`X, y`)** for maximum knowledge in production.

---

## 6. Random Forests (Ensemble Learning)

* **The Concept**:
  * Instead of relying on a single tree (which easily underfits or overfits), a **Random Forest** builds dozens or hundreds of decision trees (by default 100).
  * It takes the **average** of all their individual predictions ("Wisdom of the Crowd").
* **Why "Random"?**:
  * Each tree trains on a random subsample of the data and a random subset of features.
  * Individual tree biases cancel out, dramatically improving accuracy.
* **Scikit-Learn Implementation**:
  ```python
  from sklearn.ensemble import RandomForestRegressor
  rf_model = RandomForestRegressor(random_state=1)
  rf_model.fit(train_X, train_y)
  rf_preds = rf_model.predict(val_X)
  rf_mae = mean_absolute_error(val_y, rf_preds)
  ```
* **Performance Gain**:
  * On the Iowa dataset, MAE dropped from **\$27,282** (best single tree) down to **\$21,857** (Random Forest) — an instant \$5,400+ improvement with zero hyperparameter tuning required!
