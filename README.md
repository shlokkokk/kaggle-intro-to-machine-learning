# Kaggle: Intro to Machine Learning

Code, notes, and competition submission from completing Kaggle's **Intro to Machine Learning** certification track.

---

## 🏆 Competition Achievement

* **Competition**: [Housing Prices Competition for Kaggle Learn Users](https://www.kaggle.com/competitions/home-data-for-ml-course)
* **Model**: `RandomForestRegressor`
* **Public Score (MAE)**: **`17,860.00818`**
* **Leaderboard Rank**: **#1,593** (out of 3,630+ participants worldwide)

---

## 📂 Repository Structure

```text
├── .gitignore          # Ignored files (caches, IDE config)
├── explore_data.py     # End-to-end ML pipeline (EDA -> Decision Tree -> Validation -> Random Forest)
├── NOTES.md            # Comprehensive concept-by-concept study guide & cheat sheet
├── README.md           # Project documentation and summary
└── data/
    ├── melb_data.csv   # Melbourne Housing dataset (practice data)
    └── submission.csv  # Final competition submission predictions
```

---

## 📚 What Was Covered

1. **Basic Data Exploration (EDA)**: Reading CSVs with `pandas`, statistical summary with `.describe()`, and handling missing values with `.dropna()`.
2. **First ML Model**: Splitting target variable `y` (SalePrice) and feature matrix `X`, training a `DecisionTreeRegressor`, and setting `random_state` for reproducibility.
3. **Model Validation**: The in-sample memorization trap (\$434 MAE) vs. honest out-of-sample evaluation using `train_test_split` (\$265k MAE).
4. **Underfitting vs. Overfitting**: Analyzing tree depth and finding the optimal `max_leaf_nodes` sweet spot.
5. **Random Forests**: Leveraging ensemble learning (100+ trees) to cut prediction errors down dramatically without hyperparameter tuning.
6. **Kaggle Competition**: Generating predictions on unseen test data (`test.csv`) and making a live submission scored on the global leaderboard.

---

## 🚀 How to Run

1. Make sure Python 3.x and required libraries are installed:
   ```bash
   pip install pandas scikit-learn
   ```

2. Run the main pipeline:
   ```bash
   python explore_data.py
   ```

3. Read the complete concept notes:
   Open [NOTES.md](NOTES.md) for in-depth explanations, code snippets, and machine learning intuition.
