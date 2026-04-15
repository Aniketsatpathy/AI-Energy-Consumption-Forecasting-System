from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_models(X_train, y_train):
    """
    Train and save models
    """

    # =========================
    # 1️⃣ Linear Regression (Baseline)
    # =========================
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # =========================
    # 2️⃣ Random Forest (TUNED)
    # =========================
    rf = RandomForestRegressor(
        n_estimators=200,     # More trees = better learning
        max_depth=10,         # Prevent overfitting
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1             # Use all CPU cores
    )

    rf.fit(X_train, y_train)

    # =========================
    # 3️⃣ Save models
    # =========================
    os.makedirs("models", exist_ok=True)

    joblib.dump(lr, "models/linear_regression.pkl")
    joblib.dump(rf, "models/random_forest.pkl")

    return lr, rf