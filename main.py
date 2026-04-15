import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.feature_engineering import create_features
from src.train_model import train_models
from src.evaluate import evaluate_model
from src.forecast import forecast
from src.utils import create_dirs


def main():

    print("🚀 Starting Energy Forecasting Pipeline...\n")

    # =========================
    # 1️⃣ Setup
    # =========================
    create_dirs()

    # =========================
    # 2️⃣ Load Data
    # =========================
    print("📥 Loading data...")
    df = load_data("data/raw/household_power_consumption.txt")
    print("Raw Shape:", df.shape)

    # =========================
    # 3️⃣ Preprocessing
    # =========================
    print("\n🧹 Preprocessing data...")
    df = preprocess_data(df)
    print("After Preprocessing:", df.shape)

    # =========================
    # 4️⃣ Feature Engineering
    # =========================
    print("\n🧠 Creating features...")
    df = create_features(df)
    print("After Feature Engineering:", df.shape)

    print("\nColumns:", df.columns.tolist())

    # =========================
    # 5️⃣ Split Data (Time-Based)
    # =========================
    X = df.drop(['energy', 'timestamp'], axis=1)
    y = df['energy']

    split = int(len(df) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print("\nTrain Size:", X_train.shape)
    print("Test Size:", X_test.shape)

    # =========================
    # 6️⃣ Train Models
    # =========================
    print("\n🤖 Training models...")
    lr, rf = train_models(X_train, y_train)

    # =========================
    # 7️⃣ Predictions
    # =========================
    print("\n🔮 Generating predictions...")
    y_pred_lr = lr.predict(X_test)
    y_pred_rf = rf.predict(X_test)

    # =========================
    # 8️⃣ Evaluation
    # =========================
    print("\n📊 Evaluating models...")

    rmse_lr, r2_lr = evaluate_model(y_test, y_pred_lr, "Linear Regression")
    rmse_rf, r2_rf = evaluate_model(y_test, y_pred_rf, "Random Forest")

    # =========================
    # 9️⃣ Forecast (Next 24 Hours)
    # =========================
    print("\n📈 Forecasting next 24 hours...")
    future_preds = forecast(rf, X_test, steps=24)

    forecast_df = pd.DataFrame(future_preds, columns=["forecast"])
    forecast_df.to_csv("outputs/predictions.csv", index=False)

    print("Forecast saved to outputs/predictions.csv")

    # =========================
    # 🔟 Visualization
    # =========================
    print("\n📉 Creating visualization...")

    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label="Actual")
    plt.plot(y_pred_rf, label="Predicted")
    plt.legend()
    plt.title("Actual vs Predicted Energy Consumption")

    plt.savefig("images/actual_vs_predicted.png")
    plt.show()

    print("\n✅ Pipeline Completed Successfully!")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()