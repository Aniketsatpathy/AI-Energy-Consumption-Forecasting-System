import pandas as pd

def create_features(df):
    """
    Advanced feature engineering for time-series forecasting
    """

    # =========================
    # 0️⃣ Keep only required columns
    # =========================
    df = df[['timestamp', 'energy']]

    # Ensure sorted order
    df = df.sort_values('timestamp')

    # =========================
    # 1️⃣ TIME FEATURES
    # =========================
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month

    # =========================
    # 2️⃣ LAG FEATURES (CORE MEMORY)
    # =========================
    df['lag_1'] = df['energy'].shift(1)
    df['lag_24'] = df['energy'].shift(24)
    df['lag_168'] = df['energy'].shift(168)

    # =========================
    # 3️⃣ ROLLING FEATURES (TREND AWARENESS)
    # =========================
    df['rolling_mean_24'] = df['energy'].rolling(window=24).mean()
    df['rolling_std_24'] = df['energy'].rolling(window=24).std()

    # =========================
    # 4️⃣ INTERACTION FEATURE (PATTERN BOOST)
    # =========================
    df['hour_weekday'] = df['hour'] * df['day_of_week']

    # =========================
    # 5️⃣ DROP NULL VALUES
    # =========================
    df = df.dropna()

    return df