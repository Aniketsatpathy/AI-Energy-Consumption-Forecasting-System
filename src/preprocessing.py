import pandas as pd

def preprocess_data(df):
    """
    Full preprocessing pipeline for energy dataset
    """

    # Combine Date + Time → timestamp
    df['timestamp'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        format='%d/%m/%Y %H:%M:%S'
    )

    # Drop original columns
    df = df.drop(['Date', 'Time'], axis=1)

    # Convert target column
    df['Global_active_power'] = pd.to_numeric(
        df['Global_active_power'],
        errors='coerce'
    )

    # Drop missing values
    df = df.dropna()

    # Rename
    df = df.rename(columns={
        'Global_active_power': 'energy'
    })

    # Set index
    df = df.set_index('timestamp')

    # ✅ FIXED HERE (IMPORTANT)
    df = df.resample('h').mean()

    # Reset index
    df = df.reset_index()

    # Sort
    df = df.sort_values('timestamp')

    return df


# TEST BLOCK
if __name__ == "__main__":
    from data_loader import load_data

    print("Loading data...")
    df = load_data("data/raw/household_power_consumption.txt")

    print("Running preprocessing...")
    df = preprocess_data(df)

    # Keep only necessary columns
    df = df[['timestamp', 'energy']]
    
    print("\nProcessed Data:")
    print(df.head())
    
    print("\nShape:", df.shape)

    