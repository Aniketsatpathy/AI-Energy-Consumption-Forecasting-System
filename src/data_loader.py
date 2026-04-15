import pandas as pd

def load_data(file_path):
    """
    Load dataset with correct separator and parsing
    """
    df = pd.read_csv(
        file_path,
        sep=';',              # VERY IMPORTANT
        na_values=['?'],      # Missing values
        low_memory=False
    )
    return df

df = load_data("data/raw/household_power_consumption.txt")

print(df.head())
print(df.columns)