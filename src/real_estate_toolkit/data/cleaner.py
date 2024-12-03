import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataset:
    - Handles missing values.
    - Drops columns with too many missing values.

    :param df: Original DataFrame.
    :return: Cleaned DataFrame.
    """
    # Drop columns with more than 50% missing values
    threshold = len(df) * 0.5
    df = df.dropna(axis=1, thresh=threshold)

    # Fill remaining missing values
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        df.loc[:, col] = df[col].fillna(df[col].median())  # Use .loc[] to avoid SettingWithCopyWarning

    for col in df.select_dtypes(include=["object"]).columns:
        df.loc[:, col] = df[col].fillna("Unknown")  # Use .loc[] to avoid SettingWithCopyWarning

    return df