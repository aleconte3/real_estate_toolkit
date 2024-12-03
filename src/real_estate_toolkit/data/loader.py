import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file into a pandas DataFrame.

    :param file_path: Path to the CSV file.
    :return: DataFrame containing the loaded data.
    """
    return pd.read_csv(file_path, sep=";")