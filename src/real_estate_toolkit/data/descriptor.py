import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def describe_data(df: pd.DataFrame):
    """
    Prints basic descriptive statistics for the DataFrame.

    :param df: DataFrame to describe.
    """
    print("Descriptive statistics:")
    print(df.describe())

def plot_correlation(df: pd.DataFrame, target: str):
    """
    Plots a correlation heatmap for the DataFrame using only numeric columns.

    :param df: DataFrame to analyze.
    :param target: Target column for correlation analysis.
    """
    # Select numeric columns only
    numeric_df = df.select_dtypes(include=["float64", "int64"])

    # Compute the correlation matrix
    correlation = numeric_df.corr()

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=False, cmap="coolwarm")
    plt.title(f"Correlation Heatmap with {target}")
    plt.show()