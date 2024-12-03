import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from real_estate_toolkit.data.loader import load_data
from real_estate_toolkit.data.cleaner import clean_data
from real_estate_toolkit.data.descriptor import describe_data, plot_correlation

def main():
    # Load the data
    file_path = "data/train.csv"  # Ensure this path is correct
    data = load_data(file_path)
    print("Data loaded successfully!")
    print(data.head())

    # Clean the data
    cleaned_data = clean_data(data)
    print("Data cleaned successfully!")
    print(cleaned_data.head())

    # Descriptive statistics
    describe_data(cleaned_data)

    # Correlation heatmap
    plot_correlation(cleaned_data, target="SalePrice")

if __name__ == "__main__":
    main()

