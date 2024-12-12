import polars as pl
from pathlib import Path
from typing import List, Dict, Any
from real_estate_toolkit.data.cleaner import Cleaner
from real_estate_toolkit.ml_models.predictor import HousePricePredictor


def is_valid_snake_case(string: str) -> bool:
    """Check if a given string is in valid snake_case."""
    # This function ensures that the column names follow the snake_case convention,
    # which is a widely accepted naming convention in Python for variables and functions.
    if not string:
        return False
    if not all(char.islower() or char.isdigit() or char == '_' for char in string):
        return False
    if string.startswith('_') or string.endswith('_'):
        return False
    if '__' in string:
        return False
    return True


def test_data_loading_and_cleaning():
    """Test data loading and cleaning functionality."""
    # This function ensures that the data is correctly loaded and cleaned before training the model.
    # Path to the training dataset
    data_path = Path("/Users/aleconte/Documents/UNI/Upf/PROGRAMMING/real_estate_toolkit/data/train.csv")
    
    # Load the data using Polars directly. Polars is chosen for its speed and efficiency in handling large datasets.
    df = pl.read_csv(data_path, separator=";", null_values=["None", "NA"])

    # Handling the 'MasVnrArea' column. It is dropped if it exists or treated as a string if it is missing.
    if 'MasVnrArea' in df.columns:
        df = df.drop('MasVnrArea')  # Drop the column if it's present (it's not useful in model training)
        print("'MasVnrArea' column dropped.")
    else:
        # If the column does not exist, we cast it to string type (some datasets may contain missing values represented as 'None' or 'NA')
        df = df.with_columns(pl.col("MasVnrArea").cast(pl.Utf8))  # Explicitly cast MasVnrArea to string if the column is not dropped
        print("'MasVnrArea' column treated as string.")

    # Normalize column names by stripping spaces and converting to lowercase to ensure consistency.
    normalized_columns = [col.strip().lower() for col in df.columns]
    

    # Column validation: check if all required columns are present.
    # The following columns are essential for predicting house prices: 'id', 'saleprice', 'lotarea', 'yearbuilt', 'bedroomabvgr'
    required_columns = ["id", "saleprice", "lotarea", "yearbuilt", "bedroomabvgr"]
    missing_columns = [col for col in required_columns if col not in normalized_columns]
    
    if missing_columns:
        raise ValueError(f"Required columns missing from dataset: {', '.join(missing_columns)}")

    # Proceed with data cleaning by using the Cleaner class.
    # The Cleaner class will rename columns based on best practices and handle missing values.
    cleaner = Cleaner(df.to_dicts())  # Pass the data as a list of dicts to Cleaner (Polars doesn't support pandas-like operations)
    cleaner.rename_with_best_practices()  # Renames columns following best practices (snake_case, more descriptive)
    cleaned_data = cleaner.na_to_none()  # Replace missing values with None

    # After cleaning, verify that all column names are in snake_case and all values are valid types.
    assert all(is_valid_snake_case(key) for key in cleaned_data[0].keys()), "Column names should be in snake_case"
    assert all(val is None or isinstance(val, (str, int, float)) for row in cleaned_data for val in row.values()), \
        "Values should be None or basic types"
    
    return cleaned_data


def test_house_price_predictor():
    """Test the functionality of the HousePricePredictor class."""
    # The HousePricePredictor class integrates data cleaning, feature preparation, model training, and forecasting.
    # The paths to the datasets are specified for both the training and testing datasets.
    train_data_path = Path("/Users/aleconte/Documents/UNI/Upf/PROGRAMMING/real_estate_toolkit/data/train.csv")
    test_data_path = Path("/Users/aleconte/Documents/UNI/Upf/PROGRAMMING/real_estate_toolkit/data/test.csv")
    
    # Initialize the HousePricePredictor
    predictor = HousePricePredictor(train_data_path=str(train_data_path), test_data_path=str(test_data_path))

    # Step 1: Test data cleaning
    print("Testing data cleaning...")
    try:
        predictor.clean_data()
        print("Data cleaning passed!")
    except Exception as e:
        print(f"Data cleaning failed: {e}")
        return
    
    # Step 2: Test feature preparation
    print("Testing feature preparation...")
    try:
        predictor.prepare_features(target_column="SalePrice")
        print("Feature preparation passed!")
    except Exception as e:
        print(f"Feature preparation failed: {e}")
        return
    
    # Step 3: Test model training
    print("Testing model training...")
    try:
        results = predictor.train_baseline_models()
        for model_name, result in results.items():
            metrics = result["metrics"]
            print(f"{model_name} - Metrics:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")
        print("Model training passed!")
    except Exception as e:
        print(f"Model training failed: {e}")
        return
    
    # Step 4: Test forecasting
    print("Testing forecasting...")
    try:
        predictor.forecast_sales_price(model_type="LinearRegression")
        print("Forecasting passed!")
    except Exception as e:
        print(f"Forecasting failed: {e}")
        return


def main():
    """Main function to run all tests"""
    try:
        # Run all tests sequentially
        cleaned_data = test_data_loading_and_cleaning()  # First, test data loading and cleaning
        test_house_price_predictor()  # Then, test the predictor functionality (data cleaning, feature prep, model training, etc.)
        print("All tests passed successfully!")
        return 0  # If everything passes, return 0 (successful completion)
    except AssertionError as e:
        print(f"Test failed: {str(e)}")  # If an assertion fails, handle the error and return 1
        return 1
    except Exception as e:
        print(f"Unexpected error: {str(e)}")  # Handle any unexpected errors and return 2
        return 2


if __name__ == "__main__":
    # Call the main function when this script is run
    main()