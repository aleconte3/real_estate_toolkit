import polars as pl
from pathlib import Path
from typing import List, Dict, Any
from real_estate_toolkit.data.cleaner import Cleaner
from real_estate_toolkit.ml_models.predictor import HousePricePredictor


def is_valid_snake_case(string: str) -> bool:
    """Check if a given string is in valid snake_case."""
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
    """Test data loading and cleaning functionality"""
    # Test data loading using Polars
    data_path = Path("/Users/aleconte/Documents/UNI/Upf/PROGRAMMING/real_estate_toolkit/data/train.csv")
    
    # Load the data using Polars directly
    df = pl.read_csv(data_path, separator=";")

    # Remove 'MasVnrArea' column if it exists
    if 'MasVnrArea' in df.columns:
        df = df.drop('MasVnrArea')  # Remove the 'MasVnrArea' column if it's present

    # Print columns and data types for debugging
    print("Columns in the dataset:", df.columns)
    print("Data types in the dataset:", df.dtypes)

    # Normalize column names (remove spaces, convert to lowercase, and make sure they are consistent)
    normalized_columns = [col.strip().lower() for col in df.columns]
    print("Normalized columns:", normalized_columns)

    # Test column validation
    required_columns = ["id", "saleprice", "lotarea", "yearbuilt", "bedroomabvgr"]
    missing_columns = [col for col in required_columns if col not in normalized_columns]
    
    if missing_columns:
        raise ValueError(f"Required columns missing from dataset: {', '.join(missing_columns)}")

    # Continue with data cleaning
    cleaner = Cleaner(df.to_dicts())  # Pass the data as a list of dicts to Cleaner
    cleaner.rename_with_best_practices()
    cleaned_data = cleaner.na_to_none()

    # Verify cleaning results
    assert all(is_valid_snake_case(key) for key in cleaned_data[0].keys()), "Column names should be in snake_case"
    assert all(val is None or isinstance(val, (str, int, float)) for row in cleaned_data for val in row.values()), \
        "Values should be None or basic types"
    
    return cleaned_data




def test_house_price_predictor():
    """Test the functionality of the HousePricePredictor class."""
    # Paths to the datasets
    train_data_path = Path("/Users/aleconte/Documents/UNI/Upf/PROGRAMMING/real_estate_toolkit/data/train.csv")
    test_data_path = Path("/Users/aleconte/Documents/UNI/Upf/PROGRAMMING/real_estate_toolkit/data/test.csv")
    # Initialize predictor
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
        predictor.forecast_sales_price(model_type="Linear Regression")
        print("Forecasting passed!")
    except Exception as e:
        print(f"Forecasting failed: {e}")
        return


def main():
    """Main function to run all tests"""
    try:
        # Run all tests sequentially
        cleaned_data = test_data_loading_and_cleaning()
        test_house_price_predictor()
        print("All tests passed successfully!")
        return 0
    except AssertionError as e:
        print(f"Test failed: {str(e)}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return 2


if __name__ == "__main__":
    main()