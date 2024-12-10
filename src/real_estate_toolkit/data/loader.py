from pathlib import Path
from typing import List, Dict, Union
import polars as pl

class DataLoader:
    """Class for loading and basic processing of real estate data."""
    def __init__(self, data_path: Path):
        self.data_path = data_path

    def load_data_from_csv(self) -> List[Dict[str, Union[str, float]]]:
        """Load data from CSV file into a list of dictionaries."""
        
        # Define schema overrides for problematic columns
        schema_overrides = {
            'MasVnrArea': pl.Utf8  # Treat 'MasVnrArea' as a string (Utf8) to support non-numeric values
        }

        # Read CSV with schema overrides, null values, and ignore errors
        df = pl.read_csv(self.data_path, 
                         separator=";",         # Specify separator
                         null_values=["None", "NA"],   # Handling missing values
                         schema_overrides=schema_overrides, 
                         infer_schema_length=10000,     # Increase the schema inference length
                         ignore_errors=True)            # Skip problematic rows

        # Converting Polars DataFrame into a list of dictionaries
        data = df.to_dicts()

        # Print out the dataframe head, columns, and dtypes for debugging
        print("Dataframe head:")
        print(df.head())
        print("Columns in dataset:", df.columns)
        print("Data types:", df.dtypes)

        return data

    def validate_columns(self, required_columns: List[str]) -> bool:
        """Validate that all required columns are present in the dataset."""
        # Load data to check columns
        df = pl.read_csv(self.data_path, separator=";", null_values=["None", "NA"])
        columns = df.columns
        
        # Print out the dataframe head, columns, and dtypes for debugging
        print("Columns in dataset:", columns)
        print("Data types:", df.dtypes)
        
        # Check if all required columns are present
        return all(col in columns for col in required_columns)

# Test the loader functionality
data_path = Path("/Users/aleconte/Documents/UNI/Upf/PROGRAMMING/real_estate_toolkit/data/train.csv")
loader = DataLoader(data_path)

# Call load_data_from_csv to load and print data
loader.load_data_from_csv()

# Validate columns in the dataset
required_columns = ["Id", "SalePrice", "LotArea", "YearBuilt", "BedroomAbvGr"]
is_valid = loader.validate_columns(required_columns)
print(f"All required columns are present: {is_valid}")