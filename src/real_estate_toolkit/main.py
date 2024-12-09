import re
from pathlib import Path
from typing import List, Dict, Any
import polars as pl

from real_estate_toolkit.data.loader import DataLoader
from real_estate_toolkit.data.cleaner import Cleaner
from real_estate_toolkit.data.descriptor import Descriptor
from real_estate_toolkit.data.descriptor_numpy import DescriptorNumpy
from real_estate_toolkit.agent_based_model.houses import House, QualityScore
from real_estate_toolkit.agent_based_model.market import HousingMarket
from real_estate_toolkit.agent_based_model.consumers import Consumer, Segment
from real_estate_toolkit.agent_based_model.simulation import (
    Simulation, 
    CleaningMarketMechanism, 
    AnnualIncomeStatistics,
    ChildrenRange
)
from real_estate_toolkit.analytics.exploratory import MarketAnalyzer
from real_estate_toolkit.ml_models.predictor import HousePricePredictor


class DataLoader:
    def __init__(self, file_path: Path, delimiter: str = ","):
        self.file_path = file_path
        self.delimiter = delimiter

    def load_data_from_csv(self) -> List[Dict[str, Any]]:
        """
        Load data from a CSV file with the specified delimiter and handle `NA` as null.
        Increase `infer_schema_length` to ensure better type inference.
        """
        try:
            df = pl.read_csv(
                self.file_path,
                separator=self.delimiter,
                null_values=["NA"],  # Treat "NA" as null
                infer_schema_length=10000  # Increase the number of rows used to infer the schema
            )
            return df.to_dicts()
        except Exception as e:
            raise RuntimeError(f"Error loading data from {self.file_path}: {e}")

    def validate_columns(self, required_columns: List[str]) -> bool:
        """
        Validate if required columns are in the dataset.
        """
        data = self.load_data_from_csv()
        if not data:
            return False
        dataset_columns = set(data[0].keys())
        return all(column in dataset_columns for column in required_columns)


class Cleaner:
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def fill_missing_with_mode(self, columns: List[str]):
        """
        Fill missing values in the specified columns with the mode (most common value).
        """
        for column in columns:
            mode_value = self._calculate_mode(column)
            for row in self.data:
                if row[column] is None:
                    row[column] = mode_value

    def _calculate_mode(self, column: str):
        """
        Calculate the mode (most common value) for a column.
        """
        values = [row[column] for row in self.data if row[column] is not None]
        return max(set(values), key=values.count)

    def rename_with_best_practices(self):
        """
        Rename column names to follow best practices like snake_case.
        """
        def to_snake_case(name: str) -> str:
            return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()

        for row in self.data:
            snake_case_row = {to_snake_case(k): v for k, v in row.items()}
            row.clear()
            row.update(snake_case_row)

    def na_to_none(self):
        """
        Replace NA or missing values with None in the dataset.
        """
        for row in self.data:
            for key, value in row.items():
                if value in ["NA", None]:
                    row[key] = None
        return self.data


def is_valid_snake_case(string: str) -> bool:
    """
    Check if a given string is in valid snake_case.
    """
    if not string:
        return False
    if not all(char.islower() or char.isdigit() or char == '_' for char in string):
        return False
    if string.startswith('_') or string.endswith('_') or '__' in string:
        return False
    return True


def test_data_loading_and_cleaning():
    """Test data loading and cleaning functionality."""
    data_path = Path("/Users/aleconte/Documents/UNI/Upf/PROGRAMMING/real_estate_toolkit/data/train.csv")
    loader = DataLoader(data_path, delimiter=";")  # Specifica il delimitatore
    required_columns = ["Id", "SalePrice", "LotArea", "YearBuilt", "BedroomAbvGr"]
    assert loader.validate_columns(required_columns), "Required columns missing from dataset"

    data = loader.load_data_from_csv()
    assert isinstance(data, list), "Data should be returned as a list"
    assert all(isinstance(row, dict) for row in data), "Each row should be a dictionary"

    cleaner = Cleaner(data)
    cleaner.rename_with_best_practices()
    print("Column names after renaming:", list(cleaner.data[0].keys()))  # Debug
    cleaned_data = cleaner.na_to_none()

    columns_to_fill_with_mode = [
        "bsmt_fin_type1", "bsmt_fin_type2", "heating", "heating_q_c",
        "central_air", "electrical", "kitchen_qual", "functional",
        "fireplace_qu"
    ]
    cleaner.fill_missing_with_mode(columns_to_fill_with_mode)

    assert all(is_valid_snake_case(key) for key in cleaned_data[0].keys()), "Column names should be in snake_case"
    assert all(val is None or isinstance(val, (str, int, float)) for row in cleaned_data for val in row.values()), \
        "Values should be None or basic types"
    return cleaned_data

def test_descriptive_statistics(cleaned_data: List[Dict[str, Any]]):
    """Test descriptive statistics functionality."""
    # Converti i dati in un DataFrame di Polars
    cleaned_df = pl.DataFrame(cleaned_data)

    descriptor = Descriptor(cleaned_df)
    descriptor_numpy = DescriptorNumpy(cleaned_df)

    none_ratios = descriptor.none_ratio()
    none_ratios_numpy = descriptor_numpy.none_ratio()
    assert isinstance(none_ratios, dict), "None ratios should be returned as dictionary"
    assert set(none_ratios.keys()) == set(none_ratios_numpy.keys()), "Both implementations should handle same columns"

    numeric_columns = ["sale_price", "lot_area"]
    averages = descriptor.average(numeric_columns)
    medians = descriptor.median(numeric_columns)
    percentiles = descriptor.percentile(numeric_columns, 75)

    averages_numpy = descriptor_numpy.average(numeric_columns)
    medians_numpy = descriptor_numpy.median(numeric_columns)
    percentiles_numpy = descriptor_numpy.percentile(numeric_columns, 75)

    for col in numeric_columns:
        assert abs(averages[col] - averages_numpy[col]) < 1e-6, f"Average calculations differ for {col}"
        assert abs(medians[col] - medians_numpy[col]) < 1e-6, f"Median calculations differ for {col}"
        assert abs(percentiles[col] - percentiles_numpy[col]) < 1e-6, f"Percentile calculations differ for {col}"

    type_modes = descriptor.type_and_mode()
    type_modes_numpy = descriptor_numpy.type_and_mode()
    assert set(type_modes.keys()) == set(type_modes_numpy.keys()), "Both implementations should handle same columns"

    return numeric_columns


def main():
    """Main function to run all tests."""
    try:
        cleaned_data = test_data_loading_and_cleaning()
        test_descriptive_statistics(cleaned_data)
        print("All tests passed successfully!")
    except AssertionError as e:
        print(f"Test failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()