import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Union

@dataclass
class DescriptorNumpy:
    """Class for describing real estate data using NumPy."""
    data: np.ndarray  # The data is now a NumPy ndarray instead of a list of dictionaries

    def none_ratio(self, columns: List[str] = "all") -> Dict[str, float]:
        """Compute the ratio of None (or np.nan) values per column.
        If columns = "all", compute for all.
        Validate that column names are correct. If not, raise an exception.
        Return a dictionary with the column name and the ratio of None (or np.nan) values.
        """
        if columns == "all":
            columns = range(self.data.shape[1])  # Use all column indices if "all" is passed

        result = {}
        for column in columns:
            if column >= self.data.shape[1]:
                raise ValueError(f"Column index {column} does not exist in the dataset.")
            
            none_count = np.isnan(self.data[:, column]).sum()  # Count np.nan values
            total_count = self.data.shape[0]
            result[column] = none_count / total_count

        return result

    def average(self, columns: List[str] = "all") -> Dict[str, float]:
        """Compute the average value for numeric variables. Omit None (or np.nan) values.
        If columns = "all", compute for all numeric ones.
        Return a dictionary with the column name and its average.
        """
        if columns == "all":
            columns = range(self.data.shape[1])  # Use all column indices if "all" is passed

        result = {}
        for column in columns:
            if column >= self.data.shape[1]:
                raise ValueError(f"Column index {column} does not exist in the dataset.")
            
            values = self.data[:, column]
            valid_values = values[~np.isnan(values)]  # Remove np.nan values
            if valid_values.size == 0:
                result[column] = np.nan
            else:
                result[column] = np.mean(valid_values)
        
        return result

    def median(self, columns: List[str] = "all") -> Dict[str, float]:
        """Compute the median value for numeric variables. Omit None (or np.nan) values.
        If columns = "all", compute for all numeric ones.
        Return a dictionary with the column name and its median.
        """
        if columns == "all":
            columns = range(self.data.shape[1])  # Use all column indices if "all" is passed

        result = {}
        for column in columns:
            if column >= self.data.shape[1]:
                raise ValueError(f"Column index {column} does not exist in the dataset.")

            values = self.data[:, column]
            valid_values = values[~np.isnan(values)]  # Remove np.nan values
            if valid_values.size == 0:
                result[column] = np.nan
            else:
                result[column] = np.median(valid_values)

        return result

    def percentile(self, columns: List[str] = "all", percentile: int = 50) -> Dict[str, float]:
        """Compute the percentile value for numeric variables. Omit None (or np.nan) values.
        If columns = "all", compute for all numeric ones.
        Return a dictionary with the column name and its percentile value.
        """
        if columns == "all":
            columns = range(self.data.shape[1])  # Use all column indices if "all" is passed

        result = {}
        for column in columns:
            if column >= self.data.shape[1]:
                raise ValueError(f"Column index {column} does not exist in the dataset.")

            values = self.data[:, column]
            valid_values = values[~np.isnan(values)]  # Remove np.nan values
            if valid_values.size == 0:
                result[column] = np.nan
            else:
                result[column] = np.percentile(valid_values, percentile)

        return result

    def type_and_mode(self, columns: List[str] = "all") -> Dict[str, Union[Tuple[str, float], Tuple[str, str]]]:
        """Compute the mode for variables. Omit None (or np.nan) values.
        If columns = "all", compute for all.
        Return a dictionary with the column name and a tuple containing the type and mode.
        """
        if columns == "all":
            columns = range(self.data.shape[1])  # Use all column indices if "all" is passed

        result = {}
        for column in columns:
            if column >= self.data.shape[1]:
                raise ValueError(f"Column index {column} does not exist in the dataset.")

            values = self.data[:, column]
            valid_values = values[~np.isnan(values)]  # Remove np.nan values
            if valid_values.size == 0:
                result[column] = (None, None)
            else:
                mode_value = np.bincount(valid_values.astype(int)).argmax()  # Compute mode (most frequent value)
                column_type = str(valid_values.dtype)  # Get the type of the column
                result[column] = (column_type, mode_value)

        return result