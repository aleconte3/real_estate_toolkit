from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Union
import numpy as np
import statistics

@dataclass
class Descriptor:
    """Class for describing real estate data."""
    data: List[Dict[str, Any]]

    def none_ratio(self, columns: List[str] = "all") -> Dict[str, float]:
        """Compute the ratio of None value per column.
        If columns = "all", compute for all.
        Validate that column names are correct. If not, raise an exception.
        Return a dictionary with the key as the column name and value as the ratio of None values.
        """
        if columns == "all":
            columns = self.data[0].keys()  # Use the keys (column names) from the first row

        result = {}
        for column in columns:
            if column not in self.data[0]:
                raise ValueError(f"Column '{column}' does not exist in the dataset.")

            none_count = sum(1 for row in self.data if row.get(column) is None)
            total_count = len(self.data)
            result[column] = none_count / total_count

        return result

    def average(self, columns: List[str] = "all") -> Dict[str, float]:
        """Compute the average value for numeric variables. Omit None values.
        If columns = "all", compute for all numeric ones.
        Validate that column names are correct and correspond to a numeric variable.
        Return a dictionary with the key as the column name and value as the average.
        """
        if columns == "all":
            columns = [col for col in self.data[0].keys() if isinstance(self.data[0][col], (int, float))]

        result = {}
        for column in columns:
            if column not in self.data[0]:
                raise ValueError(f"Column '{column}' does not exist in the dataset.")
            
            values = [row[column] for row in self.data if row.get(column) is not None]
            if not values:
                result[column] = None
            else:
                result[column] = sum(values) / len(values)
        
        return result

    def median(self, columns: List[str] = "all") -> Dict[str, float]:
        """Compute the median value for numeric variables. Omit None values.
        If columns = "all", compute for all numeric ones.
        Validate that column names are correct and correspond to a numeric variable.
        Return a dictionary with the key as the column name and value as the median.
        """
        if columns == "all":
            columns = [col for col in self.data[0].keys() if isinstance(self.data[0][col], (int, float))]

        result = {}
        for column in columns:
            if column not in self.data[0]:
                raise ValueError(f"Column '{column}' does not exist in the dataset.")

            values = [row[column] for row in self.data if row.get(column) is not None]
            if not values:
                result[column] = None
            else:
                result[column] = statistics.median(values)
        
        return result

    def percentile(self, columns: List[str] = "all", percentile: int = 50) -> Dict[str, float]:
        """Compute the percentile value for numeric variables. Omit None values.
        If columns = "all", compute for all numeric ones.
        Validate that column names are correct and correspond to a numeric variable.
        Return a dictionary with the key as the column name and value as the percentile.
        """
        if columns == "all":
            columns = [col for col in self.data[0].keys() if isinstance(self.data[0][col], (int, float))]

        result = {}
        for column in columns:
            if column not in self.data[0]:
                raise ValueError(f"Column '{column}' does not exist in the dataset.")

            values = [row[column] for row in self.data if row.get(column) is not None]
            if not values:
                result[column] = None
            else:
                result[column] = np.percentile(values, percentile)
        
        return result

    def type_and_mode(self, columns: List[str] = "all") -> Dict[str, Union[Tuple[str, float], Tuple[str, str]]]:
        """Compute the mode for variables. Omit None values.
        If columns = "all", compute for all.
        Validate that column names are correct. If not, raise an exception.
        Return a dictionary with the key as the column name and value as a tuple of the variable type and the mode.
        """
        if columns == "all":
            columns = self.data[0].keys()

        result = {}
        for column in columns:
            if column not in self.data[0]:
                raise ValueError(f"Column '{column}' does not exist in the dataset.")

            values = [row[column] for row in self.data if row.get(column) is not None]
            if not values:
                result[column] = (None, None)
            else:
                # Calculate mode (most frequent value)
                mode_value = statistics.mode(values)
                column_type = type(values[0]).__name__  # Get the type of the column
                result[column] = (column_type, mode_value)
        
        return result