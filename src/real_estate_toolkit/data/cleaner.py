from dataclasses import dataclass
from typing import Dict, List, Any
import re  # For converting to snake_case

@dataclass
class Cleaner:
    """Class for cleaning real estate data."""
    data: List[Dict[str, Any]]

    def rename_with_best_practices(self) -> None:
        """Rename the columns with best practices (e.g., snake_case for descriptive names)."""
        # Iterate over each dictionary in the dataset (each row)
        for row in self.data:
            new_row = {}
            for col_name, value in row.items():
                # Rename the column to snake_case
                new_col_name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', col_name).lower()  # Converts to snake_case
                new_row[new_col_name] = value
            # Replace the row with the new column names
            self.data[self.data.index(row)] = new_row

    def na_to_none(self) -> List[Dict[str, Any]]:
        """Replace "NA" with None in all values with "NA" in the dictionary."""
        for row in self.data:
            for col_name, value in row.items():
                if value == "NA":  # If the value is "NA", replace it with None
                    row[col_name] = None
        return self.data