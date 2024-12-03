from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union
import polars as pl  # Assicurati di avere il pacchetto Polars installato

@dataclass
class DataLoader:
    """Class for loading and basic processing of real estate data."""
    data_path: Path

    def load_data_from_csv(self) -> List[Dict[str, Union[str, float]]]:
        """Load data from CSV file into a list of dictionaries."""
        # Carica i dati dal CSV usando Polars
        df = pl.read_csv(self.data_path)
        
        # Converte il DataFrame di Polars in una lista di dizionari
        data = df.to_dicts()

        return data

    def validate_columns(self, required_columns: List[str]) -> bool:
        """Validate that all required columns are present in the dataset."""
        # Carica i dati dal CSV per verificare le colonne
        df = pl.read_csv(self.data_path)
        
        # Recupera le colonne del dataframe
        columns = df.columns
        
        # Verifica che tutte le colonne richieste siano nel dataset
        return all(col in columns for col in required_columns)