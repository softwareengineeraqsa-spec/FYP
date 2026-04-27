import pandas as pd
import numpy as np
from src.logger import logger

class DataIngestion:
    def __init__(
            self,
            file_path: str,
            delimiter: str = ',',
            output_dir: str = 'data/raw/raw_data.parquet',
    ):
        self.file_path = file_path
        self.output_dir = output_dir

    def read_data(self) -> pd.DataFrame:
        """Reads data from a parquet file into a pandas DataFrame."""
        logger.info(f"Reading data from {self.file_path}")
        try:
            data = pd.read_parquet(self.file_path)

            # save the data
            self.save_data(data)
            logger.info(f"Data reading complete from {self.file_path}")
            return self.output_dir, data
        except Exception as e:
            raise RuntimeError(f"Error reading data from {self.file_path}: {e}")
        
    def save_data(self, data: pd.DataFrame) -> None:
        """Saves the DataFrame to a Parquet file."""
        logger.info(f"Saving data to {self.output_dir}")
        try:
            data.to_parquet(self.output_dir, index=False)
            logger.info(f"Data saving complete to {self.output_dir}")
        except Exception as e:
            raise RuntimeError(f"Error saving data to {self.output_dir}: {e}")