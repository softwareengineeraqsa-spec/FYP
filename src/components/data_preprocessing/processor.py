import pandas as pd
import numpy as np
from src.logger import logger
from src.utils.utils import save_report  
import os

class DataPreprocessor:
    def __init__(
        self,
        input_dir: str,
        output_dir: str = "data/processed/processed_data.parquet",
        report_path: str = "reports/preprocessing_report.json",
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.report_path = report_path

    def process_data(self) -> pd.DataFrame:
        data = self._load_data()
        logger.info("Starting data preprocessing")

        report = {"initial_shape": data.shape}

        # Drop duplicates
        duplicates = data.duplicated().sum()
        data = data.drop_duplicates().reset_index(drop=True)
        report["duplicates_removed"] = int(duplicates)

        # Handle missing values
        total_missing = int(data.isnull().sum().sum())
        report["missing_before"] = total_missing
        if total_missing > 0:
            data = self.handle_missing_values(data)
        report["missing_after"] = int(data.isnull().sum().sum())

        # Column stats
        report["column_summary"] = self._generate_column_summary(data)
        report["final_shape"] = data.shape

        # Save processed data
        self._save_data(data)
        # Save preprocessing report
        save_report(report, self.report_path, report_type="json")

        logger.info("Data preprocessing complete")
        return self.output_dir, data

    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        num_cols = data.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            data[col].fillna(data[col].mean(), inplace=True)

        cat_cols = data.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            data[col].fillna(data[col].mode()[0], inplace=True)

        return data

    def _load_data(self, input_path=None) -> pd.DataFrame:
        if input_path is None:
            input_path = self.input_dir
        return pd.read_parquet(input_path)

    def _save_data(self, data: pd.DataFrame, output_path=None):
        if output_path is None:
            output_path = self.output_dir
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data.to_parquet(output_path, index=False)

    def _generate_column_summary(self, df: pd.DataFrame) -> dict:
        summary = {}
        for col in df.columns:
            summary[col] = {
                "dtype": str(df[col].dtype),
                "missing_values": int(df[col].isnull().sum()),
                "unique_values": int(df[col].nunique()),
            }
            if np.issubdtype(df[col].dtype, np.number):
                summary[col]["mean"] = float(df[col].mean())
                summary[col]["std"] = float(df[col].std())
        return summary
