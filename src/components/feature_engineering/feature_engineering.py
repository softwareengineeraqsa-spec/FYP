from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.components.data_preprocessing.processor import DataPreprocessor
import numpy as np
import pandas as pd
from src.logger import logger
from src.utils.utils import save_report  
import matplotlib.pyplot as plt
import os


class FeatureEngineering(DataPreprocessor):
    def __init__(self, input_dir, output_dir="data/processed/final_data.parquet", vif_report_path="reports/vif_report.csv"):
        super().__init__(input_dir=input_dir, output_dir=output_dir, report_path=vif_report_path)

    def apply_vif(self, threshold: float = 5.0) -> pd.DataFrame:
        data = super()._load_data()
        logger.info("Starting VIF calculation")
        
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
        vif_data = data[numerical_cols].copy()
        
        while True:
            vif_values = [variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])]
            max_vif = max(vif_values)
            if max_vif < threshold:
                break
            else:
                drop_idx = vif_values.index(max_vif)
                logger.info(f"Dropping {vif_data.columns[drop_idx]} with VIF={max_vif:.2f}")
                vif_data = vif_data.drop(columns=vif_data.columns[drop_idx])

        # Save final VIF report
        vif_report = pd.DataFrame({"feature": vif_data.columns,
                                "VIF": [variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])]})
        save_report(vif_report, self.report_path, report_type="csv")

        final_data = pd.concat([vif_data, data[categorical_cols]], axis=1)
        super()._save_data(final_data)
        return self.output_dir, final_data


    def save_vif_plot(self, vif_df: pd.DataFrame, plot_path: str):
        """
        Saves a bar chart of VIF values.

        Parameters:
        - vif_df: pd.DataFrame with columns ['feature', 'VIF']
        - plot_path: str, path to save the plot (PNG)
        """
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        try:
            plt.figure(figsize=(10, 6))
            plt.bar(vif_df['feature'], vif_df['VIF'], color='skyblue')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel("VIF")
            plt.title("Variance Inflation Factor per Feature")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"VIF plot saved successfully to {plot_path}")
        except Exception as e:
            raise RuntimeError(f"Error saving VIF plot to {plot_path}: {e}")

