import os
import json
import pandas as pd
from src.logger import logger

def save_report(report_data, report_path: str, report_type: str = "json"):
    """
    Saves a report to the specified path in JSON or CSV format.

    Parameters:
    - report_data: dict (for JSON) or pd.DataFrame (for CSV)
    - report_path: str, where the report will be saved
    - report_type: str, "json" or "csv" (default "json")
    """
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    try:
        if report_type.lower() == "json":
            if not isinstance(report_data, dict):
                raise TypeError("For JSON report, report_data must be a dict")
            with open(report_path, "w") as f:
                json.dump(report_data, f, indent=4)
        elif report_type.lower() == "csv":
            if not isinstance(report_data, pd.DataFrame):
                raise TypeError("For CSV report, report_data must be a pandas DataFrame")
            report_data.to_csv(report_path, index=False)
        else:
            raise ValueError("report_type must be either 'json' or 'csv'")
        logger.info(f"Report saved successfully to {report_path}")
    except Exception as e:
        raise RuntimeError(f"Error saving report to {report_path}: {e}")
