import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from src.logger import logger

class ModelEvaluation:
    """
    Evaluates a trained model on test data, computes metrics, saves reports and plots.
    """

    def __init__(
        self,
        model_path: str,
        test_data_path: str,
        report_dir: str = "reports/",
        plot_dir: str = "reports/plots/"
    ):
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.report_dir = report_dir
        self.plot_dir = plot_dir

        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        self.model = self._load_model()
        self.X_test, self.y_test = self._load_test_data()

    # === Load model and test data ===
    def _load_model(self):
        try:
            model = joblib.load(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model from {self.model_path}: {e}")
            raise

    def _load_test_data(self):
        try:
            df = pd.read_parquet(self.test_data_path)
            X_test = df.drop("AttackCategory", axis=1)
            y_test = df["AttackCategory"]

            # Drop rows with missing labels
            mask = y_test.notna()
            X_test = X_test[mask]
            y_test = y_test[mask]

            logger.info(f"Test data loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")
            return X_test, y_test
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            raise


    # === Evaluate model ===
    def evaluate(self):
        y_pred = self.model.predict(self.X_test)

        metrics = {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "precision": precision_score(self.y_test, y_pred, average="weighted"),
            "recall": recall_score(self.y_test, y_pred, average="weighted"),
            "f1": f1_score(self.y_test, y_pred, average="weighted"),
            "classification_report": classification_report(self.y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(self.y_test, y_pred).tolist()
        }

        logger.info(f"Evaluation metrics: {metrics}")
        self._save_report(metrics)
        self._save_confusion_matrix(metrics["confusion_matrix"])
        self._save_roc_curve()
        return metrics

    # === Save JSON report ===
    def _save_report(self, metrics: dict):
        test_report_path = os.path.join(self.report_dir, os.path.basename(self.model_path).replace(".pkl", "test_report.json"))
        pd.Series(metrics).to_json(test_report_path)
        logger.info(f"Evaluation report saved at {test_report_path}")

    # === Save confusion matrix plot ===
    def _save_confusion_matrix(self, conf_matrix):
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        path = os.path.join(self.plot_dir, "test_confusion_matrix.png")
        plt.savefig(path)
        plt.close()
        logger.info(f"Confusion matrix saved at {path}")

    # === Save ROC curve plot (binary only) ===
    def _save_roc_curve(self):
        if len(np.unique(self.y_test)) == 2:
            try:
                y_prob = self.model.predict_proba(self.X_test)[:, 1]
                fpr, tpr, _ = roc_curve(self.y_test, y_prob)
                roc_auc = auc(fpr, tpr)

                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
                plt.plot([0, 1], [0, 1], "r--")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve")
                plt.legend()
                path = os.path.join(self.plot_dir, "test_roc_curve.png")
                plt.savefig(path)
                plt.close()
                logger.info(f"ROC curve saved at {path}")
            except Exception as e:
                logger.warning(f"Skipping ROC curve: {e}")
