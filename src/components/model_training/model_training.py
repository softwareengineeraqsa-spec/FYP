import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from src.components.data_transformation.data_transformation import DataTransformation
from src.logger import logger


class ModelTrainer(DataTransformation):
    """
    Handles model training, evaluation, visualization, and saving.
    """

    def __init__(
        self,
        model_dir: str = "models/",
        report_dir: str = "reports/",
        model_name: str = "ddos_model.pkl"
    ):
        super().__init__(
            input_dir="data/processed/final_data.parquet",
            train_dir="data/processed/train_data.parquet",
            test_dir="data/processed/test_data.parquet"
        )
        self.model_dir = model_dir
        self.report_dir = report_dir
        self.plot_dir = os.path.join(report_dir, "plots")
        self.model_name = model_name
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

    # ========== MODEL PIPELINE ==========

    def model_pipeline(
        self,
        preprocessor,
        classifier: str = "RandomForest",
        classifier_params: dict = None
    ) -> Pipeline:
        """Create full pipeline with preprocessing and classifier."""
        if classifier_params is None:
            classifier_params = {}

        if classifier in ["RandomForest", "LogisticRegression"]:
            classifier_params.setdefault("class_weight", "balanced")

        if classifier == "RandomForest":
            model = RandomForestClassifier(**classifier_params, random_state=42)
        elif classifier == "LogisticRegression":
            model = LogisticRegression(**classifier_params, max_iter=1000, random_state=42)
        else:
            raise ValueError(f"Unsupported classifier: {classifier}")

        return Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

    # ========== TRAINING ==========

    def train(
        self,
        classifier: str = "RandomForest",
        classifier_params: dict = None
    ):
        """Train, evaluate, visualize, and save model and reports."""
        X_train, X_test, y_train, y_test = self.split_data()

        all_features = X_train.columns.tolist()
        binary_cols = [c for c in all_features if X_train[c].nunique() == 2]
        numerical_cols = [c for c in all_features if X_train[c].dtype in [np.int64, np.float64] and c not in binary_cols]
        categorical_cols = [c for c in all_features if c not in numerical_cols + binary_cols]

        preprocessor = self.processor(numerical_cols, binary_cols, categorical_cols)
        pipeline = self.model_pipeline(preprocessor, classifier, classifier_params)

        logger.info(f"Training {classifier} model with class_weight='balanced'")
        pipeline.fit(X_train, y_train)

        # Evaluation
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Save model
        model_path = os.path.join(self.model_dir, self.model_name)
        joblib.dump(pipeline, model_path)
        logger.info(f"Model saved at {model_path}")

        # Save metrics
        metrics_report = {
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": conf_matrix.tolist()
        }
        report_path = os.path.join(self.report_dir, self.model_name.replace(".pkl", "_report.json"))
        pd.Series(metrics_report).to_json(report_path)

        # Visualization
        self._save_confusion_matrix(conf_matrix, acc)
        self._save_roc_curve(pipeline, X_test, y_test)
        logger.info("All evaluation plots saved.")

        return pipeline, metrics_report

    # ========== CROSS-VALIDATION ==========

    def cross_validate(
        self,
        classifier: str = "RandomForest",
        classifier_params: dict = None,
        cv_folds: int = 5
    ):
        """Perform detailed cross-validation capturing all metrics."""
        X_train, X_test, y_train, y_test = self.split_data()

        all_features = X_train.columns.tolist()
        binary_cols = [c for c in all_features if X_train[c].nunique() == 2]
        numerical_cols = [c for c in all_features if X_train[c].dtype in [np.int64, np.float64] and c not in binary_cols]
        categorical_cols = [c for c in all_features if c not in numerical_cols + binary_cols]

        preprocessor = self.processor(numerical_cols, binary_cols, categorical_cols)
        pipeline = self.model_pipeline(preprocessor, classifier, classifier_params)

        logger.info(f"Running {cv_folds}-fold cross-validation with full metrics for {classifier}")

        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision_macro',
            'recall': 'recall_macro',
            'f1': 'f1_macro'
        }

        results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv_folds,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )

        # Compute mean and std for each metric
        summary = {
            metric: {
                'train_mean': np.mean(results[f"train_{metric}"]),
                'train_std': np.std(results[f"train_{metric}"]),
                'test_mean': np.mean(results[f"test_{metric}"]),
                'test_std': np.std(results[f"test_{metric}"])
            }
            for metric in scoring.keys()
        }

        df_results = pd.DataFrame(results)
        summary_df = pd.DataFrame(summary).T

        # Save reports
        report_path = os.path.join(self.report_dir, f"{classifier.lower()}_cv_results.csv")
        summary_path = os.path.join(self.report_dir, f"{classifier.lower()}_cv_summary.csv")
        df_results.to_csv(report_path, index=False)
        summary_df.to_csv(summary_path)
        logger.info(f"Cross-validation results saved at {report_path} and summary at {summary_path}")

        # Plot all metrics across folds
        self._save_cv_metrics_plot(df_results)

        return {
            "summary": summary_df.to_dict(),
            "raw": df_results.to_dict(orient="list")
        }

    # ========== VISUALIZATION HELPERS ==========

    def _save_confusion_matrix(self, conf_matrix, accuracy):
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix (Accuracy: {accuracy:.2f})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        path = os.path.join(self.plot_dir, "train_confusion_matrix.png")
        plt.savefig(path)
        plt.close()
        logger.info(f"Confusion matrix saved at {path}")

    def _save_roc_curve(self, pipeline, X_test, y_test):
        try:
            if len(np.unique(y_test)) == 2:
                y_prob = pipeline.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)

                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
                plt.plot([0, 1], [0, 1], "r--")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve")
                plt.legend()
                plt.tight_layout()
                path = os.path.join(self.plot_dir, "train_roc_curve.png")
                plt.savefig(path)
                plt.close()
                logger.info(f"ROC curve saved at {path}")
        except Exception as e:
            logger.warning(f"Skipping ROC curve: {e}")

    def _save_cv_metrics_plot(self, df_results):
        """Plot test metric trends across folds for deeper performance insight."""
        plt.figure(figsize=(10, 6))
        metrics = [col for col in df_results.columns if col.startswith("test_")]
        for metric in metrics:
            plt.plot(
                range(1, len(df_results[metric]) + 1),
                df_results[metric],
                marker="o",
                label=metric.replace("test_", "").capitalize()
            )

        plt.title("Cross-Validation Metrics Across Folds")
        plt.xlabel("Fold")
        plt.ylabel("Score")
        plt.legend()
        plt.tight_layout()
        path = os.path.join(self.plot_dir, "train_cv_metrics_across_folds.png")
        plt.savefig(path)
        plt.close()
        logger.info(f"Cross-validation metrics plot saved at {path}")
