from src.logger import logger
from src.components.data_ingestion.data_ingestion import DataIngestion
from src.components.data_preprocessing.processor import DataPreprocessor
from src.components.feature_engineering.feature_engineering import FeatureEngineering
from src.components.data_transformation.data_transformation import DataTransformation
from src.components.model_training.model_training import ModelTrainer
from src.components.model_evulation.model_evaluation import ModelEvaluation

def main():
    logger.info("Starting the malware-detection program")
    # apply data ingestion
    # data_ingestion = DataIngestion(file_path='temp_data/final_dataset.parquet',output_dir='data/raw/raw_data.parquet')
    # raw_data_path, raw_data = data_ingestion.read_data()
    # logger.info("Data ingestion completed successfully")

    # # apply data preprocessing
    # data_preprocessor = DataPreprocessor(input_dir=raw_data_path, output_dir='data/processed/processed_data.parquet', report_path='reports/preprocessing_report.json')
    # processed_data_path, processed_data = data_preprocessor.process_data()
    # logger.info("Data preprocessing completed successfully")
    # # apply feature engineering
    # feature_engineering = FeatureEngineering(input_dir=processed_data_path, output_dir='data/processed/final_data.parquet', vif_report_path='reports/vif_report.csv')
    # final_data_path, final_data = feature_engineering.apply_vif(threshold=5.0)
    # logger.info("Feature engineering completed successfully")

    # do model training
    logger.info("Starting model training")
    trainer = ModelTrainer()
    trainer.train(classifier="RandomForest", classifier_params={"n_estimators": 100, "max_depth": 10})
    logger.info("Model training completed successfully")

    logger.info("Cross validation and evaluation")
    trainer.cross_validate(classifier="RandomForest", classifier_params={"n_estimators": 100, "max_depth": 10})

    logger.info("Starting model evaluation")
    evaluator = ModelEvaluation(
        model_path="models/ddos_model.pkl",
        test_data_path="data/processed/test_data.parquet"
    )

    metrics = evaluator.evaluate()

if __name__ == "__main__":
    main()
