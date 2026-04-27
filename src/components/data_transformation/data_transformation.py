import numpy as np
import pandas as pd
from src.logger import logger
from src.components.data_preprocessing.processor import DataPreprocessor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os

class DataTransformation(DataPreprocessor):
    """
    Handles data splitting, preprocessing, and transformation pipelines for ML modeling.
    Supports numerical, binary, and multi-category features.
    """
    def __init__(
        self,
        input_dir: str = "data/processed/final_data.parquet",
        train_dir: str = "data/processed/train_data.parquet",
        test_dir: str = "data/processed/test_data.parquet"
    ):
        super().__init__(input_dir=input_dir)
        self.train_dir = train_dir
        self.test_dir = test_dir

    def split_data(self, test_size: float = 0.2, random_state: int = 42):
        """Split dataset into training and testing sets, encode target labels."""
        data = super()._load_data(input_path=self.input_dir)
        data = data.drop(columns=['Label'])  # Drop original label column if exists
        logger.info("Splitting data into train and test sets")

        if 'AttackCategory' not in data.columns:
            raise ValueError("Target column 'AttackCategory' not found in dataset")

        

        # Encode target labels  
        train_data, test_data = train_test_split(
           data, test_size=test_size, random_state=random_state, stratify=data['AttackCategory']
        )
        logger.info(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")

        label_encoder = LabelEncoder()
        train_data['AttackCategory'] = label_encoder.fit_transform(train_data['AttackCategory'])
        test_data['AttackCategory'] = label_encoder.transform(test_data['AttackCategory'])
        logger.info(f"Target classes encoded: {list(label_encoder.classes_)}")

        #save train and test data
        super()._save_data(train_data, self.train_dir)
        super()._save_data(test_data, self.test_dir)
        return (
            train_data.drop(columns=['AttackCategory']),
            test_data.drop(columns=['AttackCategory']),
            train_data['AttackCategory'],
            test_data['AttackCategory']
        )

    
    @staticmethod
    def log_transform_func(x):
        """Apply log(1 + x) transformation."""
        return np.log1p(x)

    @staticmethod
    def numerical_pipeline():
        """Pipeline for numerical features: median imputation, log transform, scaling"""
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            # ('log', FunctionTransformer(DataTransformation.log_transform_func)),
            ('scaler', StandardScaler())
        ])
        return numeric_pipeline

    @staticmethod
    def binary_pipeline():
        """Pipeline for binary features: most frequent imputation + one-hot encoding"""
        binary_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=True, handle_unknown='ignore'))
        ])
        return binary_pipeline

    @staticmethod
    def categorical_pipeline():
        """Pipeline for multi-category features: most frequent imputation + one-hot encoding"""
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(sparse_output=True, handle_unknown='ignore',drop='first'))
        ])
        return categorical_pipeline

    def processor(self, numerical_cols, binary_cols, categorical_cols):
        """Combine all pipelines into a ColumnTransformer"""
        transformers = []
        if numerical_cols:
            transformers.append(('num', self.numerical_pipeline(), numerical_cols))
        if binary_cols:
            transformers.append(('bin', self.binary_pipeline(), binary_cols))
        if categorical_cols:
            transformers.append(('cat', self.categorical_pipeline(), categorical_cols))

        preprocessor = ColumnTransformer(transformers=transformers,remainder='passthrough')
        return preprocessor