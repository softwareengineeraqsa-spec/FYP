
# Malware Detection Project Documentation

## 1. Project Overview

This project is designed to detect DDoS (Distributed Denial-of-Service) attacks from network traffic data. It implements a complete machine learning pipeline, from data ingestion and preprocessing to model training, evaluation, and deployment via a REST API. The core of the project is a classification model that distinguishes between benign and malicious (DDoS) network traffic.

## 2. Project Structure

The project is organized into the following directories:

```
.
├── data/                # Raw, processed, and temporary data
├── logs/                # Log files
├── models/              # Trained machine learning models
├── notebooks/           # Jupyter notebooks for exploratory data analysis
├── reports/             # Reports and plots from model training and evaluation
├── src/                 # Source code
│   ├── api/             # FastAPI application for model serving
│   ├── components/      # Core ML pipeline components
│   │   ├── data_ingestion/
│   │   ├── data_preprocessing/
│   │   ├── data_transformation/
│   │   ├── feature_engineering/
│   │   ├── model_evaluation/
│   │   └── model_training/
│   ├── logger/          # Logging configuration
│   └── utils/           # Utility functions
├── main.py              # Main script to run the ML pipeline
├── test.py              # Test scripts
├── pyproject.toml       # Project dependencies
└── README.md            # Project README
```

## 3. Installation and Setup

To set up the project, you need to have Python installed. The dependencies are listed in `pyproject.toml`. You can install them using a package manager like `pip`.

```bash
pip install -r requirements.txt 
```
*Note: You may need to create a `requirements.txt` file from `pyproject.toml` or install the dependencies manually.*

## 4. ML Pipeline Workflow

The machine learning pipeline is orchestrated in `main.py` and consists of several modular components.

### 4.1. Data Ingestion

- **Component**: `src/components/data_ingestion/data_ingestion.py`
- **Class**: `DataIngestion`
- **Functionality**: Reads the initial dataset from a Parquet file (`temp_data/final_dataset.parquet`) and saves it to the `data/raw/` directory.

### 4.2. Data Preprocessing

- **Component**: `src/components/data_preprocessing/processor.py`
- **Class**: `DataPreprocessor`
- **Functionality**:
    - Loads the raw data.
    - Removes duplicate records.
    - Handles missing values by imputing numerical columns with the mean and categorical columns with the mode.
    - Saves the cleaned data to `data/processed/processed_data.parquet`.
    - Generates a JSON report with preprocessing statistics.

### 4.3. Feature Engineering

- **Component**: `src/components/feature_engineering/feature_engineering.py`
- **Class**: `FeatureEngineering`
- **Functionality**:
    - Applies Variance Inflation Factor (VIF) to reduce multicollinearity among numerical features.
    - Iteratively removes features with a VIF score above a specified threshold (default is 5.0).
    - Saves the final dataset with selected features to `data/processed/final_data.parquet`.
    - Generates a CSV report with the VIF scores of the final features.

### 4.4. Data Transformation

- **Component**: `src/components/data_transformation/data_transformation.py`
- **Class**: `DataTransformation`
- **Functionality**:
    - Splits the data into training and testing sets (80/20 split) with stratification on the target variable.
    - Encodes the categorical target variable (`AttackCategory`) into numerical labels.
    - Defines `sklearn.pipeline.Pipeline` for numerical, binary, and categorical features:
        - **Numerical Pipeline**: Imputes missing values with the median and applies `StandardScaler`.
        - **Binary/Categorical Pipeline**: Imputes missing values with the most frequent value and applies `OneHotEncoder`.
    - Combines these pipelines into a single `ColumnTransformer` for consistent preprocessing.

### 4.5. Model Training

- **Component**: `src/components/model_training/model_training.py`
- **Class**: `ModelTrainer`
- **Functionality**:
    - Inherits from `DataTransformation` to access the preprocessing pipelines.
    - Creates a full `sklearn.pipeline.Pipeline` that combines the preprocessor with a classifier (default is `RandomForestClassifier`).
    - Trains the model on the training data.
    - Evaluates the model on the test set and saves a performance report (accuracy, precision, recall, F1-score) and plots (confusion matrix, ROC curve).
    - Performs k-fold cross-validation (default k=5) to assess model stability and saves detailed reports.
    - Saves the trained pipeline (preprocessor + model) as a single `.pkl` file (`models/ddos_model.pkl`).

### 4.6. Model Evaluation

- **Component**: `src/components/model_evulation/model_evaluation.py`
- **Class**: `ModelEvaluation`
- **Functionality**:
    - Loads the saved model pipeline and the test dataset.
    - Evaluates the model's performance on the unseen test data.
    - Generates and saves final evaluation metrics and visualizations (confusion matrix, ROC curve) in the `reports/` directory.

## 5. API Usage

The project includes a FastAPI application for serving the trained model.

- **API script**: `src/api/main.py`

### Endpoints

- **`GET /`**: Serves an HTML page with a simple UI for uploading a file for prediction.
- **`POST /predict`**: The main prediction endpoint.
    - **Request**:
        - `file`: An uploaded Parquet file containing network traffic data. **The input data should contain the original 78 network flow features as described in Section 7.1. Data Schema. The model's internal pipeline will handle feature selection and preprocessing.**
        - `model_type`: A form field to specify the model type (currently not used to switch models but is part of the request).
    - **Response**: A JSON object containing:
        - `encoded_predictions`: A list of numerical predictions (0 for "Benign", 1 for "DDoS").
        - `decoded_predictions`: A list of human-readable predictions ("Benign" or "DDoS").
        - `prediction_time`: The time taken for the prediction.
        - `distribution_encoded`: A dictionary with the count of each encoded prediction.
        - `distribution_decoded`: A dictionary with the count of each decoded prediction.

### Running the API

The API can be run directly from `src/api/main.py`.

```bash
python src/api/main.py
```

## 6. Utilities

- **Script**: `src/utils/utils.py`
- **Function**: `save_report`
- **Description**: A helper function to save reports in either JSON or CSV format. It ensures the output directory exists before saving.

## 7. Notebooks

- **Notebook**: `notebooks/cyber_attack_detection.ipynb`
- **Purpose**: This Jupyter notebook is used for initial exploratory data analysis (EDA).
- **Key Steps**:
    - Loads multiple Parquet files, each corresponding to a different type of DDoS attack (Syn, MSSQL, Portmap, etc.).
    - Maps the fine-grained attack labels to a single "DDoS" category to create a binary classification problem (DDoS vs. Benign).
    - Merges the individual datasets into a single, final dataset for the ML pipeline.
    - The notebook provides valuable insights into the data distribution and the rationale for the data merging strategy.

### 7.1. Data Schema

The dataset includes the following columns.  
**Note:** Column names must match exactly as listed below.

| Column Name                | Data Type |
|-----------------------------|------------|
| Protocol                    | int        |
| Fwd_Packet_Length_Std       | float      |
| Bwd_Packet_Length_Min       | int        |
| Bwd_Packet_Length_Std       | float      |
| Flow_Bytes_s                | float      |
| Fwd_IAT_Min                 | int        |
| Bwd_IAT_Total               | int        |
| Bwd_IAT_Mean                | float      |
| Bwd_IAT_Min                 | int        |
| Bwd_PSH_Flags               | int        |
| Fwd_URG_Flags               | int        |
| Bwd_URG_Flags               | int        |
| Fwd_Header_Length           | int        |
| Bwd_Header_Length           | int        |
| Fwd_Packets_s               | float      |
| Bwd_Packets_s               | float      |
| FIN_Flag_Count              | int        |
| SYN_Flag_Count              | int        |
| RST_Flag_Count              | int        |
| PSH_Flag_Count              | int        |
| ACK_Flag_Count              | int        |
| CWE_Flag_Count              | int        |
| ECE_Flag_Count              | int        |
| Down_Up_Ratio               | int        |
| Fwd_Avg_Bytes_Bulk          | int        |
| Fwd_Avg_Packets_Bulk        | int        |
| Fwd_Avg_Bulk_Rate           | int        |
| Bwd_Avg_Bytes_Bulk          | int        |
| Bwd_Avg_Packets_Bulk        | int        |
| Bwd_Avg_Bulk_Rate           | int        |
| Subflow_Fwd_Packets         | int        |
| Subflow_Fwd_Bytes           | int        |
| Init_Fwd_Win_Bytes          | int        |
| Init_Bwd_Win_Bytes          | int        |
| Fwd_Act_Data_Packets        | int        |
| Active_Mean                 | float      |
| Active_Std                  | float      |
| Idle_Std                    | float      |
| Idle_Min                    | int        |
