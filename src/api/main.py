import pandas as pd
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse,JSONResponse
from fastapi.exceptions import HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from src.api.schema.schemas import AttackData
import joblib
import numpy as np
import io
import time
from collections import Counter
import uvicorn
from pyngrok import ngrok

# Create a FastAPI app
app = FastAPI()

# Define expected schema (columns)
EXPECTED_COLUMNS = [
    "Protocol", "Fwd Packet Length Std", "Bwd Packet Length Min", "Bwd Packet Length Std",
    "Flow Bytes/s", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Min",
    "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Init Fwd Win Bytes", "Init Bwd Win Bytes", "Fwd Act Data Packets", "Active Mean",
    "Active Std", "Idle Std", "Idle Min"
]


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files
app.mount("/static", StaticFiles(directory="src/api/templates/static"), name="static")

# templates
templates = Jinja2Templates(directory="src/api/templates")

# Load the trained model
model = joblib.load("models/ddos_model.pkl")

MODEL_PATHS = {
    "network": "models/ddos_model.pkl",
    "system": "models/ddos_model.pkl",
    "kernel": "models/ddos_model.pkl"
}


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Define the prediction endpoint
@app.post("/predict")
def predict(file: UploadFile = File(...), model_type: str = Form(...)):
    """
    Upload a dataset (CSV or Parquet) and make predictions using the selected model.
    """

    # Validate model type
    if model_type not in MODEL_PATHS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_type '{model_type}'. Must be one of: {list(MODEL_PATHS.keys())}"
        )

    # Load the corresponding model
    try:
        model = joblib.load(MODEL_PATHS[model_type])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

    # Read uploaded file (support CSV and Parquet)
    try:
        content = file.file.read()
        if file.content_type == "text/csv" or file.filename.endswith(".csv"):
            df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        elif file.content_type in ["application/parquet", "application/octet-stream"] or file.filename.endswith(".parquet"):
            df = pd.read_parquet(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Upload CSV or Parquet.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

    # Handle extra columns (only keep expected)
    df = df[[col for col in df.columns if col in EXPECTED_COLUMNS]]

    # Handle missing columns
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        return JSONResponse(status_code=400, content={
                "error": "Missing required columns",
                "missing_columns": missing_cols
        })

    # Run prediction
    try:
        start_time = time.time()
        predictions = model.predict(df)
        end_time = time.time()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

    prediction_time = round(end_time - start_time, 4)

    # Label decoding (customize per model if needed)
    label_mapping = {0: "Benign", 1: "DDoS"}
    decoded_predictions = [label_mapping.get(int(p), "Unknown") for p in predictions]

    # Compute prediction distributions
    encoded_distribution = {str(k): int(v) for k, v in Counter(predictions).items()}
    decoded_distribution = {label_mapping.get(int(k), "Unknown"): int(v) for k, v in Counter(predictions).items()}

    print(f"Encoded distribution: {encoded_distribution}")
    print(f"Decoded distribution: {decoded_distribution}")

    # Return structured JSON response
    return JSONResponse(content={
        "encoded_predictions": predictions.tolist(),
        "decoded_predictions": decoded_predictions,
        "prediction_time": prediction_time,
        "distribution_encoded": encoded_distribution,
        "distribution_decoded": decoded_distribution
    })

if __name__=="__main__":
    # Step 1: Create an ngrok tunnel for port 8000
    # public_url = ngrok.connect(8000)
    # print(f"Public URL: {public_url.public_url}")

    # Step 2: Run the FastAPI app
    uvicorn.run(app, host="127.0.0.1", port=8000)