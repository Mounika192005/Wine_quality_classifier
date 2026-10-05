from pathlib import Path

import joblib
import numpy as np

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ---------------------------------------------------------
# PROJECT STRUCTURE
#
# My_practice(ML)/
# ├── app.py                  <- this file
# ├── Model/wine_quality_model.joblib
# └── UI/
#     ├── templates/index.html
#     └── static/css/style.css, static/js/scripts.js
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "Model" / "wine_quality_model.joblib"
UI_DIR = BASE_DIR / "UI"
HTML_PATH = UI_DIR / "templates" / "index.html"
STATIC_DIR = UI_DIR / "static"


# ---------------------------------------------------------
# APP
# ---------------------------------------------------------

app = FastAPI(
    title="Wine Quality Classifier",
    description="FastAPI application for wine quality prediction",
    version="1.0.0"
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# STATIC FILES (CSS / JS)  ->  /static/css/style.css
#                              /static/js/scripts.js
# ---------------------------------------------------------

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------
# FEATURES (same order as the training data)
# ---------------------------------------------------------

FEATURES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

try:
    loaded_model = joblib.load(MODEL_PATH)

    # Case 1: the joblib file contains the model directly
    if hasattr(loaded_model, "predict"):
        model = loaded_model

    # Case 2: the joblib file contains a dictionary
    elif isinstance(loaded_model, dict):
        if "model" not in loaded_model:
            raise ValueError(
                "Model file is a dictionary but does not contain a 'model' key."
            )

        model = loaded_model["model"]

        if "columns" in loaded_model:
            FEATURES = list(loaded_model["columns"])

    else:
        raise ValueError(
            "Unsupported model file format. "
            "It should contain a trained model or a dictionary containing 'model'."
        )

    print("✅ Model loaded successfully from:", MODEL_PATH)
    print("✅ Features:", FEATURES)

except Exception as e:
    model = None
    print("❌ Model loading failed:", e)
    print("   Looked for the model at:", MODEL_PATH)


# ---------------------------------------------------------
# INPUT SCHEMA
# ---------------------------------------------------------

class WineInput(BaseModel):

    fixed_acidity: float = Field(..., description="Fixed acidity")
    volatile_acidity: float = Field(..., description="Volatile acidity")
    citric_acid: float = Field(..., description="Citric acid")
    residual_sugar: float = Field(..., description="Residual sugar")
    chlorides: float = Field(..., description="Chlorides")
    free_sulfur_dioxide: float = Field(..., description="Free sulfur dioxide")
    total_sulfur_dioxide: float = Field(..., description="Total sulfur dioxide")
    density: float = Field(..., description="Density")
    pH: float = Field(..., description="pH")
    sulphates: float = Field(..., description="Sulphates")
    alcohol: float = Field(..., description="Alcohol")


# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(HTML_PATH)


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "running",
        "model_loaded": model is not None,
        "features": FEATURES
    }


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

@app.post("/predict")
def predict(data: WineInput):

    if model is None:
        return {
            "success": False,
            "error": "Model could not be loaded."
        }

    try:

        # Values in the EXACT order used during training
        values = [
            data.fixed_acidity,
            data.volatile_acidity,
            data.citric_acid,
            data.residual_sugar,
            data.chlorides,
            data.free_sulfur_dioxide,
            data.total_sulfur_dioxide,
            data.density,
            data.pH,
            data.sulphates,
            data.alcohol
        ]

        # The model was trained on a DataFrame with column names,
        # so give it the same column names to avoid sklearn warnings.
        if hasattr(model, "feature_names_in_"):
            import pandas as pd

            input_data = pd.DataFrame(
                [values],
                columns=list(model.feature_names_in_)
            )
        else:
            input_data = np.array([values], dtype=float)

        prediction = model.predict(input_data)

        # Convert numpy output into a normal Python value
        result = prediction[0]

        if hasattr(result, "item"):
            result = result.item()

        return {
            "success": True,
            "prediction": result
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }


# ---------------------------------------------------------
# RUN  (python app.py)
# ---------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
