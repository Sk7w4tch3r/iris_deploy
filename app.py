"""
main.py
Serves the trained Iris classifier as a REST API using FastAPI.

Before running this, train the model once:
    python train_model.py

Then run the API with:
    uvicorn main:app --reload --port 8000

Visit http://127.0.0.1:8000/docs for the interactive API docs.
"""

from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = "iris_model.joblib"

# Loaded once at startup via the lifespan handler below, not per-request -
# loading from disk is fast for a model this small, but the same pattern
# matters even more for larger models where loading is slow
ml_artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model...")
    try:
        ml_artifacts["bundle"] = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        raise RuntimeError(
            f"Could not find {MODEL_PATH}. Run 'python train_model.py' first."
        )
    print("Model loaded. API ready.")
    yield
    ml_artifacts.clear()


app = FastAPI(title="Iris Classifier API", version="1.0.0", lifespan=lifespan)


class IrisFeatures(BaseModel):
    """The four measurements (in cm) the model needs to make a prediction."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            }
        }
    )

    sepal_length: float = Field(..., gt=0, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, description="Petal width in cm")


class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_class_index: int
    probabilities: dict[str, float]


@app.get("/health")
def health_check():
    """Confirms the API is running AND the model is loaded."""
    return {"status": "ok", "model_loaded": "bundle" in ml_artifacts}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    bundle = ml_artifacts.get("bundle")
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    model = bundle["model"]
    target_names = bundle["target_names"]

    # scikit-learn expects a 2D array: one row per sample, in the same
    # feature order the model was trained on
    input_row = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]]

    predicted_index = int(model.predict(input_row)[0])
    predicted_label = target_names[predicted_index]

    probabilities = model.predict_proba(input_row)[0]
    probability_map = {
        target_names[i]: round(float(p), 4) for i, p in enumerate(probabilities)
    }

    return PredictionResponse(
        predicted_class=predicted_label,
        predicted_class_index=predicted_index,
        probabilities=probability_map,
    )
