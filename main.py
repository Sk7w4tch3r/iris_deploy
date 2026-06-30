import joblib
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, ConfigDict
from fastapi import FastAPI


MODEL_PATH = "knn.joblib"

ml_artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ml_artifacts["bundle"] = joblib.load(MODEL_PATH)
        print("model loaded successfully")
    except Exception as e:
        print(e)
    yield


app = FastAPI(title="IRIS flower classifer", lifespan=lifespan)


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0, description="Sepal length in cm")
    sepal_width: float = Field(..., ge=0, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0, description="Petal length in cm")
    petal_width: float = Field(..., ge=0, description="Petal width in cm")


class PredictionResponses(BaseModel):
    predicted_class : str



@app.post("/predict", response_model=PredictionResponses)
def predict(features: IrisFeatures):
    model = ml_artifacts["bundle"]["model"]
    target_names = ml_artifacts["bundle"]["target_names"]

    sepal_length = features.sepal_length
    sepal_width = features.sepal_width
    petal_length = features.petal_length
    petal_width = features.petal_width

    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    predicted_class = target_names[prediction[0]]

    return PredictionResponses(
        predicted_class=predicted_class
    )