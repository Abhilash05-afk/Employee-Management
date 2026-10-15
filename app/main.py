from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import EmployeeData, PredictionResponse
from app.database import get_connection


# ============================================================
# APPLICATION SETUP
# ============================================================

app = FastAPI(
    title="Employee Attrition Prediction API",
    description="REST API for predicting employee attrition using a machine learning model.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ============================================================
# PREDICTION SERVICE
# ============================================================

class Predictions:
    """Encapsulates model loading, attrition predictions, and stored history."""

    def __init__(self, model=None):
        self.model = model

    def load_model(self, model_path: Path):
        try:
            self.model = joblib.load(model_path)
            return self.model is not None
        except Exception as error:
            self.model = None
            print(f"Error loading model: {error}")
            return False

    def health_check(self):
        return {
            "status": "healthy" if self.model is not None else "unhealthy",
            "model_loaded": self.model is not None,
        }

    def predict_attrition(self, employee: EmployeeData):
        if self.model is None:
            raise ValueError("Machine learning model could not be loaded.")

        employee_data = employee.model_dump()
        input_data = pd.DataFrame([employee_data])

        prediction = self.model.predict(input_data)[0]
        probabilities = self.model.predict_proba(input_data)[0]

        classes = self.model.named_steps["model"].classes_
        yes_index = list(classes).index("Yes")

        attrition_probability = round(float(probabilities[yes_index]), 4)

        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO prediction_history
            (age, department, job_role, overtime, prediction, attrition_probability)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                employee.Age,
                employee.Department,
                employee.JobRole,
                employee.OverTime,
                prediction,
                attrition_probability,
            ),
        )
        connection.commit()
        cursor.close()
        connection.close()

        return {
            "prediction": prediction,
            "attrition_probability": attrition_probability,
        }

    def get_prediction_history(self):
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT id, age, department, job_role, overtime,
                   prediction, attrition_probability, created_at
            FROM prediction_history
            ORDER BY created_at DESC
            """
        )

        rows = cursor.fetchall()
        cursor.close()
        connection.close()

        predictions = []
        for row in rows:
            predictions.append({
                "id": row[0],
                "age": row[1],
                "department": row[2],
                "job_role": row[3],
                "overtime": row[4],
                "prediction": row[5],
                "attrition_probability": float(row[6]),
                "created_at": row[7],
            })

        return predictions


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "attrition_model.pkl"

predictions_service = Predictions()
predictions_service.load_model(MODEL_PATH)
model = predictions_service.model


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Employee Attrition Prediction API",
        "status": "running",
        "docs": "/docs"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    return predictions_service.health_check()


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict", response_model=PredictionResponse)
def predict_attrition(employee: EmployeeData):
    if predictions_service.model is None:
        raise HTTPException(
            status_code=500,
            detail="Machine learning model could not be loaded."
        )

    try:
        return predictions_service.predict_attrition(employee)
    except ValueError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}",
        ) from error


@app.get("/predictions")
def get_predictions():
    return predictions_service.get_prediction_history()