"""
Module: app.py

Description:
    Точка входа FastAPI-приложения.
    Отвечает за:
    - инициализацию API
    - загрузку обученной ML-модели
    - приём HTTP-запросов
    - возврат предсказаний модели
"""

import os

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, ValidationError
from sklearn.exceptions import NotFittedError 

from src.inference import load_model, predict

# Константы
MAX_PREDICTION_MINUTES = 300  # 5 часов
VALID_TRAFFIC_LEVELS = {1, 2, 3, 4, 5}
BINARY_VALUES = {0, 1}


# Создаём экземпляр FastAPI-приложения
app = FastAPI()


# ====== Pydantic-модели ======

class DeliveryFeatures(BaseModel):
    """
    Схема входных данных для модели.

    Используется FastAPI для:
    - валидации входного JSON
    - автогенерации Swagger-документации
    """
    distance_km: float
    prep_time_avg: int
    precip_mm: float
    traffic_level: int
    hour_sin: float
    is_fast_food: int
    is_express_delivery: int
    base_speed_kmh: float
    items_count: int
    hour_cos: float


class DeliveryPrediction(BaseModel):
    """Схема ответа API."""
    predicted_time_minutes: float


# ====== Загрузка модели ======

logger.info("Loading model")
MODEL_PATH = os.path.join("models", "model.joblib")
MODEL = load_model(MODEL_PATH)
logger.info("Model loaded successfully")


# ====== Endpoints ======

@app.get("/")
def health_check():
    """
    Health-check endpoint.

    Используется для проверки, что сервис жив
    (например, в Docker, Kubernetes, monitoring).
    """
    return {"status": "ok"}


@app.post("/predict", response_model=DeliveryPrediction)
def get_prediction(features: DeliveryFeatures):
    """
    Endpoint для получения предсказания ML-модели.

    Parameters
    ----------
    features : DeliveryFeatures
        Признаки параметров доставки, переданные в JSON.

    Returns
    -------
    DeliveryPrediction
        Предсказанное время доставки в минутах

    Raises
    ------
    HTTPException
        - 400: Неверные входные данные
        - 422: Ошибка валидации данных
        - 500: Внутренняя ошибка сервера
    """
    try:
        # 1. Проверка, что модель загружена
        if MODEL is None:
            logger.error("Model is not loaded")
            raise HTTPException(
                status_code=503,
                detail="Model is not available. Please try again later."
            )

        # 2. Преобразование входных данных в DataFrame
        try:
            input_data = features.model_dump()
            logger.debug(f"Input data: {input_data}")

            # Проверка на наличие None значений
            if any(v is None for v in input_data.values()):
                raise ValueError("Input data contains None values")

            data = pd.DataFrame([input_data])

        except Exception as e:
            logger.error(f"Error converting input data to DataFrame: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid data format: {str(e)}"
            )

        # 3. Проверка на пустой DataFrame
        if data.empty:
            logger.error("Empty DataFrame after conversion")
            raise HTTPException(
                status_code=400,
                detail="No data provided for prediction"
            )

        # 4. Проверка наличия всех необходимых признаков
        expected_features = [
            "distance_km", "prep_time_avg", "precip_mm", "traffic_level",
            "hour_sin", "is_fast_food", "is_express_delivery",
            "base_speed_kmh", "items_count", "hour_cos"
        ]

        missing_features = [
            f for f in expected_features if f not in data.columns
        ]
        if missing_features:
            logger.error(f"Missing features: {missing_features}")
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {missing_features}"
            )

        # 5. Проверка типов данных и допустимых значений
        # Проверка на отрицательные значения в distance_km
        if data["distance_km"].iloc[0] < 0:
            raise HTTPException(
                status_code=400,
                detail="Distance cannot be negative"
            )

        # Проверка на допустимые значения traffic_level
        traffic_level = data["traffic_level"].iloc[0]
        if traffic_level not in VALID_TRAFFIC_LEVELS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid traffic_level: {traffic_level}. "
                       f"Must be {sorted(VALID_TRAFFIC_LEVELS)}"
            )

        # Проверка на допустимые значения is_fast_food (0 или 1)
        if data["is_fast_food"].iloc[0] not in BINARY_VALUES:
            raise HTTPException(
                status_code=400,
                detail="is_fast_food must be 0 or 1"
            )

        # Проверка на допустимые значения is_express_delivery (0 или 1)
        if data["is_express_delivery"].iloc[0] not in BINARY_VALUES:
            raise HTTPException(
                status_code=400,
                detail="is_express_delivery must be 0 or 1"
            )

        # Проверка на неотрицательное количество items_count
        if data["items_count"].iloc[0] <= 0:
            raise HTTPException(
                status_code=400,
                detail="items_count must be positive"
            )

        # Проверка на NaN значения
        if data.isnull().any().any():
            nan_columns = data.columns[data.isnull().any()].tolist()
            logger.error(f"NaN values found in columns: {nan_columns}")
            raise HTTPException(
                status_code=400,
                detail=f"NaN values found in features: {nan_columns}"
            )

        # 6. Проверка на бесконечные значения
        if data.isin([np.inf, -np.inf]).any().any():
            logger.error("Infinite values found in input data")
            raise HTTPException(
                status_code=400,
                detail="Input data contains infinite values"
            )

        # 7. Получение предсказания с проверкой
        try:
            prediction = predict(MODEL, data)
        except NotFittedError as e:
            logger.error(f"Model not fitted: {e}")
            raise HTTPException(
                status_code=500,
                detail="Model is not properly trained"
            )
        except ValueError as e:
            logger.error(f"Value error during prediction: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid input for model prediction: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error during prediction: {e}")
            raise HTTPException(
                status_code=500,
                detail="Error during model prediction"
            )

        # 8. Проверка предсказания
        if prediction is None or len(prediction) == 0:
            logger.error("Empty prediction received from model")
            raise HTTPException(
                status_code=500,
                detail="Model returned empty prediction"
            )

        # 9. Извлечение и проверка результата
        try:
            estimated_time = float(prediction[0])
        except (TypeError, ValueError, IndexError) as e:
            logger.error(f"Error converting prediction to float: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Invalid prediction format: {str(e)}"
            )

        # 10. Проверка на реалистичность предсказания
        if estimated_time < 0:
            logger.warning(f"Negative prediction received: {estimated_time}")
            estimated_time = max(estimated_time, 0)

        if estimated_time > MAX_PREDICTION_MINUTES:
            logger.warning(
                f"Unusually high prediction: {estimated_time} minutes"
            )

        # Логирование успешного предсказания
        logger.info(
            f"Successful prediction: {estimated_time:.2f} min "
            f"for input: {input_data}"
        )

        return DeliveryPrediction(predicted_time_minutes=estimated_time)

    except HTTPException:
        raise

    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=422,
            detail=f"Validation error: {str(e)}"
        )

    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty data error: {e}")
        raise HTTPException(
            status_code=400,
            detail="Empty data provided"
        )

    except KeyError as e:
        logger.error(f"Key error in data processing: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Missing expected field: {str(e)}"
        )

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
