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

# FastAPI — фреймворк для создания REST API
from fastapi import FastAPI, HTTPException

# Pydantic используется FastAPI для валидации входных и выходных данных
from pydantic import BaseModel

import pandas as pd

# Loguru — удобный логгер для логирования событий приложения
from loguru import logger

# Импорт функций инференса:
# load_model — загрузка сохранённой модели
# predict — получение предсказаний
from src.inference import load_model, predict


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
    """
    Схема ответа API.

    """
    predicted_time_minutes: float


# ====== Загрузка модели ======

# Логируем старт загрузки модели
logger.info("Loading model")

# Путь до сохранённой модели
MODEL_PATH = os.path.join("models", "model.joblib")

# Загружаем модель один раз при старте приложения,
# а не при каждом запросе (важно для производительности)
MODEL = load_model(MODEL_PATH)

logger.info("Model loaded successfully")


# ====== Endpoints ======

# Декоратор @app.get("/") регистрирует функцию как обработчик GET-запросов по корневому пути "/"
# GET-запросы обычно используются для получения данных (без изменения состояния сервера)
@app.get("/")
def health_check():
    """
    Health-check endpoint.

    Используется для проверки, что сервис жив
    (например, в Docker, Kubernetes, monitoring).
    """
    return {"status": "ok"}


# @app.post указывает, что это endpoint для обработки POST-запросов
# POST обычно используется для отправки данных на сервер (как в нашем случае - признаков для предсказания)
# response_model=DeliveryPrediction - указывает FastAPI на формат выходных данных
# Это обеспечивает автоматическую валидацию и документацию в Swagger
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
    """
    try:
        # Преобразуем входные данные в DataFrame,
        # т.к. большинство sklearn-моделей ожидают именно такой формат
        data = pd.DataFrame([features.model_dump()])

        # Получаем предсказание модели
        prediction = predict(MODEL, data)

        logger.info(f"Raw model prediction: {prediction[0]}")

        estimated_time = float(prediction[0])

        logger.info(f"Predicted delivery time: {estimated_time:.2f} min")
        return DeliveryPrediction(predicted_time_minutes=estimated_time)

    except Exception as e:
        # Логируем ошибку и возвращаем HTTP 500
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )

    # Возвращаем результат в формате Pydantic-модели
    return DeliveryPrediction(predicted_time_minutes=predicted_time_minutes)
