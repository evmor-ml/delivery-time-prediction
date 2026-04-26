import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

# Инициализация приложения
app = FastAPI(title="Delivery Time Prediction Service")

# 1. Схема входных данных (то, что присылает клиент)
class DeliveryRequest(BaseModel):
    distance_km: float = Field(..., description="Расстояние в километрах", gt=0)
    prep_time_avg: int = Field(..., description="Среднее время на приготовление заказа", ge=1, le=5)
    precip_mm: int = Field(..., description="Наличие осадков в миллиметрах")
    traffic_level: int = Field(..., description="Уровень трафика (от 1 до 5)")
    hour: int = Field(..., description="Время заказа (полных часов, от 0 до 23)")
    is_fast_food: int = Field(..., description="Фастфуд (0 - нет, 1 - да)")
    is_express_delivery: int = Field(..., description="Срочная доставка (0 - нет, 1 - да)")
    items_count: int = Field(..., description="Количество наименований в заказе")
    base_speed_kmh: float = Field(..., description="Скорость транспорта курьера (от 25 до 50)")

# 2. Схема ответа
class DeliveryResponse(BaseModel):
    predicted_time_min: float
    status: str = "success"

# 3. Загрузка модели при старте
MODEL_PATH = "models/model.joblib"

try:
    # Загружаем модель (joblib)
    model = joblib.load(MODEL_PATH)
    logger.info("Модель успешно загружена")
except Exception as e:
    logger.error(f"Ошибка при загрузке модели: {e}")
    model = None

# 4. Функция предобработки
def preprocess_data(data: DeliveryRequest) -> pd.DataFrame:
    """Трансформирует JSON в DataFrame для модели"""
    df = pd.DataFrame([data.model_dump()])
    
    # Формирование необходимых признаков
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/ 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/ 24)
    
    # Удаляем уже лишнее поле
    df = df.drop(columns=['hour'])
    features_order = ['distance_km', 'prep_time_avg', 'precip_mm', 'traffic_level', 'hour_sin', 'is_fast_food', 'is_express_delivery', 'base_speed_kmh',
    'items_count', 'hour_cos']

    df = df[features_order] 
    return df

# 5. Endpoint предсказания
@app.post("/predict", response_model=DeliveryResponse)
async def predict_delivery_time(request: DeliveryRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не инициализирована")
    
    try:
        # Предобработка
        features = preprocess_data(request)
        
        # Получение предсказания
        prediction = model.predict(features)
        result = float(prediction[0])
        
        logger.info(f"Запрос: {request.model_dump()} | Предсказание: {result:.2f}")
        
        return DeliveryResponse(predicted_time_min=round(result, 2))
        
    except Exception as e:
        logger.error(f"Ошибка инференса: {e}")
        raise HTTPException(status_code=500, detail="Ошибка при расчете предсказания")

# 6. Проверка работоспособности
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}
