# 🚚 Delivery Time Prediction Service

Сервис предсказания времени доставки на основе обученной ML-модели.

Проект реализует подход **Model as a Service (MaaS)**: модель обёрнута в REST API с использованием **FastAPI** и возвращает предсказанное время доставки в минутах.

---

## 📋 Описание задачи

Модель предсказывает **время доставки в минутах** на основе следующих признаков заказа:

| Признак | Тип | Описание |
|---------|-----|----------|
| `distance_km` | float | Расстояние доставки в километрах |
| `prep_time_avg` | int | Среднее время приготовления заказа, минут |
| `precip_mm` | float | Количество осадков в мм |
| `traffic_level` | int | Уровень трафика: 0 — низкий, 5 — высокий |
| `hour_sin` | float | Синус часа заказа |
| `hour_cos` | float | Косинус часа заказа |
| `is_fast_food` | int | Флаг быстрого питания, 0/1 |
| `is_express_delivery` | int | Флаг экспресс-доставки, 0/1 |
| `base_speed_kmh` | float | Базовая скорость курьера, км/ч |
| `items_count` | int | Количество позиций в заказе |

### Выходное значение

| Поле | Тип | Описание |
|------|-----|----------|
| `predicted_time_minutes` | float | Предсказанное время доставки |

---

## 📁 Структура проекта

```text
delivery-time-prediction/
├── models/
│   └── model.joblib
├── src/
│   ├── __init__.py
│   ├── app.py
│   └── inference.py
├── tests/
├── requirements.txt
└── README.md
```

---

## 🧠 Архитектура решения

1. Модель обучена на подготовленных данных.
2. Обученная модель сохранена в `models/model.joblib`.
3. При запуске сервиса модель загружается в память.
4. Входные данные валидируются через Pydantic.
5. На выходе API возвращает предсказанное время доставки.

---

## 🛠️ Установка и запуск

### 1. Клонирование репозитория

```bash
git clone https://github.com/evmor-ml/delivery-time-prediction
cd delivery-time-prediction
```

### 2. Создание виртуального окружения

#### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate
```

```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Запуск сервиса

```bash
uvicorn src.app:app --reload --port 8000
```

После запуска сервис будет доступен по адресу:

```text
http://127.0.0.1:8000
```

---

## 📚 Документация API

FastAPI автоматически создаёт интерактивную документацию:

- Swagger UI: `http://127.0.0.1:8000/docs`

---

## 🔮 Пример запроса

### POST `/predict`

**Request body**:

```json
{
  "distance_km": 3.8,
  "prep_time_avg": 20,
  "precip_mm": 8.34,
  "traffic_level": 2,
  "hour_sin": 0.97,
  "hour_cos": -0.26,
  "is_fast_food": 0,
  "is_express_delivery": 0,
  "base_speed_kmh": 55.0,
  "items_count": 2
}
```

**Response**:

```json
{
  "predicted_time_minutes": 74.749701035996
}
```

### cURL

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "distance_km": 3.8,
    "prep_time_avg": 20,
    "precip_mm": 8.34,
    "traffic_level": 2,
    "hour_sin": 0.97,
    "hour_cos": -0.26,
    "is_fast_food": 0,
    "is_express_delivery": 0,
    "base_speed_kmh": 55.0,
    "items_count": 2
  }'
```

---

## 🔧 Health Check

Проверка работоспособности сервиса:

```bash
curl http://127.0.0.1:8000/health
```

**Response**:

```json
{
  "status": "ok"
}
```

---

## 📊 Метрики модели

- MAE: 5.6 минуты
- R² Score: 0.95

---

## 🧩 Используемые технологии

| Технология | Назначение |
|------------|-------------|
| Python 3.10+ | Язык программирования |
| FastAPI | Веб-фреймворк для API |
| Pydantic | Валидация данных |
| Scikit-learn | ML-модели и препроцессинг |
| Pandas | Обработка данных |
| Joblib | Сохранение и загрузка модели |
| Uvicorn | ASGI-сервер |
| Loguru | Логирование |

---

## 🎯 Цель проекта

Проект демонстрирует:

- деплой ML-модели регрессии как сервиса;
- работу с FastAPI и автоматическую генерацию Swagger-документации;
- предсказание бизнес-метрики в реальном времени.

---

## 📝 Примечания

- Модель уже обучена и хранится в папке `models/`.
- Все входные признаки должны быть переданы в корректном формате.
- Циклические признаки `hour_sin` и `hour_cos` вычисляются на этапе препроцессинга.

---

## 📄 Лицензия

MIT
