import tensorflow as tf
import traceback
import numpy as np
from datetime import datetime, timedelta
import os
import json
import requests
import sys
from model import TechnicalIndicators, custom_loss
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


class TinkoffClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def get_figi(self, ticker: str) -> str:
        response = requests.get(f"{self.base_url}/figi/{ticker}")
        if response.status_code != 200:
            raise Exception(f"Error getting FIGI: {response.text}")
        return response.json()["figi"]
    
    def get_candles(self, ticker: str, interval: str, days: int = 3) -> list:
        # Adjust days based on interval
        if interval in ["1min", "5min", "15min","30min"]:
            days = 1
        elif interval == "1h":
            days = min(days, 7)
        
        response = requests.post(
            f"{self.base_url}/candles",
            json={"ticker": ticker, "interval": interval, "days": days}
        )
        if response.status_code != 200:
            raise Exception(f"Error getting candles: {response.text}")
        return response.json()
    
def load_model(interval):
    """Загрузка модели с новой структурой сохранения"""
    try:
        model_dir = os.path.join(MODELS_PATH, f"model_{interval}")
        
        # Загружаем архитектуру модели из JSON
        architecture_path = os.path.join(model_dir, "architecture.json")
        with open(architecture_path, 'r') as f:
            model_json = f.read()
            
        # Создаем модель из JSON с пользовательскими объектами
        custom_objects = {
            'TechnicalIndicators': TechnicalIndicators,
            'custom_loss': custom_loss
        }
        model = tf.keras.models.model_from_json(model_json, custom_objects=custom_objects)
        
        # Загружаем веса с подавлением предупреждений
        weights_path = os.path.join(model_dir, "final_weights")
        model.load_weights(weights_path).expect_partial()  # Added expect_partial()
        
        # Компилируем модель с базовыми настройками для предсказаний
        model.compile(
            optimizer='adam',  # Используем базовый оптимизатор
            loss=custom_loss,
            metrics=[
                tf.keras.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanSquaredError()
            ]
        )
        
        print(f"Модель успешно загружена из {model_dir}")
        return model
        
    except Exception as e:
        print(f"Ошибка при загрузке модели: {str(e)}")
        print(traceback.format_exc())
        return None


def prepare_data(candles, window_size=30):
    """Подготовка данных для предсказания"""
    try:
        # Преобразуем свечи в numpy массив
        data = []
        for candle in candles:
            row = [
                candle["open"],
                candle["high"],
                candle["low"],
                candle["close"],
                candle["volume"],
                0,  # Placeholder для volume_buy
                0,  # Placeholder для volume_sell
            ]
            data.append(row)
            
        data = np.array(data)
        
        # Нормализация данных
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized_data = (data - mean) / (std + 1e-8)
        
        # Создаем окна данных
        X = []
        for i in range(len(normalized_data) - window_size + 1):
            window = normalized_data[i:i + window_size]
            X.append(window)
            
        X = np.array(X)
        
        # Добавляем технические индикаторы через слой
        technical_layer = TechnicalIndicators()
        X = technical_layer(X).numpy()
        
        # Обрезаем до нужной размерности (22 признака)
        if X.shape[-1] > 22:
            print("Обрезаем данные до нужной размерности (22 признака)")
            X = X[:, :, :22]
            
        return X, mean, std
        
    except Exception as e:
        print(f"Ошибка при подготовке данных: {e}")
        print(traceback.format_exc())
        return None, None, None

def get_prediction(model, data, mean, std):
    """Получение предсказания и его денормализация"""
    try:
        # Приводим данные к float16
        data = tf.cast(data, tf.float16)
        
        # Получаем предсказание
        predictions = model.predict(data, verbose=0)
        
        # Берем последнее предсказание и первый элемент (цена)
        price_pred = predictions[-1][0]
        
        # Денормализация
        price_pred = (price_pred * std[3]) + mean[3]
        
        # Получаем границы доверительного интервала
        lower_bound = predictions[-1][1]
        upper_bound = predictions[-1][2]
        lower_bound = (lower_bound * std[3]) + mean[3]
        upper_bound = (upper_bound * std[3]) + mean[3]
        
        return price_pred, lower_bound, upper_bound
        
    except Exception as e:
        print(f"Ошибка при получении предсказания: {e}")
        return None, None, None

def main():
    """Основная функция приложения"""
    try:
        print("\nПрогнозирование цен акций")
        print("=" * 50)
        
        # Инициализация клиента
        client = TinkoffClient(base_url="http://localhost:8000")
            
        # Выбор интервала
        print("\nДоступные интервалы:")
        for i, interval in enumerate(TIME_INTERVALS, 1):
            print(f"{i}. {interval}")
            
        interval_idx = int(input("\nВыберите интервал (номер): ")) - 1
        if interval_idx < 0 or interval_idx >= len(TIME_INTERVALS):
            print("Неверный номер интервала")
            return
            
        interval = TIME_INTERVALS[interval_idx]
        
        # Ввод тикера
        ticker = input("\nВведите тикер акции: ").upper()
        
        # Загружаем модель
        print("\nЗагрузка модели...")
        model = load_model(interval)
        if model is None:
            return
            
        # Получаем данные
        print("\nПолучение исторических данных...")
        try:
            candles = client.get_candles(
                ticker=ticker,
                interval=interval,
                days=3  # За последние 3 дня
            )
        except Exception as e:
            print(f"Ошибка при получении данных: {e}")
            return
            
        if not candles:
            print("Не удалось получить данные")
            return
            
        # Подготовка данных
        print("\nПодготовка данных...")
        X, mean, std = prepare_data(candles)
        if X is None:
            return
            
        # Получение предсказания
        print("\nРасчет прогноза...")
        price_pred, lower_bound, upper_bound = get_prediction(model, X, mean, std)
        if price_pred is None:
            return
            
        # Вывод результатов
        print("\nРезультаты прогноза:")
        print("=" * 50)
        print(f"Тикер: {ticker}")
        print(f"Интервал: {interval}")
        print(f"Предсказанная цена: {price_pred:.2f}")
        print(f"Доверительный интервал: [{lower_bound:.2f} - {upper_bound:.2f}]")
            
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")

if __name__ == "__main__":
    main()