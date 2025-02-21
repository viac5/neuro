import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import traceback
import sys
from model import TechnicalIndicators, custom_loss
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

def load_test_data(interval, symbol):
    """Загрузка тестовых данных для конкретного тикера"""
    try:
        symbol_path = os.path.join(PROCESSED_DATA_PATH, symbol)
        X_test = np.load(os.path.join(symbol_path, f"X_test_{interval}.npy"))
        y_test = np.load(os.path.join(symbol_path, f"y_test_{interval}.npy"))
        
        print(f"Исходная форма входных данных: {X_test.shape}")
        print(f"Загружены данные для {symbol}: {len(X_test)} записей")
        
        # Skip technical indicators preprocessing if shape is already correct
        if X_test.shape[-1] == 22:  # Expected input shape
            print("Данные уже в правильном формате")
            return X_test, y_test
            
        # Apply TechnicalIndicators layer to preprocess data
        technical_layer = TechnicalIndicators()
        X_test_processed = technical_layer(X_test).numpy()
        print(f"Форма данных после обработки: {X_test_processed.shape}")
        
        # Select only first 22 features if we have more
        if X_test_processed.shape[-1] > 22:
            print(f"Обрезаем данные до нужной размерности (22 признака)")
            X_test_processed = X_test_processed[:, :, :22]
            print(f"Итоговая форма данных: {X_test_processed.shape}")
            
        return X_test_processed, y_test
    except FileNotFoundError:
        print(f"Данные не найдены для {symbol} - {interval}")
        return None, None
    except Exception as e:
        print(f"Ошибка при обработке данных: {str(e)}")
        return None, None
    
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

def plot_predictions(interval, symbol):
    """Визуализация предсказаний модели для конкретного тикера"""
    try:
        # Загрузка тестовых данных для тикера
        X_test, y_test = load_test_data(interval, symbol)
        if X_test is None:
            return
            
        print(f"Тестовых записей для {symbol}: {len(X_test)}")
        print(f"Форма входных данных: {X_test.shape}")
        
        # Загружаем модель с новой структурой
        model = load_model(interval)
        if model is None:
            return
            
        # Приводим данные к float16 для соответствия модели
        X_test = tf.cast(X_test, tf.float16)
        
        # Получение предсказаний
        print(f"Получение предсказаний для {symbol}...")
        predictions = model.predict(X_test, batch_size=32, verbose=0)
        print(f"Форма предсказаний до обработки: {predictions.shape}")
        
        # Извлекаем только предсказания цены (первый элемент из трех)
        predictions = predictions[:, 0]  # Берем только предсказания цены
        y_test = y_test[:, 0] if len(y_test.shape) > 1 else y_test
        
        print(f"Форма предсказаний после обработки: {predictions.shape}")
        print(f"Форма целевых значений: {y_test.shape}")
        
        # Создание директорий для графиков
        symbol_plots_dir = os.path.join(MODELS_PATH, 'plots', symbol)
        if not os.path.exists(symbol_plots_dir):
            os.makedirs(symbol_plots_dir)
        
        # Построение графика
        plt.figure(figsize=(15, 8))
        plt.plot(y_test, label='Реальные значения', color='blue', alpha=0.7)
        plt.plot(predictions, label='Предсказания', color='red', alpha=0.7)
        plt.title(f'Предсказания для {symbol} - интервал {interval}')
        plt.xlabel('Время')
        plt.ylabel('Нормализованная цена')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Добавляем границы доверительного интервала
        lower_bound = predictions * 0.98  # Нижняя граница
        upper_bound = predictions * 1.02  # Верхняя граница
        plt.fill_between(range(len(predictions)), lower_bound, upper_bound, 
                        color='red', alpha=0.2, label='Доверительный интервал')
        
        # Сохранение графика
        save_path = os.path.join(symbol_plots_dir, f"predictions_{interval}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Расчет метрик
        mse = np.mean(np.square(y_test - predictions))
        mae = np.mean(np.abs(y_test - predictions))
        print(f"\nМетрики для {symbol} - {interval}:")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"График сохранен в {save_path}")
        
    except Exception as e:
        print(f"Ошибка при обработке {symbol} - {interval}: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    print("Начало визуализации результатов...")
    
    for interval in TIME_INTERVALS:
        print(f"\n{'='*50}")
        print(f"Обработка интервала: {interval}")
        print(f"{'='*50}")
        
        for symbol in STOCK_SYMBOLS:
            print(f"\n{'-'*30}")
            print(f"Обработка {symbol}")
            print(f"{'-'*30}")
            
            try:
                plot_predictions(interval, symbol)
                print(f"График для {symbol} - {interval} успешно создан")
            except Exception as e:
                print(f"Ошибка при создании графика для {symbol} - {interval}: {e}")
