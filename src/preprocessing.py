import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

samples_per_day = {
            "1min": 520,  # (18:50 - 10:00) * 60 мин
            "5min": 104,  # 520 / 5
            "15min": 104,  # 520 / 15
            "30min": 104,  # 520 / 30
            "hour": 35,    # 520 / 60
        }


def add_technical_indicators(df):
    """Расширенный набор технических индикаторов"""
    # Проверка и преобразование столбца времени
    if 'datetime' not in df.columns:
        if 'date' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'])
        else:
            # Создаем datetime из индекса, если нет столбцов с датой
            df['datetime'] = pd.to_datetime(df.index)
    else:
        df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Базовые индикаторы тренда
    df['MA7'] = df['close'].rolling(window=7).mean()
    df['MA14'] = df['close'].rolling(window=14).mean()
    df['MA21'] = df['close'].rolling(window=21).mean()
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # Индикаторы волатильности
    df['daily_return'] = df['close'].pct_change()
    df['volatility'] = df['daily_return'].rolling(window=14).std()
    df['ATR'] = calculate_atr(df, period=14)
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = calculate_bollinger_bands(df, period=20)
    
    # Индикаторы импульса
    df['RSI'] = calculate_rsi(df['close'])
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['Signal_Line']
    
    # Объемные индикаторы
    df['OBV'] = calculate_obv(df)
    df['volume_MA7'] = df['volume'].rolling(window=7).mean()

     # Добавить индикаторы:
    df['price_momentum'] = df['close'] / df['close'].shift(1) - 1
    df['volume_momentum'] = df['volume'] / df['volume'].rolling(window=5).mean()
    df['high_low_ratio'] = df['high'] / df['low']
    
    # Добавить временные признаки
    df['hour'] = pd.to_datetime(df['datetime']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['datetime']).dt.dayofweek
    
    return df

def calculate_atr(df, period=14):
    """Расчет Average True Range"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_bollinger_bands(df, period=20, std_dev=2):
    """Расчет линий Боллинджера"""
    middle = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower

def calculate_rsi(prices, period=14):
    """Улучшенный расчет RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_obv(df):
    """Расчет On Balance Volume"""
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    return obv


def prepare_sequences(scaled_data, sequence_length):
    """Улучшенная подготовка последовательностей"""
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        sequence = scaled_data[i-sequence_length:i]
        target_price = scaled_data[i, 3]  # close price
        
        # Добавляем проверку на резкие изменения
        price_change = abs(target_price - sequence[-1, 3])
        if price_change > 0.1:  # Если изменение больше 10%
            continue
            
        X.append(sequence)
        y.append(target_price)
    return np.array(X), np.array(y)

def split_data(X, y, interval, test_days=3, val_days=2):
    """Разделение с учетом валидационной выборки"""
    test_size = samples_per_day[interval] * test_days
    val_size = samples_per_day[interval] * val_days
    
    train_size = len(X) - (test_size + val_size)
    
    X_train = X[:train_size]
    X_val = X[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    
    y_train = y[:train_size]
    y_val = y[train_size:train_size+val_size]
    y_test = y[train_size+val_size:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_sequence_length(interval):
    """Получение оптимальной длины последовательности для разных интервалов"""
    sequence_lengths = {
        "1min": 30,
        "5min": 30,
        "15min": 30,  # Уменьшаем для 15-минутного интервала
        "30min": 30,  # Уменьшаем для 30-минутного интервала
        "hour": 30
    }
    return sequence_lengths.get(interval, 60)

def prepare_data(symbol, interval):
    """Подготовка данных для обучения"""
    # Загрузка данных
    filename = os.path.join(RAW_DATA_PATH, symbol+"_"+ f"{interval}.csv")
    print(f"Загрузка файла: {filename}")
    
    try:
        df = pd.read_csv(filename)
        total_records = len(df)
        print(f"Всего записей в файле: {total_records}")
        
        # Добавление технических индикаторов
        df = add_technical_indicators(df)
        print(f"Размер после добавления индикаторов: {df.shape}")
        
        # Удаление строк с NaN значениями
        df = df.dropna()
        print(f"Размер после удаления NaN: {df.shape}")
        
        # Проверка наличия данных
        if df.empty:
            print(f"Нет данных для {symbol} - {interval}")
            return np.array([]), np.array([]), np.array([]), np.array([]), None
        
        price_features = ['open', 'high', 'low', 'close']
        volume_features = ['volume', 'OBV', 'volume_MA7']
        technical_features = [
            'MA7', 'MA14', 'MA21', 'EMA12', 'EMA26',
            'RSI', 'MACD', 'Signal_Line', 'MACD_hist',
            'volatility', 'ATR',
            'BB_upper', 'BB_middle', 'BB_lower',
            'daily_return'
        ]
        
        features = price_features + volume_features + technical_features
        
        # Проверка наличия всех признаков
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            print(f"Отсутствуют признаки для {symbol}: {missing_features}")
            print(f"Доступные колонки: {df.columns.tolist()}")
            return np.array([]), np.array([]), np.array([]), np.array([]), None
        
         # Создаем массив для нормализованных данных ОДИН раз
        scaled_data = np.zeros_like(df[features].values)
        
        # 1. Нормализация ценовых признаков
        price_scaler = MinMaxScaler()
        price_indices = [features.index(f) for f in price_features]
        scaled_data[:, price_indices] = price_scaler.fit_transform(df[price_features])
        
        # 2. Нормализация объемных признаков
        volume_scaler = MinMaxScaler()
        volume_indices = [features.index(f) for f in volume_features]
        scaled_data[:, volume_indices] = volume_scaler.fit_transform(df[volume_features])
        
        # 3. Обработка технических индикаторов
        tech_scalers = {}
        for feature in technical_features:
            idx = features.index(feature)
            if feature in ['RSI']:
                scaled_data[:, idx] = df[feature] / 100.0
            elif feature in ['MACD', 'Signal_Line', 'MACD_hist', 'daily_return']:
                values = df[feature].values
                mean = np.mean(values)
                std = np.std(values)
                scaled_data[:, idx] = (values - mean) / (std + 1e-8)
            else:
                tech_scaler = MinMaxScaler()
                scaled_data[:, idx] = tech_scaler.fit_transform(df[feature].values.reshape(-1, 1)).ravel()
                tech_scalers[feature] = tech_scaler
        
        # Сохраняем все скейлеры
        scalers = {
            'price': price_scaler,
            'volume': volume_scaler,
            'technical': tech_scalers,
            'features': features,
            'price_features': price_features,
            'volume_features': volume_features,
            'technical_features': technical_features
        }

        sequence_length = get_sequence_length(interval)
        print(f"Используется длина последовательности: {sequence_length}")
        
        # Проверка достаточности данных
        if len(scaled_data) <= sequence_length:
            print(f"Недостаточно данных для {symbol}. Необходимо: {sequence_length}, Имеется: {len(scaled_data)}")
            return np.array([]), np.array([]), np.array([]), np.array([]), None

        # Создание последовательностей для LSTM
        X, y = prepare_sequences(scaled_data, sequence_length)
        print(f"Размер X после создания последовательностей: {X.shape}")
        print(f"Размер y после создания последовательностей: {y.shape}")
    
         # Обновленный вызов split_data с передачей interval
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, interval, test_days=3, val_days=2)
        
        print(f"Использовано для обучения {symbol}: {len(X_train)} записей")
        print(f"Использовано для валидации {symbol}: {len(X_val)} записей")
        print(f"Использовано для тестирования {symbol}: {len(X_test)} записей (примерно 3 дня)")
        
        # Обновляем возвращаемые значения, чтобы включить валидационную выборку
        return X_train, X_val, X_test, y_train, y_val, y_test, scalers
        
    except FileNotFoundError:
        print(f"Файл не найден: {filename}")
        empty_array = np.array([])
        return empty_array, empty_array, empty_array, empty_array, empty_array, empty_array, None
    except Exception as e:
        print(f"Ошибка при обработке {symbol} - {interval}: {str(e)}")
        empty_array = np.array([])
        return empty_array, empty_array, empty_array, empty_array, empty_array, empty_array, None
    
def save_data(X_train, X_val, X_test, y_train, y_val, y_test, symbol, interval, sequence_length, scalers):
    """Сохранение данных и метаданных"""
    symbol_path = os.path.join(PROCESSED_DATA_PATH, symbol)
    if not os.path.exists(symbol_path):
        os.makedirs(symbol_path)
    
    # Сохранение данных, включая валидационную выборку
    np.save(os.path.join(symbol_path, f"X_train_{interval}.npy"), X_train)
    np.save(os.path.join(symbol_path, f"X_val_{interval}.npy"), X_val)
    np.save(os.path.join(symbol_path, f"X_test_{interval}.npy"), X_test)
    np.save(os.path.join(symbol_path, f"y_train_{interval}.npy"), y_train)
    np.save(os.path.join(symbol_path, f"y_val_{interval}.npy"), y_val)
    np.save(os.path.join(symbol_path, f"y_test_{interval}.npy"), y_test)
    
    # Обновленные метаданные
    metadata = {
        'symbol': symbol,
        'interval': interval,
        'sequence_length': sequence_length,
        'input_shape': X_train.shape[1:] if X_train.size > 0 else None,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'features': scalers['features'],
        'price_features': scalers['price_features'],
        'volume_features': scalers['volume_features'],
        'technical_features': scalers['technical_features']
    }
    
    with open(os.path.join(symbol_path, f"metadata_{interval}.json"), 'w') as f:
        json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    for symbol in STOCK_SYMBOLS:
        print(f"\n{'='*50}")
        print(f"Обработка данных для тикера {symbol}")
        print(f"{'='*50}")
        
        for interval in TIME_INTERVALS:
            print(f"\n{'-'*30}")
            print(f"Интервал: {interval}")
            print(f"{'-'*30}")
            
            try:
                X_train, X_val, X_test, y_train, y_val, y_test, scalers = prepare_data(symbol, interval)
                if X_train.size > 0:
                    save_data(X_train, X_val, X_test, y_train, y_val, y_test, 
                             symbol, interval, get_sequence_length(interval), scalers)
                    print(f"Размер обучающей выборки: {X_train.shape}")
                    print(f"Размер валидационной выборки: {X_val.shape}")
                    print(f"Размер тестовой выборки: {X_test.shape}")
                else:
                    print(f"Не удалось создать выборки для {symbol} - {interval}")
            except Exception as e:
                print(f"Ошибка при обработке {symbol} - {interval}: {str(e)}")