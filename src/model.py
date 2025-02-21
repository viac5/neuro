import traceback
import tensorflow as tf
# Заменяем импорты на альтернативные
Sequential = tf.keras.Sequential
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Adam = tf.keras.optimizers.Adam
Add = tf.keras.layers.Add
BatchNormalization = tf.keras.layers.BatchNormalization
MultiHeadAttention = tf.keras.layers.MultiHeadAttention
LayerNormalization = tf.keras.layers.LayerNormalization
Input = tf.keras.layers.Input
Concatenate = tf.keras.layers.Concatenate
Conv1D = tf.keras.layers.Conv1D
Bidirectional = tf.keras.layers.Bidirectional
register_keras_serializable=tf.keras.utils.register_keras_serializable
import numpy as np
import json
import os
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


# Загрузка переменных окружения
def load_environment():
    """Загрузка переменных окружения"""
    try:
        # Путь к файлу .env
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        
        # Загружаем переменные из файла
        load_dotenv(env_path)
        
        # Проверяем загрузку
        tf_log_level = os.getenv('TF_CPP_MIN_LOG_LEVEL')
        tf_xla_flags = os.getenv('TF_XLA_FLAGS')
        tf_onednn_opts = os.getenv('TF_ENABLE_ONEDNN_OPTS')
        
        print("Загруженные переменные окружения:")
        print(f"TF_CPP_MIN_LOG_LEVEL: {tf_log_level}")
        print(f"TF_XLA_FLAGS: {tf_xla_flags}")
        print(f"TF_ENABLE_ONEDNN_OPTS: {tf_onednn_opts}")
        
    except Exception as e:
        print(f"Ошибка при загрузке переменных окружения: {e}")

def configure_gpu():
    """Настройка GPU и проверка доступности"""
    try:
        # Очищаем сессию и кэш GPU
        tf.keras.backend.clear_session()
        
        # Явно включаем GPU
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            # Разрешаем рост памяти GPU
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
                
            # Устанавливаем видимое устройство
            tf.config.set_visible_devices(physical_devices[0], 'GPU')
            
            # Устанавливаем конфигурацию по умолчанию
            tf.config.experimental.set_virtual_device_configuration(
                physical_devices[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
            )
            
            # Устанавливаем mixed precision
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            
            print(f"Найдено GPU устройств: {len(physical_devices)}")
            print("GPU успешно настроен с mixed_float16")
            
            # Проверяем GPU
            with tf.device('/GPU:0'):
                a = tf.random.normal([1000, 1000])
                b = tf.random.normal([1000, 1000])
                c = tf.matmul(a, b)
                print(f"Тестовое вычисление выполнено на: {c.device}")
                
            return True
        else:
            print("GPU не обнаружены!")
            return False
            
    except Exception as e:
        print(f"Ошибка при настройке GPU: {str(e)}")
        print(traceback.format_exc())
        return False

def setup_cuda_paths():
    """Настройка путей CUDA"""
    cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
    os.environ["CUDA_PATH"] = cuda_path
    os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={cuda_path}"
    
    # Проверяем наличие libdevice.10.bc
    libdevice_path = os.path.join(cuda_path, "nvvm", "libdevice", "libdevice.10.bc")
    if not os.path.exists("libdevice.10.bc") and os.path.exists(libdevice_path):
        try:
            os.symlink(libdevice_path, "libdevice.10.bc")
        except Exception as e:
            print(f"Предупреждение: Не удалось создать символическую ссылку: {e}")
            try:
                from shutil import copy2
                copy2(libdevice_path, "libdevice.10.bc")
            except Exception as e:
                print(f"Ошибка: Не удалось скопировать libdevice.10.bc: {e}")

def add_seasonal_features(x):
    """Добавление сезонных признаков с корректной обработкой типов данных"""
    class SeasonalFeatures(tf.keras.layers.Layer):
        def __init__(self, period=20):
            super().__init__()
            self.period = period
            
        def call(self, x):
            # Get input dtype to match it
            dtype = x.dtype
            batch_size = tf.shape(x)[0]
            seq_len = tf.shape(x)[1]
            
            # Create seasonal component with matching dtype
            t = tf.cast(tf.range(seq_len), dtype)  # Cast directly to input dtype
            pi_const = tf.cast(np.pi, dtype)  # Cast pi to input dtype
            period = tf.cast(self.period, dtype)  # Cast period to input dtype
            
            # Calculate seasonal component with matching dtype
            seasonal = tf.sin(2.0 * pi_const * t / period)
            seasonal = tf.cast(seasonal, dtype)  # Ensure dtype consistency
            seasonal = tf.expand_dims(seasonal, 0)
            seasonal = tf.expand_dims(seasonal, -1)
            seasonal = tf.tile(seasonal, [batch_size, 1, 1])
            
            # Both tensors should now have the same dtype (float16)
            return tf.concat([x, seasonal], axis=-1)
            
        def compute_output_shape(self, input_shape):
            return (input_shape[0], input_shape[1], input_shape[2] + 1)
        
        def get_config(self):
            config = super().get_config()
            config.update({"period": self.period})
            return config
    
    return SeasonalFeatures()(x)

def check_environment():
    """Проверка переменных окружения"""
    variables = {
        'TF_CPP_MIN_LOG_LEVEL': os.getenv('TF_CPP_MIN_LOG_LEVEL'),
        'TF_XLA_FLAGS': os.getenv('TF_XLA_FLAGS'),
        'TF_ENABLE_ONEDNN_OPTS': os.getenv('TF_ENABLE_ONEDNN_OPTS')
    }
    
    print("\nТекущие переменные окружения:")
    for name, value in variables.items():
        print(f"{name}: {value}")

@register_keras_serializable()
class TechnicalIndicators(tf.keras.layers.Layer):
        def call(self, price_data):
            # RSI
            delta = price_data[:, 1:] - price_data[:, :-1]
            gain = tf.maximum(delta, 0)
            loss = tf.maximum(-delta, 0)
            
            # Правильный расчет средних значений
            avg_gain = tf.reduce_mean(gain, axis=1, keepdims=True)
            avg_loss = tf.reduce_mean(loss, axis=1, keepdims=True)
            rs = avg_gain / (avg_loss + 1e-7)
            
            # RSI с правильной размерностью
            rsi = 100 - (100 / (1 + rs))
            rsi = tf.tile(rsi, [1, tf.shape(price_data)[1], 1])
            
            # MACD с корректными размерностями
            seq_len = tf.shape(price_data)[1]
            # Используем tf.slice вместо Python indexing
            last_12 = tf.maximum(seq_len - 12, 0)
            last_26 = tf.maximum(seq_len - 26, 0)
            
            ema12 = tf.reduce_mean(
                tf.slice(price_data, [0, last_12, 0], [-1, -1, -1]), 
                axis=1, 
                keepdims=True
            )
            ema26 = tf.reduce_mean(
                tf.slice(price_data, [0, last_26, 0], [-1, -1, -1]), 
                axis=1, 
                keepdims=True
            )
            macd = ema12 - ema26
            macd = tf.tile(macd, [1, seq_len, 1])
            
            # Bollinger Bands
            sma = tf.reduce_mean(price_data, axis=1, keepdims=True)
            squared_diff = tf.square(price_data - sma)
            std = tf.sqrt(tf.reduce_mean(squared_diff, axis=1, keepdims=True))
            
            upper_band = sma + (2 * std)
            lower_band = sma - (2 * std)
            
            # Расширяем размерности для bands
            upper_band = tf.tile(upper_band, [1, seq_len, 1])
            lower_band = tf.tile(lower_band, [1, seq_len, 1])
            
            # Объединяем все индикаторы
            return tf.concat([
                price_data,
                rsi,
                macd,
                upper_band,
                lower_band
            ], axis=-1)
        
        def compute_output_shape(self, input_shape):
            return (input_shape[0], input_shape[1], input_shape[2] * 5)
        
        def __init__(self, **kwargs):
            super(TechnicalIndicators, self).__init__(**kwargs)
            
        def get_config(self):
            config = super(TechnicalIndicators, self).get_config()
            return config


@register_keras_serializable()
def custom_loss(y_true, y_pred):
    """Комбинированная функция потерь для многомерного выхода с правильной редукцией"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # MSE для цены
    mse_price = tf.reduce_mean(tf.square(y_true[..., 0] - y_pred[..., 0]))
    
    # Huber loss с явным указанием reduction
    huber = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
    huber_bounds = huber(y_true[..., 1:], y_pred[..., 1:])
    huber_bounds = tf.reduce_mean(huber_bounds)
    
    # Комбинируем потери
    total_loss = 0.7 * mse_price + 0.3 * huber_bounds
    
    # Получаем глобальный размер батча для правильной нормализации
    global_batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
    
    return total_loss * (1. / global_batch_size)

def technical_layer(price_data):
    """Расширенный набор технических индикаторов"""
    return TechnicalIndicators()(price_data)

def multi_scale_attention(x, num_heads=8):
    """Механизм внимания для разных временных масштабов"""
    # Краткосрочное внимание
    attention_short = MultiHeadAttention(num_heads=num_heads, key_dim=32)(x, x)
    
    # Среднесрочное внимание (уменьшаем временное разрешение)
    x_medium = tf.keras.layers.AveragePooling1D(3)(x)
    attention_medium = MultiHeadAttention(num_heads=num_heads, key_dim=32)(x_medium, x_medium)
    attention_medium = tf.keras.layers.UpSampling1D(3)(attention_medium)
    
    # Объединяем разные масштабы
    return Add()([attention_short, attention_medium])

def directional_loss(y_true, y_pred):
    """Функция потерь, учитывающая направление движения цены"""
    # Cast inputs to float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Get price values (first dimension)
    y_true_price = y_true[..., 0]
    y_pred_price = y_pred[..., 0]
    
    # Calculate MSE for all outputs
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Calculate directional penalty only for price
    # Ensure we have a batch dimension when calculating differences
    true_diff = y_true_price[:, 1:] - y_true_price[:, :-1]
    pred_diff = y_pred_price[:, 1:] - y_pred_price[:, :-1]
    
    direction_true = tf.sign(true_diff)
    direction_pred = tf.sign(pred_diff)
    direction_penalty = tf.reduce_mean(tf.abs(direction_true - direction_pred))
    
    # Combine losses with weights
    total_loss = mse + 0.2 * direction_penalty
    
    return total_loss


def create_model(input_shape, learning_rate=0.001):
    """Создание улучшенной модели для предсказания цен"""
    with tf.device('/GPU:0'):
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        
        inputs = Input(shape=input_shape, dtype=tf.float16)
    
        # Разделение входных данных по типам
        price_features = inputs[:, :, :4]  # OHLC
        price_features = technical_layer(price_features)
        volume_features = inputs[:, :, 4:7]  # объемы
        tech_features = inputs[:, :, 7:]  # технические индикаторы
    
        # Добавляем сезонные признаки
        x = add_seasonal_features(inputs)
    
    
        # Улучшенная обработка ценовых данных
        price_conv = Conv1D(32, 3, padding='same', activation='relu')(price_features)
        price_lstm = Bidirectional(LSTM(64, 
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            recurrent_regularizer=tf.keras.regularizers.l1(1e-4)
        ))(price_conv)
    
        # Улучшенная обработка объемов
        volume_conv = Conv1D(16, 3, padding='same', activation='relu')(volume_features)
        volume_lstm = Bidirectional(LSTM(32, 
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        ))(volume_conv)
    
        # Улучшенная обработка технических индикаторов
        tech_conv = Conv1D(32, 3, padding='same', activation='relu')(tech_features)
        tech_lstm = Bidirectional(LSTM(64, return_sequences=True))(tech_conv)
    
        # Объединение признаков
        x = Concatenate()([price_lstm, volume_lstm, tech_lstm])
    
        # Улучшенный механизм внимания
        x = multi_scale_attention(x)  # Добавляем здесь
        x = LayerNormalization()(x)
    
        x = Conv1D(64, 3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
    
        # Добавляем squeeze-and-excitation блок
        se = tf.keras.layers.GlobalAveragePooling1D()(x)
        se = Dense(64 // 4, activation='relu')(se)
        se = Dense(64, activation='sigmoid')(se)
        se = tf.keras.layers.Reshape((1, 64))(se)
        x = tf.keras.layers.multiply([x, se])

        # Дополнительные слои обработки
        x = Bidirectional(LSTM(128, 
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            activity_regularizer=tf.keras.regularizers.l1(1e-5)
        ))(x)
        x = LayerNormalization()(x)
        x = Dropout(0.3)(x)
    
        # Финальные слои
        x_res = x
        x = LSTM(64)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Add()([x, Dense(64)(x_res[:,-1,:])])  # Skip connection
    
        # Выходной слой с множественными предсказаниями
        outputs = Dense(3, name='predictions')(x)  # Added name to the layer

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
        # Компиляция с новой функцией потерь
        optimizer = Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )   
    
        model.compile(
            optimizer=optimizer,
            loss=custom_loss,
            metrics=[
                tf.keras.metrics.MeanAbsoluteError(name='mean_absolute_error'),
                tf.keras.metrics.MeanAbsolutePercentageError(name='mean_absolute_percentage_error'),
                tf.keras.metrics.MeanSquaredError(name='mean_squared_error')
            ],
            # Добавляем название для метрик валидации
            weighted_metrics=[
                tf.keras.metrics.MeanSquaredError(name='val_mean_squared_error')
            ]
        )
        return model

def load_data(interval, symbol):
    """Загрузка обработанных данных для конкретного тикера"""
    symbol_path = os.path.join(PROCESSED_DATA_PATH, symbol)
    metadata_path = os.path.join(symbol_path, f"metadata_{interval}.json")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Файл метаданных не найден: {metadata_path}")
    
    # Загрузка метаданных
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Обновленный список файлов, включая валидационную выборку
    files = [
        os.path.join(symbol_path, f"X_train_{interval}.npy"),
        os.path.join(symbol_path, f"X_val_{interval}.npy"),
        os.path.join(symbol_path, f"X_test_{interval}.npy"),
        os.path.join(symbol_path, f"y_train_{interval}.npy"),
        os.path.join(symbol_path, f"y_val_{interval}.npy"),
        os.path.join(symbol_path, f"y_test_{interval}.npy")
    ]
    
    for file in files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Файл данных не найден: {file}")
    
    # Загрузка данных, включая валидационную выборку
    X_train = np.load(files[0])
    X_val = np.load(files[1])
    X_test = np.load(files[2])
    y_train = np.load(files[3])  # Add these lines to load y data
    y_val = np.load(files[4])    # Add these lines to load y data
    y_test = np.load(files[5])   # Add these lines to load y data
    
    # Reshape y data
    y_train = np.expand_dims(y_train, axis=-1)
    y_val = np.expand_dims(y_val, axis=-1)
    y_test = np.expand_dims(y_test, axis=-1)
    
    # Create bounds (example: ±2% of the price)
    y_train = np.concatenate([
        y_train,  # predicted price
        y_train * 0.98,  # lower bound
        y_train * 1.02   # upper bound
    ], axis=-1)
    
    y_val = np.concatenate([
        y_val,
        y_val * 0.98,
        y_val * 1.02
    ], axis=-1)
    
    y_test = np.concatenate([
        y_test,
        y_test * 0.98,
        y_test * 1.02
    ], axis=-1)
    
    print(f"Data shapes for {interval}_{symbol}:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")  # Should now be (samples, 3)
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")  # Should now be (samples, 3)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, metadata

class WarmUpCallback(tf.keras.callbacks.Callback):
    """Постепенное увеличение learning rate в начале обучения"""
    def __init__(self, warmup_epochs=5, initial_lr=0.001):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # Используем tf.keras.backend для установки learning rate
            if hasattr(self.model.optimizer, 'lr'):
                new_lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)

def get_callbacks(interval, initial_lr=0.001):
    """Оптимизированные callbacks для единой модели"""
    model_dir = os.path.join(MODELS_PATH, f"model_{interval}")
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    callbacks = [
        # Раннее останавливание
        tf.keras.callbacks.EarlyStopping(
            monitor='val_mean_squared_error',
            patience=PATIENCE,
            restore_best_weights=True,
            mode='min',
            min_delta=0.0001
        ),
        
        # Сохранение лучшей модели
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "best_weights"),
            monitor='val_mean_squared_error',
            save_best_only=True,
            save_weights_only=True,  # Сохраняем только веса для скорости
            mode='min'
        ),
        
        # Уменьшение learning rate
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_mean_squared_error',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            mode='min',
            verbose=1
        ),
        
        # Оптимизированное логирование
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_dir, 'logs'),
            update_freq=50,  # Логируем реже для скорости
            profile_batch=0
        ),
        
        WarmUpCallback(warmup_epochs=5, initial_lr=initial_lr)
    ]
    return callbacks

def train_model(interval):
    """Обучение единой модели последовательно на разных тикерах"""
    print(f"\nОбучение модели для интервала {interval}")
    
    try:
        # Configure GPU before creating strategy
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Configure the first GPU only
            gpu = gpus[0]
            try:
                # Разрешаем рост памяти перед другими настройками
                tf.config.experimental.set_memory_growth(gpu, True)
                
                # Устанавливаем видимое устройство
                tf.config.set_visible_devices(gpu, 'GPU')
                
                print(f"GPU настроен: {gpu}")
            except RuntimeError as e:
                print(f"Ошибка настройки GPU: {e}")
                return False
        else:
            raise RuntimeError("GPU не обнаружены")

        # Создаем стратегию для GPU после настройки
        strategy = tf.distribute.OneDeviceStrategy("/GPU:0")
        print(f"Используется стратегия: {strategy}")
        
        # Устанавливаем mixed precision
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        
        with strategy.scope():
            # Создаем модель на основе первого тикера
            first_symbol = STOCK_SYMBOLS[0]
            X_train, X_val, X_test, y_train, y_val, y_test, metadata = load_data(interval, first_symbol)
            input_shape = metadata['input_shape']
            
            # Создание единой модели
            model = create_model(input_shape)
            print("\nСтруктура модели:")
            model.summary()
            
            # Проверка использования GPU
            print("\nПроверка GPU:")
            print(f"TensorFlow видит GPU: {tf.test.is_built_with_cuda()}")
            print(f"Текущее устройство: {tf.test.gpu_device_name()}")
            
            # Get model device placement
            print("\nРазмещение слоев модели:")
            for layer in model.layers:
                print(f"{layer.name}: {layer.dtype_policy}")
            
            training_history = {}
            callbacks = get_callbacks(interval, LEARNING_RATE)
        
            for symbol in STOCK_SYMBOLS:
                print(f"\nДообучение на данных {symbol}")
                X_train, X_val, X_test, y_train, y_val, y_test, metadata = load_data(interval, symbol)
                
                # Convert data to float16
                X_train = tf.cast(X_train, tf.float16)
                y_train = tf.cast(y_train, tf.float16)
                X_val = tf.cast(X_val, tf.float16)
                y_val = tf.cast(y_val, tf.float16)
                
                with tf.device('/GPU:0'):
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        callbacks=callbacks,
                        verbose=1,
                        use_multiprocessing=True,
                        workers=4
                    )
            
                # Конвертируем значения в обычные Python типы
                history_dict = {}
                for key, value in history.history.items():
                    history_dict[key] = [float(v) for v in value]
                    training_history[symbol] = history_dict
            
                evaluation = model.evaluate(X_test, y_test, verbose=0)
                print(f"\nРезультаты тестирования для {symbol}:")
                print(f"Loss: {evaluation[0]:.4f}")
                print(f"MAE: {evaluation[1]:.4f}")
        
            # Сохраняем финальную модель и историю
            model_dir = os.path.join(MODELS_PATH, f"model_{interval}")
            model.save_weights(os.path.join(model_dir, "final_weights"), save_format='tf')
        
            # Сохраняем архитектуру отдельно
            model_json = model.to_json()
            with open(os.path.join(model_dir, "architecture.json"), 'w') as f:
                f.write(model_json)
            
            # Сохраняем преобразованную историю обучения
            with open(os.path.join(model_dir, "training_history.json"), 'w') as f:
                json.dump(training_history, f, indent=2)
        
            print(f"\nМодель сохранена: {model_dir}")
        
            return model, training_history
        
    except Exception as e:
        print(f"Ошибка при обучении модели: {str(e)}")
        raise


if __name__ == "__main__":
    # Настройка CUDA и окружения
    setup_cuda_paths()
    load_environment()
    
    # Устанавливаем корректные переменные окружения
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    
    # Исправляем флаги XLA
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2 --tf_xla_enable_xla_devices"
    
    # Очищаем сессию TensorFlow и проверяем GPU
    tf.keras.backend.clear_session()
    
    # Проверяем доступность GPU
    print("TensorFlow version:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices('GPU'))
    print(f"Доступные тикеры: {', '.join(STOCK_SYMBOLS)}")
    
    # Enable mixed precision
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    for interval in TIME_INTERVALS:
        print(f"\n{'='*50}")
        print(f"Обработка интервала: {interval}")
        print(f"{'='*50}")
        
        try:
            model, history = train_model(interval)
            print(f"Модель для интервала {interval} успешно обучена")
        except Exception as e:
            print(f"Ошибка при обучении модели для интервала {interval}: {e}")
            traceback.print_exc()

