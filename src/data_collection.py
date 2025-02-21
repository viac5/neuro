from tinkoff.invest import Client
from tinkoff.invest.utils import now
import pandas as pd
from datetime import datetime, timedelta
import os
import time
import sys
import time
from tinkoff.invest.utils import quotation_to_decimal
from tinkoff.invest.schemas import CandleInterval
#############          PYTHON 3.12.8            ###############
# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

def get_figi_by_ticker(client, ticker):
    """Получение FIGI по тикеру с учетом класса акции"""
    try:
        # Получаем все акции
        shares = client.instruments.shares().instruments
        
        # Ищем конкретную акцию
        for share in shares:
            if share.ticker == ticker:
                # Для российских акций проверяем, что это основной класс (TQBR)
                if share.class_code == 'TQBR':
                    print(f"Найден FIGI для {ticker}: {share.figi} (TQBR)")
                    return share.figi
                
        print(f"Не найден подходящий инструмент для {ticker}")
        return None
        
    except Exception as e:
        print(f"Ошибка при поиске FIGI для {ticker}: {e}")
        return None

def convert_candle_to_dict(candle):
    """Конвертация свечи в словарь"""
    return {
        'time': candle.time,
        'open': float(quotation_to_decimal(candle.open)),
        'high': float(quotation_to_decimal(candle.high)),
        'low': float(quotation_to_decimal(candle.low)),
        'close': float(quotation_to_decimal(candle.close)),
        'volume': candle.volume
    }

def get_interval_enum(interval_str):
    """Конвертация строкового интервала в enum"""
    interval_map = {
        "1min": CandleInterval.CANDLE_INTERVAL_1_MIN,
        "5min": CandleInterval.CANDLE_INTERVAL_5_MIN,
        "15min": CandleInterval.CANDLE_INTERVAL_15_MIN,
        "30min": CandleInterval.CANDLE_INTERVAL_30_MIN,
        "hour": CandleInterval.CANDLE_INTERVAL_HOUR
    }
    return interval_map.get(interval_str)

def is_trading_day(date):
    """Проверка, является ли день торговым"""
    return date.weekday() < 5  # 0-4 это пн-пт

def is_trading_hours(datetime, symbol):
    """Проверка времени торгов с учетом биржи"""
    # Преобразуем в МСК (UTC+3)
    msk_time = datetime + timedelta(hours=3)
    
    # Для российских акций (MOEX)
    russian_stocks = ["SBER","GAZP", "SELG","SMLT","UGLD","LKOH","VKCO","T","ROSN","AFLT","MGNT","NVTK","VTBR","YDEX","X5","OZON"]
    if symbol in russian_stocks:
        # Торговая сессия MOEX: 10:00-18:50 МСК
        return (
            (msk_time.hour > 10 or (msk_time.hour == 10 and msk_time.minute >= 0)) and
            (msk_time.hour < 18 or (msk_time.hour == 18 and msk_time.minute <= 50))
        )
    else:
        # Для иностранных акций оставляем текущую логику
        return (
            msk_time.hour > 16 or (msk_time.hour == 16 and msk_time.minute >= 30)
        ) and (
            msk_time.hour < 23
        )

def get_max_days_for_interval(interval_str):
    """Получение максимального периода для интервала"""
    max_days = {
        "1min": 90,     # 7 дней для 1-минутных свечей
        "5min": 120,    # 15 дней для 5-минутных свечей
        "15min": 270,   # 15 дней для 15-минутных свечей
        "30min": 360,   # 30 дней для 30-минутных свечей
        "hour": 480     # 60 дней для часовых свечей
    }
    return max_days.get(interval_str, 1)

def get_expected_records(interval):
    """Расчет ожидаемого количества записей для интервала"""
    trading_minutes = 520  # 8 часов 40 минут (с 10:00 до 18:40)
    
    interval_minutes = {
        "1min": 1,
        "5min": 5,
        "15min": 15,
        "30min": 30,
        "hour": 60
    }
    
    minutes = interval_minutes.get(interval, 1)
    return trading_minutes // minutes

def fetch_stock_data(client, figi, interval, days_back=30):
    """Получение исторических данных"""
    try:
        max_days = get_max_days_for_interval(interval)
        end_date = now()
        start_date = end_date - timedelta(days=min(days_back, max_days))
        
        # Проверяем, что конечная дата - торговый день
        while not is_trading_day(end_date.date()):
            end_date -= timedelta(days=1)
        
        # Если запрашиваем минутные данные, проверяем торговые часы
        if interval.endswith('min'):
            while not is_trading_hours(end_date):
                end_date -= timedelta(hours=1)
        
        interval_enum = get_interval_enum(interval)
        if not interval_enum:
            print(f"Неподдерживаемый интервал: {interval}")
            return None

        print(f"Запрашиваем данные с {start_date} по {end_date}")
        candles = client.market_data.get_candles(
            figi=figi,
            from_=start_date,
            to=end_date,
            interval=interval_enum
        )

        candles_data = [convert_candle_to_dict(candle) for candle in candles.candles]
        if candles_data:
            df = pd.DataFrame(candles_data)
            df.set_index('time', inplace=True)
            print(f"Успешно получены данные для интервала {interval}: {len(df)} записей")
            return df
        else:
            print(f"Нет данных для интервала {interval}")
            return None

    except Exception as e:
        print(f"Ошибка при получении данных: {e}")
        return None

def save_data(df, symbol, interval):
    """Сохранение данных в CSV файл"""
    if df is None or df.empty:
        print(f"Нет данных для сохранения ({symbol}_{interval})")
        return
        
    if not os.path.exists(RAW_DATA_PATH):
        os.makedirs(RAW_DATA_PATH)
        
    filename = f"{RAW_DATA_PATH}{symbol}_{interval}.csv"
    df.to_csv(filename)
    print(f"Данные сохранены в {filename}")

def collect_data_for_day(client, figi, symbol, interval, date):
    """Получение данных за один торговый день"""
    if not is_trading_day(date.date()):
        return None
        
    # Устанавливаем время с учетом биржи
    russian_stocks = ['SBER', 'GAZP', 'LKOH', 'YNDX', 'VTBR', 'ROSN', 'TCSG']
    if symbol in russian_stocks:
        # Для российских акций (MOEX: 10:00-18:50 МСК)
        start_time = date.replace(hour=7, minute=0, second=0, microsecond=0)   # 10:00 МСК
        end_time = date.replace(hour=15, minute=50, second=0, microsecond=0)   # 18:50 МСК
    else:
        # Для иностранных акций
        start_time = date.replace(hour=13, minute=30, second=0, microsecond=0) # 16:30 МСК
        end_time = date.replace(hour=20, minute=0, second=0, microsecond=0)    # 23:00 МСК
    
    try:
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                candles = client.market_data.get_candles(
                    figi=figi,
                    from_=start_time,
                    to=end_time,
                    interval=get_interval_enum(interval)
                )
                
                if not candles.candles:
                    print(f"Нет данных за {date.date()} для {symbol}")
                    if attempt < max_retries - 1:
                        #time.sleep(retry_delay)
                        continue
                    return None
                    
                df = pd.DataFrame([convert_candle_to_dict(candle) for candle in candles.candles])
                df.set_index('time', inplace=True)
                print(f"Получено {len(df)} записей за {date.date()} для {symbol}")
                return df
                
            except Exception as e:
                print(f"Попытка {attempt + 1} не удалась для {symbol}: {e}")
                if attempt < max_retries - 1:
                    #time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise
                    
    except Exception as e:
        print(f"Ошибка при получении данных за {date.date()} для {symbol}: {e}")
        return None


def collect_all_data():
    """Сбор данных для всех тикеров и интервалов"""
    with Client(TINKOFF_TOKEN) as client:
        for symbol in STOCK_SYMBOLS:
            print(f"\n{'='*50}")
            print(f"Обработка тикера: {symbol}")
            print(f"{'='*50}")
            
            figi = get_figi_by_ticker(client, symbol)
            if not figi:
                print(f"Не удалось найти FIGI для {symbol}")
                continue
            
            for interval in TIME_INTERVALS:
                print(f"\nПолучение данных для {symbol} - интервал {interval}...")
                max_days = get_max_days_for_interval(interval)
                all_data = []
                current_date = now()
                
                for day in range(max_days):
                    date = current_date - timedelta(days=day)
                    if is_trading_day(date.date()):
                        df = collect_data_for_day(client, figi, symbol, interval, date)
                        if df is not None and not df.empty:
                            all_data.append(df)
                        time.sleep(0.1)  # Увеличенная пауза для MOEX
                
                if all_data:
                    combined_df = pd.concat(all_data)
                    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
                    combined_df = combined_df.sort_index()
                    save_data(combined_df, symbol, interval)
                else:
                    print(f"Нет данных для сохранения ({symbol}_{interval})")
            
            print(f"\nОбработка тикера {symbol} завершена")
            time.sleep(1)  # Пауза между тикерами

def list_available_stocks(client):
    """Вывод списка доступных акций"""
    instruments = client.instruments.shares().instruments
    print("\nДоступные акции:")
    for instrument in instruments:
        if instrument.trading_status == 1:  # Торгуется
            print(f"Тикер: {instrument.ticker:<6} | Название: {instrument.name:<30} | FIGI: {instrument.figi}")


if __name__ == "__main__":
    collect_all_data()