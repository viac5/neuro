from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QComboBox, QLineEdit, QPushButton, QLabel, QFrame,
                            QToolTip)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtCharts import (QChart, QChartView, QCandlestickSet, 
                            QCandlestickSeries, QValueAxis, QDateTimeAxis,
                            QLineSeries)
from PyQt6.QtCore import QDateTime, QPoint, QPointF, QRectF
from PyQt6.QtGui import QPainter, QColor, QBrush, QPen
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict import TinkoffClient, prepare_data, get_prediction, load_model
from config import *

class PredictionWorker(QThread):
    """Отдельный поток для выполнения предсказаний"""
    finished = pyqtSignal(tuple)
    error = pyqtSignal(str)
    data_ready = pyqtSignal(list)  # New signal for chart data
    
    def __init__(self, ticker, interval):
        super().__init__()
        self.ticker = ticker
        self.interval = interval
        
    def run(self):
        try:
            client = TinkoffClient()
            # Get candles first for the chart
            candles = client.get_candles(self.ticker, self.interval, days=3)
            if not candles:
                self.error.emit("Не удалось получить данные")
                return
                
            # Emit candles data for the chart
            self.data_ready.emit(candles)
            
            # Continue with prediction
            model = load_model(self.interval)
            if not model:
                self.error.emit("Ошибка загрузки модели")
                return
                
            X, mean, std = prepare_data(candles)
            if X is None:
                self.error.emit("Ошибка подготовки данных")
                return
                
            price_pred, lower_bound, upper_bound = get_prediction(model, X, mean, std)
            if price_pred is None:
                self.error.emit("Ошибка получения предсказания")
                return
                
            self.finished.emit((price_pred, lower_bound, upper_bound))
            
        except Exception as e:
            self.error.emit(str(e))

class ChartWindow(QMainWindow):
    def __init__(self, candles):
        super().__init__()
        self.setWindowTitle("График цены")
        self.setMinimumSize(800, 600)
        
        # Create chart
        self.chart = QChart()
        self.chart.setBackgroundBrush(QBrush(QColor("#1e1e1e")))
        self.chart.setTitleBrush(QBrush(QColor("white")))
        self.chart.legend().hide()
        
        # Create candlestick series with improved visibility
        self.series = QCandlestickSeries()
        
        # Set colors without outline
        increasing_color = QColor("#26a69a")  # Green
        decreasing_color = QColor("#ef5350")  # Red
        
        self.series.setIncreasingColor(increasing_color)
        self.series.setDecreasingColor(decreasing_color)
        
        # Adjust width and pen
        self.series.setBodyWidth(0.5)  # Make candles wider
        self.series.setPen(QPen(Qt.PenStyle.NoPen))  # Remove outline
        
        # Add data
        min_price = float('inf')
        max_price = float('-inf')
        for candle in candles:
            timestamp = QDateTime.fromString(candle["time"], Qt.DateFormat.ISODate)
            candlestick_set = QCandlestickSet(
                float(candle["open"]),
                float(candle["high"]),
                float(candle["low"]),
                float(candle["close"]),
                timestamp.toMSecsSinceEpoch()
            )
            self.series.append(candlestick_set)
            min_price = min(min_price, float(candle["low"]))
            max_price = max(max_price, float(candle["high"]))
        
        self.chart.addSeries(self.series)
        
        # Create axes with improved visibility
        axis_x = QDateTimeAxis()
        axis_x.setFormat("dd.MM\nHH:mm")
        axis_x.setTitleText("Время")
        axis_x.setLabelsColor(QColor("white"))
        axis_x.setTitleBrush(QBrush(QColor("white")))
        axis_x.setGridLineColor(QColor("#404040"))
        self.chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        self.series.attachAxis(axis_x)
        
        # Add padding to price range
        price_padding = (max_price - min_price) * 0.05
        axis_y = QValueAxis()
        axis_y.setTitleText("Цена")
        axis_y.setRange(min_price - price_padding, max_price + price_padding)
        axis_y.setLabelsColor(QColor("white"))
        axis_y.setTitleBrush(QBrush(QColor("white")))
        axis_y.setGridLineColor(QColor("#404040"))
        axis_y.setLabelFormat("%.2f")
        self.chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        self.series.attachAxis(axis_y)
        
        # Create chart view with improved rendering
        self.chartview = QChartView(self.chart)
        self.chartview.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.chartview.setBackgroundBrush(QBrush(QColor("#1e1e1e")))
        self.setCentralWidget(self.chartview)
        
        # Enable tooltip
        self.chartview.setMouseTracking(True)
        self.chartview.mouseMoveEvent = self.show_tooltip

        # Enable zoom with mouse wheel
        self.chartview.wheelEvent = self.zoom_chart
        
    def show_tooltip(self, event):
        pos = QPointF(event.pos())
        value = self.chart.mapToValue(pos)
        
        for set in self.series.sets():
            rect = self.chart.plotArea()
            x_pos = self.chart.mapToPosition(QPointF(set.timestamp(), 0)).x()
            if abs(x_pos - pos.x()) < 5:  # 5 pixels tolerance
                dt = QDateTime.fromMSecsSinceEpoch(int(set.timestamp()))
                tooltip = (f"Время: {dt.toString('dd.MM.yyyy HH:mm')}\n"
                         f"Открытие: {set.open():.2f}\n"
                         f"Максимум: {set.high():.2f}\n"
                         f"Минимум: {set.low():.2f}\n"
                         f"Закрытие: {set.close():.2f}")
                # Use globalPosition() instead of globalPos()
                global_pos = event.globalPosition().toPoint()
                QToolTip.showText(global_pos, tooltip)
                return
        QToolTip.hideText()
    
    def zoom_chart(self, event):
        # Determine the zoom factor
        zoom_in_factor = 1.1
        zoom_out_factor = 0.9
        
        # Zoom in or out based on the direction of the wheel
        if event.angleDelta().y() > 0:
            self.chartview.chart().zoom(zoom_in_factor)
        else:
            self.chartview.chart().zoom(zoom_out_factor)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Прогноз цен акций")
        self.setMinimumSize(400, 300)
        
        # Создаем центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Основной layout
        layout = QVBoxLayout(central_widget)
        
        # Интервал
        interval_layout = QHBoxLayout()
        interval_label = QLabel("Интервал:")
        interval_label.setFixedWidth(60)  # Fixed width for label
        self.interval_combo = QComboBox()
        self.interval_combo.setFixedWidth(100)  # Fixed width for combo
        self.interval_combo.addItems(TIME_INTERVALS)
        interval_layout.addWidget(interval_label)
        interval_layout.addWidget(self.interval_combo)
        interval_layout.addStretch()  # Add stretch to align left
        layout.addLayout(interval_layout)
        
        # Тикер
        ticker_layout = QHBoxLayout()
        ticker_label = QLabel("Тикер:")
        ticker_label.setFixedWidth(60)  # Same width as interval label
        self.ticker_input = QLineEdit()
        self.ticker_input.setFixedWidth(100)  # Same width as combo
        self.ticker_input.setPlaceholderText("SBER")
        ticker_layout.addWidget(ticker_label)
        ticker_layout.addWidget(self.ticker_input)
        ticker_layout.addStretch()  # Add stretch to align left
        layout.addLayout(ticker_layout)
        
        # Кнопки
        buttons_layout = QHBoxLayout()
        self.predict_button = QPushButton("Получить прогноз")
        self.predict_button.clicked.connect(self.start_prediction)
        self.chart_button = QPushButton("Открыть график")
        self.chart_button.setEnabled(False)
        self.chart_button.clicked.connect(self.show_chart)
        buttons_layout.addWidget(self.predict_button)
        buttons_layout.addWidget(self.chart_button)
        layout.addLayout(buttons_layout)
        
        # Результаты
        results_frame = QFrame()
        results_frame.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Sunken)
        results_layout = QVBoxLayout(results_frame)
        
        self.price_label = QLabel("Предсказанная цена: -")
        self.bounds_label = QLabel("Доверительный интервал: -")
        
        results_layout.addWidget(self.price_label)
        results_layout.addWidget(self.bounds_label)
        layout.addWidget(results_frame)
        
        # Статус
        self.status_label = QLabel()
        layout.addWidget(self.status_label)
        
        # Initialize variables
        self.worker = None
        self.candles = None
        self.chart_window = None
        
    def show_chart(self):
        """Show chart window"""
        if self.candles:
            self.chart_window = ChartWindow(self.candles)
            self.chart_window.show()
            
    def start_prediction(self):
        ticker = self.ticker_input.text().upper()
        if not ticker:
            self.status_label.setText("Введите тикер")
            return
            
        interval = self.interval_combo.currentText()
        
        self.predict_button.setEnabled(False)
        self.chart_button.setEnabled(False)
        self.status_label.setText("Выполняется прогноз...")
        
        self.worker = PredictionWorker(ticker, interval)
        self.worker.finished.connect(self.handle_results)
        self.worker.error.connect(self.handle_error)
        self.worker.data_ready.connect(self.update_chart_data)
        self.worker.start()
        
    def update_chart_data(self, candles):
        """Store candles data and enable chart button"""
        self.candles = candles
        self.chart_button.setEnabled(True)
        
    def handle_results(self, results):
        price_pred, lower_bound, upper_bound = results
        self.price_label.setText(f"Предсказанная цена: {price_pred:.2f}")
        self.bounds_label.setText(f"Доверительный интервал: [{lower_bound:.2f} - {upper_bound:.2f}]")
        self.status_label.setText("Прогноз выполнен")
        self.predict_button.setEnabled(True)
        
    def handle_error(self, error_msg):
        self.status_label.setText(f"Ошибка: {error_msg}")
        self.predict_button.setEnabled(True)
        self.chart_button.setEnabled(False)