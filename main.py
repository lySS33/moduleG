# main.py - обновленная версия с поддержкой видео файлов и камер
import sys
import os
import cv2
import numpy as np
import random
from datetime import datetime
from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread, QUrl
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QBrush, QColor, QFont, QDesktopServices
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget

class VideoFileHandler:
    """обработчик видео файлов"""
    
    def __init__(self):
        self.video_capture = None
        self.current_video_path = None
        self.video_fps = 30
        self.video_width = 640
        self.video_height = 480
        self.total_frames = 0
        self.current_frame = 0
    
    def load_video(self, file_path):
        """загрузка видео файла"""
        if self.video_capture:
            self.video_capture.release()
        
        self.video_capture = cv2.VideoCapture(file_path)
        if not self.video_capture.isOpened():
            return False
        
        self.current_video_path = file_path
        self.video_fps = int(self.video_capture.get(cv2.CAP_PROP_FPS))
        self.video_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        
        return True
    
    def get_next_frame(self):
        """получение следующего кадра"""
        if not self.video_capture or not self.video_capture.isOpened():
            return None
        
        ret, frame = self.video_capture.read()
        if ret:
            self.current_frame += 1
            return frame
        return None
    
    def seek_frame(self, frame_number):
        """переход к определенному кадру"""
        if not self.video_capture or not self.video_capture.isOpened():
            return False
        
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self.current_frame = frame_number
        return True
    
    def get_frame_position(self):
        """получение текущей позиции"""
        if not self.video_capture or not self.video_capture.isOpened():
            return 0
        return self.current_frame
    
    def get_progress(self):
        """получение прогресса воспроизведения"""
        if self.total_frames == 0:
            return 0
        return (self.current_frame / self.total_frames) * 100
    
    def release(self):
        """освобождение ресурсов"""
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
    
    def get_video_info(self):
        """получение информации о видео"""
        return {
            'path': self.current_video_path,
            'fps': self.video_fps,
            'width': self.video_width,
            'height': self.video_height,
            'total_frames': self.total_frames,
            'current_frame': self.current_frame,
            'duration': self.total_frames / self.video_fps if self.video_fps > 0 else 0
        }

class ColorDefectDetector:
    """детектор цветных дефектов на коробках"""
    
    def __init__(self):
        # цвета дефектов в формате HSV
        self.defect_colors_hsv = {
            'красный_дефект': ([0, 100, 100], [10, 255, 255]),
            'красный_дефект2': ([170, 100, 100], [180, 255, 255]),
            'синий_дефект': ([100, 100, 100], [130, 255, 255]),
            'зеленый_дефект': ([40, 100, 100], [80, 255, 255]),
            'желтый_дефект': ([20, 100, 100], [40, 255, 255]),
        }
        
        # параметры детекции
        self.min_contour_area = 50
        self.max_contour_area = 5000
        
        # цвет для отображения в BGR
        self.display_colors = {
            'красный': (0, 0, 255),
            'синий': (255, 0, 0),
            'зеленый': (0, 255, 0),
            'желтый': (0, 255, 255)
        }
        
        # статистика
        self.defect_count = 0
        self.red_count = 0
        self.blue_count = 0
        self.green_count = 0
        self.yellow_count = 0
    
    def detect_color_defects(self, frame):
        """обнаружение цветных дефектов"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detected_defects = []
        
        for defect_name, color_ranges in self.defect_colors_hsv.items():
            # обрабатываем диапазоны цветов
            if isinstance(color_ranges[0], list) and isinstance(color_ranges[1], list):
                masks = [cv2.inRange(hsv, 
                                   np.array(color_ranges[0], dtype=np.uint8),
                                   np.array(color_ranges[1], dtype=np.uint8))]
            else:
                masks = []
                for i in range(0, len(color_ranges), 2):
                    lower = np.array(color_ranges[i], dtype=np.uint8)
                    upper = np.array(color_ranges[i+1], dtype=np.uint8)
                    masks.append(cv2.inRange(hsv, lower, upper))
            
            if len(masks) > 1:
                mask = masks[0]
                for m in masks[1:]:
                    mask = cv2.bitwise_or(mask, m)
            else:
                mask = masks[0]
            
            # улучшаем маску
            mask = self.enhance_mask(mask)
            
            # находим контуры
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if self.min_contour_area < area < self.max_contour_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    if self.is_valid_defect(frame, x, y, w, h):
                        defect_info = self.analyze_defect(frame, contour, defect_name)
                        if defect_info:
                            detected_defects.append(defect_info)
                            
                            if 'красный' in defect_name:
                                self.red_count += 1
                            elif 'синий' in defect_name:
                                self.blue_count += 1
                            elif 'зеленый' in defect_name:
                                self.green_count += 1
                            elif 'желтый' in defect_name:
                                self.yellow_count += 1
        
        self.defect_count += len(detected_defects)
        return detected_defects
    
    def enhance_mask(self, mask):
        """улучшение маски"""
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask
    
    def is_valid_defect(self, frame, x, y, w, h):
        """проверка валидности дефекта"""
        height, width = frame.shape[:2]
        margin = 10
        
        if (x < margin or y < margin or 
            x + w > width - margin or 
            y + h > height - margin):
            return False
        
        if h > 0:
            aspect_ratio = w / h
            if aspect_ratio > 5 or aspect_ratio < 0.2:
                return False
        
        return True
    
    def analyze_defect(self, frame, contour, defect_name):
        """анализ дефекта"""
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        M = cv2.moments(contour)
        if M['m00'] != 0:
            center_x = int(M['m10'] / M['m00'])
            center_y = int(M['m01'] / M['m00'])
        else:
            center_x = x + w // 2
            center_y = y + h // 2
        
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return None
        
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray_roi)
        
        return {
            'type': defect_name,
            'bbox': [x, y, w, h],
            'center': (center_x, center_y),
            'area': area,
            'contrast': contrast,
            'color': self.get_color_name(defect_name)
        }
    
    def get_color_name(self, defect_name):
        """получение имени цвета"""
        if 'красный' in defect_name:
            return 'красный'
        elif 'синий' in defect_name:
            return 'синий'
        elif 'зеленый' in defect_name:
            return 'зеленый'
        elif 'желтый' in defect_name:
            return 'желтый'
        return 'unknown'
    
    def draw_defects(self, frame, defects):
        """отрисовка дефектов"""
        for defect in defects:
            x, y, w, h = defect['bbox']
            color_name = defect['color']
            
            if color_name in self.display_colors:
                color = self.display_colors[color_name]
            else:
                color = (255, 255, 255)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            center_x, center_y = defect['center']
            cv2.circle(frame, (center_x, center_y), 5, color, -1)
            
            text = f"{defect['type'].split('_')[0]}"
            cv2.putText(frame, text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            area_text = f"{defect['area']:.0f}"
            cv2.putText(frame, area_text, (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame
    
    def reset_statistics(self):
        """сброс статистики"""
        self.defect_count = 0
        self.red_count = 0
        self.blue_count = 0
        self.green_count = 0
        self.yellow_count = 0
    
    def get_statistics(self):
        """получение статистики"""
        return {
            'total': self.defect_count,
            'red': self.red_count,
            'blue': self.blue_count,
            'green': self.green_count,
            'yellow': self.yellow_count
        }

class CameraThread(QThread):
    """поток для захвата видео с камеры"""
    frame_signal = pyqtSignal(int, np.ndarray)
    
    def __init__(self, camera_id=0):
        super().__init__()
        self.camera_id = camera_id
        self.running = False
        self.cap = None
        
    def run(self):
        self.running = True
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        while self.running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.frame_signal.emit(self.camera_id, frame)
                else:
                    self.msleep(10)
            else:
                self.msleep(1000)
                self.cap = cv2.VideoCapture(self.camera_id)
    
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.wait()

class VideoProcessingThread(QThread):
    """поток для обработки видео файла"""
    frame_signal = pyqtSignal(np.ndarray, int, int)  # frame, current_frame, total_frames
    finished_signal = pyqtSignal()
    
    def __init__(self, video_handler, defect_detector, process_defects=False):
        super().__init__()
        self.video_handler = video_handler
        self.defect_detector = defect_detector
        self.process_defects = process_defects
        self.running = False
        self.paused = False
        
    def run(self):
        self.running = True
        self.paused = False
        
        while self.running and self.video_handler.video_capture:
            if self.paused:
                self.msleep(100)
                continue
            
            frame = self.video_handler.get_next_frame()
            if frame is None:
                break
            
            # обработка дефектов если включено
            if self.process_defects:
                defects = self.defect_detector.detect_color_defects(frame)
                frame = self.defect_detector.draw_defects(frame, defects)
            
            current_frame = self.video_handler.get_frame_position()
            total_frames = self.video_handler.total_frames
            
            self.frame_signal.emit(frame, current_frame, total_frames)
            
            # регулируем скорость воспроизведения
            if self.video_handler.video_fps > 0:
                delay = int(1000 / self.video_handler.video_fps)
                self.msleep(delay)
        
        self.finished_signal.emit()
    
    def pause(self):
        self.paused = True
    
    def resume(self):
        self.paused = False
    
    def stop(self):
        self.running = False
        self.wait()

class Arm165RealApp(QMainWindow):
    """главное окно приложения с поддержкой видео файлов"""
    
    def __init__(self):
        super().__init__()
        
        # загрузка интерфейса
        uic.loadUi("arm165_interface.ui", self)
        self.setWindowTitle("ARM165 - система анализа видео с детекцией дефектов")
        
        # инициализация компонентов
        self.defect_detector = ColorDefectDetector()
        self.video_handler = VideoFileHandler()
        
        # обработчики видео
        self.video_processing_thread = None
        self.camera_threads = {}
        
        # состояния
        self.is_video_playing = False
        self.is_detecting = False
        self.show_defects = True
        self.save_results = False
        self.save_path = "video_results"
        self.log_file_path = "system_logs.txt"
        self.conveyor_enabled = False  # состояние конвейера
        
        # создаем папку для сохранения
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        # статистика
        self.video_stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'defects_found': 0,
            'start_time': None
        }
        
        # настройка интерфейса
        self.setup_interface()
        self.connect_signals()
        
        # таймеры
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_statistics_display)
        self.stats_timer.start(1000)
        
        # таймер для имитации работы конвейера
        self.conveyor_timer = QTimer()
        self.conveyor_timer.timeout.connect(self.update_conveyor_simulation)
        
        # инициализация файла логов
        self.init_log_file()
        
        # логирование
        self.write_log_message("система запущена")
        self.write_log_message("модуль видео анализа активирован")
    
    def init_log_file(self):
        """инициализация файла логов"""
        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write("\n" + "="*60 + "\n")
                f.write(f"СЕССИЯ НАЧАТА: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*60 + "\n")
            self.write_to_log_file("Файл логов инициализирован")
        except Exception as e:
            print(f"Ошибка создания файла логов: {e}")
    
    def write_to_log_file(self, message):
        """запись в текстовый файл логов"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] {message}\n"
            
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Ошибка записи в файл логов: {e}")
    
    def save_logs_to_file(self):
        """сохранение логов из интерфейса в отдельный файл"""
        try:
            if hasattr(self, 'text_log'):
                logs = self.text_log.toPlainText()
                if logs.strip():
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"logs_export_{timestamp}.txt"
                    
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write("ЭКСПОРТ ЛОГОВ ИЗ ИНТЕРФЕЙСА\n")
                        f.write(f"Дата экспорта: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write("="*60 + "\n\n")
                        f.write(logs)
                    
                    self.write_log_message(f"логи экспортированы в файл: {filename}")
                    QMessageBox.information(self, "Успех", f"Логи сохранены в файл:\n{filename}")
                else:
                    QMessageBox.warning(self, "Внимание", "Нет данных для сохранения")
        except Exception as e:
            self.write_log_message(f"ошибка экспорта логов: {str(e)}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить логи:\n{str(e)}")
    
    def save_emergency_logs(self):
        """сохранение экстренных логов"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"emergency_logs_{timestamp}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("ЭКСТРЕННЫЕ ЛОГИ СИСТЕМЫ\n")
                f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*60 + "\n\n")
                
                # системная информация
                f.write("СИСТЕМНАЯ ИНФОРМАЦИЯ:\n")
                f.write(f"  Состояние видео: {'воспроизводится' if self.is_video_playing else 'остановлено'}\n")
                f.write(f"  Детекция дефектов: {'включена' if self.is_detecting else 'выключена'}\n")
                f.write(f"  Конвейер: {'включен' if self.conveyor_enabled else 'выключен'}\n")
                f.write(f"  Активные камеры: {len(self.camera_threads)}\n\n")
                
                # статистика дефектов
                stats = self.defect_detector.get_statistics()
                f.write("СТАТИСТИКА ДЕФЕКТОВ:\n")
                f.write(f"  Всего дефектов: {stats['total']}\n")
                f.write(f"  Красных: {stats['red']}\n")
                f.write(f"  Синих: {stats['blue']}\n")
                f.write(f"  Зеленых: {stats['green']}\n")
                f.write(f"  Желтых: {stats['yellow']}\n\n")
                
                # информация о видео
                if self.video_handler.current_video_path:
                    video_info = self.video_handler.get_video_info()
                    f.write("ИНФОРМАЦИЯ О ВИДЕО:\n")
                    f.write(f"  Файл: {video_info.get('path', 'неизвестно')}\n")
                    f.write(f"  Размер: {video_info.get('width', 0)}x{video_info.get('height', 0)}\n")
                    f.write(f"  Кадров: {video_info.get('total_frames', 0)}\n")
                    f.write(f"  Текущий кадр: {video_info.get('current_frame', 0)}\n\n")
                
                # последние логи из интерфейса
                if hasattr(self, 'text_log'):
                    logs = self.text_log.toPlainText()
                    if logs.strip():
                        # берем последние 50 строк
                        lines = logs.split('\n')
                        last_lines = lines[-50:] if len(lines) > 50 else lines
                        f.write("ПОСЛЕДНИЕ СООБЩЕНИЯ ИЗ ЛОГОВ:\n")
                        f.write("-"*40 + "\n")
                        for line in last_lines:
                            f.write(line + "\n")
            
            self.write_log_message(f"экстренные логи сохранены: {filename}")
            QMessageBox.information(self, "Успех", f"Экстренные логи сохранены в файл:\n{filename}")
            
        except Exception as e:
            self.write_log_message(f"ошибка сохранения экстренных логов: {str(e)}")
    
    def setup_interface(self):
        """настройка интерфейса"""
        # создаем вкладки для видео если их нет
        if not hasattr(self, 'tabWidget'):
            self.create_video_tabs()
        
        # настраиваем метки
        self.setup_labels()
        
        # добавляем кнопки управления видео
        self.add_video_controls()
        
        # настраиваем кнопки для логирования
        self.setup_log_buttons()
        
        # создаем кнопку экстренной остановки конвейера
        self.create_emergency_stop_button()
        
        # обновляем статус бар
        self.statusBar().showMessage("готов к работе - загрузите видео файл")
    
    def create_emergency_stop_button(self):
        """создание кнопки экстренной остановки конвейера"""
        # Ищем существующую кнопку "Stop" для конвейера
        if hasattr(self, 'konveer_stop'):
            # Переименовываем и перекрашиваем существующую кнопку
            self.konveer_stop.setText("ЭКСТРЕННАЯ ОСТАНОВКА")
            self.konveer_stop.setStyleSheet("""
                QPushButton {
                    background-color: #ff4444;
                    color: white;
                    font-weight: bold;
                    border: 2px solid #cc0000;
                    border-radius: 5px;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #ff6666;
                    border: 2px solid #ff0000;
                }
                QPushButton:pressed {
                    background-color: #cc0000;
                    border: 2px solid #990000;
                }
            """)
            
            # Изменяем размер и позицию кнопки
            self.konveer_stop.setGeometry(520, 270, 151, 25)  # Увеличиваем ширину и сдвигаем вправо
            
            # Создаем отдельную кнопку для обычной остановки
            self.normal_stop_button = QPushButton("Стоп", self)
            self.normal_stop_button.setGeometry(440, 270, 71, 25)
            self.normal_stop_button.setStyleSheet("""
                QPushButton {
                    background-color: #ff9966;
                    color: white;
                    border: 1px solid #cc6633;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #ffaa77;
                }
            """)
            self.normal_stop_button.clicked.connect(self.stop_conveyor_normal)
            
            # Добавляем индикатор состояния конвейера
            self.conveyor_indicator = QLabel(self)
            self.conveyor_indicator.setGeometry(340, 300, 20, 20)
            self.conveyor_indicator.setStyleSheet("background-color: #ff4444; border-radius: 10px;")
            
            # Добавляем метку состояния
            self.conveyor_status_label = QLabel("Конвейер: ВЫКЛ", self)
            self.conveyor_status_label.setGeometry(370, 300, 101, 17)
        
        # Если кнопки нет в UI, создаем новую
        elif hasattr(self, 'konveer'):
            # Создаем новую кнопку экстренной остановки
            self.emergency_stop_button = QPushButton("ЭКСТРЕННАЯ ОСТАНОВКА", self)
            self.emergency_stop_button.setGeometry(520, 270, 151, 25)
            self.emergency_stop_button.setStyleSheet("""
                QPushButton {
                    background-color: #ff4444;
                    color: white;
                    font-weight: bold;
                    border: 2px solid #cc0000;
                    border-radius: 5px;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #ff6666;
                    border: 2px solid #ff0000;
                }
                QPushButton:pressed {
                    background-color: #cc0000;
                    border: 2px solid #990000;
                }
            """)
            self.emergency_stop_button.clicked.connect(self.emergency_stop_conveyor)
    
    def setup_log_buttons(self):
        """настройка кнопок логирования"""
        # кнопка сохранения логов
        if hasattr(self, 'btn_save_log'):
            self.btn_save_log.clicked.connect(self.save_logs_to_file)
        
        # кнопка сохранения экстренных логов
        if hasattr(self, 'btn_save_emergency_log'):
            self.btn_save_emergency_log.clicked.connect(self.save_emergency_logs)
        
        # кнопка экстренной остановки конвейера
        if hasattr(self, 'konveer_stop'):
            self.konveer_stop.clicked.connect(self.emergency_stop_conveyor)
    
    def create_video_tabs(self):
        """создание вкладок для видео"""
        # ищем контейнер для вкладок или создаем новый
        if hasattr(self, 'text_session'):
            # используем существующее поле как контейнер
            container = self.text_session.parent()
            
            # создаем QTabWidget
            self.video_tab_widget = QTabWidget(container)
            self.video_tab_widget.setGeometry(590, 910, 361, 191)
            
            # вкладка 1: исходное видео
            self.original_video_tab = QWidget()
            original_layout = QVBoxLayout()
            self.original_video_label = QLabel("исходное видео")
            self.original_video_label.setAlignment(Qt.AlignCenter)
            self.original_video_label.setStyleSheet("border: 1px solid gray; background-color: black;")
            original_layout.addWidget(self.original_video_label)
            self.original_video_tab.setLayout(original_layout)
            
            # вкладка 2: обработанное видео
            self.processed_video_tab = QWidget()
            processed_layout = QVBoxLayout()
            self.processed_video_label = QLabel("обработанное видео")
            self.processed_video_label.setAlignment(Qt.AlignCenter)
            self.processed_video_label.setStyleSheet("border: 1px solid gray; background-color: black;")
            processed_layout.addWidget(self.processed_video_label)
            self.processed_video_tab.setLayout(processed_layout)
            
            # добавляем вкладки
            self.video_tab_widget.addTab(self.original_video_tab, "исходное")
            self.video_tab_widget.addTab(self.processed_video_tab, "обработанное")
    
    def setup_labels(self):
        """настройка метки"""
        if hasattr(self, 'label_color_red'):
            self.label_color_red.setStyleSheet("color: red; font-weight: bold;")
            self.label_color_red.setText("красный: 0")
        
        if hasattr(self, 'label_color_yellow'):
            self.label_color_yellow.setStyleSheet("color: #CCCC00; font-weight: bold;")
            self.label_color_yellow.setText("желтый: 0")
        
        if hasattr(self, 'label_color_green'):
            self.label_color_green.setStyleSheet("color: green; font-weight: bold;")
            self.label_color_green.setText("зеленый: 0")
        
        if hasattr(self, 'label_color_blue'):
            self.label_color_blue.setStyleSheet("color: blue; font-weight: bold;")
            self.label_color_blue.setText("синий: 0")
    
    def add_video_controls(self):
        """добавление элементов управления видео"""
        # добавляем поле для отображения пути к файлу
        if hasattr(self, 'text_log_8'):
            self.text_log_8.setPlainText("путь к файлу не выбран")
            self.text_log_8.setReadOnly(True)
        
        # добавляем кнопку выбора файла
        if hasattr(self, 'btn_3motora'):
            self.btn_3motora.setText("выбрать файл")
        
        # добавляем кнопку открытия папки
        if hasattr(self, 'btn_6motorov'):
            self.btn_6motorov.setText("открыть папку")
        
        # добавляем метку прогресса
        if hasattr(self, 'text_log_9'):
            self.text_log_9.setPlainText("прогресс: 0%")
            self.text_log_9.setReadOnly(True)
        
        # добавляем информацию о видео
        if hasattr(self, 'text_log_10'):
            self.text_log_10.setPlainText("информация о видео")
            self.text_log_10.setReadOnly(True)
    
    def connect_signals(self):
        """подключение сигналов"""
        # основные кнопки управления
        if hasattr(self, 'btn_on'):
            self.btn_on.clicked.connect(self.on_button_on)
        if hasattr(self, 'btn_pause'):
            self.btn_pause.clicked.connect(self.on_video_pause)
        if hasattr(self, 'btn_stop'):
            self.btn_stop.clicked.connect(self.on_video_stop)
        
        # кнопки работы с видео файлами
        if hasattr(self, 'btn_3motora'):
            self.btn_3motora.clicked.connect(self.select_video_file)
        
        if hasattr(self, 'btn_6motorov'):
            self.btn_6motorov.clicked.connect(self.open_results_folder)
        
        if hasattr(self, 'btn_manual_joint_2'):
            self.btn_manual_joint_2.clicked.connect(self.load_selected_video)
        
        if hasattr(self, 'btn_manual_joint_3'):
            self.btn_manual_joint_3.clicked.connect(self.play_video)
        
        if hasattr(self, 'btn_manual_joint_4'):
            self.btn_manual_joint_4.clicked.connect(self.stop_video)
        
        # чекбоксы
        if hasattr(self, 'check_cycle_2'):
            self.check_cycle_2.setText("детектировать дефекты")
            self.check_cycle_2.stateChanged.connect(self.toggle_defect_detection)
        
        if hasattr(self, 'check_cycle_3'):
            self.check_cycle_3.setText("сохранять результаты")
            self.check_cycle_3.stateChanged.connect(self.toggle_save_results)
        
        if hasattr(self, 'check_cycle_4'):
            self.check_cycle_4.setText("показывать дефекты")
            self.check_cycle_4.setChecked(True)
            self.check_cycle_4.stateChanged.connect(self.toggle_show_defects)
        
        if hasattr(self, 'check_cycle_5'):
            self.check_cycle_5.setText("использовать камеру")
            self.check_cycle_5.stateChanged.connect(self.toggle_camera_mode)
        
        # кнопки камер
        if hasattr(self, 'btn_add_point'):
            self.btn_add_point.setText("камера 1")
            self.btn_add_point.clicked.connect(lambda: self.toggle_camera(0))
        
        if hasattr(self, 'btn_clear_list'):
            self.btn_clear_list.setText("камера 2")
            self.btn_clear_list.clicked.connect(lambda: self.toggle_camera(1))
        
        if hasattr(self, 'btn_play'):
            self.btn_play.setText("камера 3")
            self.btn_play.clicked.connect(lambda: self.toggle_camera(2))
        
        # кнопки управления камерами
        if hasattr(self, 'btn_apply_program'):
            self.btn_apply_program.setText("все камеры")
            self.btn_apply_program.clicked.connect(self.toggle_all_cameras)
        
        # кнопки управления конвейером
        if hasattr(self, 'check_konveer'):
            self.check_konveer.stateChanged.connect(self.toggle_conveyor)
        
        # слайдер скорости конвейера
        if hasattr(self, 'slider_skorost'):
            self.slider_skorost.valueChanged.connect(self.update_conveyor_speed)
    
    def toggle_conveyor(self, state):
        """включение/выключение конвейера"""
        self.conveyor_enabled = (state == Qt.Checked)
        
        if self.conveyor_enabled:
            speed = self.slider_skorost.value() if hasattr(self, 'slider_skorost') else 50
            self.write_log_message(f"конвейер включен, скорость: {speed}%")
            self.write_to_log_file(f"Конвейер включен со скоростью {speed}%")
            
            # Запускаем таймер для имитации работы конвейера
            self.conveyor_timer.start(1000)  # Обновление каждую секунду
            
            # Обновляем индикатор и метку
            self.update_conveyor_status(True, speed)
        else:
            self.write_log_message("конвейер выключен")
            self.write_to_log_file("Конвейер выключен")
            
            # Останавливаем таймер
            self.conveyor_timer.stop()
            
            self.update_conveyor_status(False, 0)
    
    def update_conveyor_speed(self, value):
        """обновление скорости конвейера"""
        if self.conveyor_enabled:
            self.write_log_message(f"скорость конвейера изменена: {value}%")
            self.write_to_log_file(f"Скорость конвейера изменена на {value}%")
            
            # В реальном приложении здесь будет код изменения скорости конвейера
            
            # Обновляем отображение скорости
            if hasattr(self, 'text_log_9'):
                self.text_log_9.setPlainText(f"скорость конвейера: {value}%")
            
            # Обновляем индикатор
            self.update_conveyor_status(True, value)
    
    def emergency_stop_conveyor(self):
        """экстренная остановка конвейера (красная кнопка)"""
        if self.conveyor_enabled:
            # Визуальный эффект экстренной остановки
            self.blink_emergency_stop()
            
            self.conveyor_enabled = False
            if hasattr(self, 'check_konveer'):
                self.check_konveer.setChecked(False)
            
            self.write_log_message("КОНВЕЙЕР ЭКСТРЕННО ОСТАНОВЛЕН!")
            self.write_to_log_file("КОНВЕЙЕР ЭКСТРЕННО ОСТАНОВЛЕН - КРАСНАЯ КНОПКА!")
            
            # Останавливаем таймер
            self.conveyor_timer.stop()
            
            # В реальном приложении здесь будет код экстренной остановки конвейера
            # Например, отключение питания, активация тормозов и т.д.
            
            self.update_conveyor_status(False, 0)
            
            # Показываем сообщение об экстренной остановке
            QMessageBox.critical(self, "ЭКСТРЕННАЯ ОСТАНОВКА", 
                               "Конвейер экстренно остановлен!\n"
                               "Проверьте систему на наличие неисправностей.")
            
            # Сохраняем экстренные логи
            self.save_emergency_logs()
        else:
            self.write_log_message("попытка экстренной остановки выключенного конвейера")
    
    def stop_conveyor_normal(self):
        """нормальная остановка конвейера (оранжевая кнопка)"""
        if self.conveyor_enabled:
            self.conveyor_enabled = False
            if hasattr(self, 'check_konveer'):
                self.check_konveer.setChecked(False)
            
            self.write_log_message("конвейер остановлен нормально")
            self.write_to_log_file("Конвейер остановлен нормально")
            
            # Останавливаем таймер
            self.conveyor_timer.stop()
            
            self.update_conveyor_status(False, 0)
            
            QMessageBox.information(self, "Конвейер", "Конвейер остановлен")
        else:
            self.write_log_message("конвейер уже выключен")
    
    def blink_emergency_stop(self):
        """мигание при экстренной остановке"""
        original_style = self.konveer_stop.styleSheet()
        for _ in range(3):
            self.konveer_stop.setStyleSheet("background-color: white; color: red; border: 3px solid red;")
            QApplication.processEvents()
            QTimer.singleShot(200, lambda: self.konveer_stop.setStyleSheet(original_style))
            QApplication.processEvents()
            QTimer.singleShot(200, lambda: self.konveer_stop.setStyleSheet("background-color: white; color: red; border: 3px solid red;"))
            QApplication.processEvents()
            QTimer.singleShot(200, lambda: self.konveer_stop.setStyleSheet(original_style))
    
    def update_conveyor_simulation(self):
        """имитация работы конвейера"""
        if self.conveyor_enabled:
            # В реальном приложении здесь будет обмен данными с контроллером конвейера
            # Сейчас просто обновляем статус
            speed = self.slider_skorost.value() if hasattr(self, 'slider_skorost') else 50
            
            # Имитация обнаружения объектов на конвейере (раз в 5 секунд)
            if random.random() < 0.2:  # 20% chance каждую секунду
                self.write_log_message(f"объект обнаружен на конвейере (скорость: {speed}%)")
    
    def update_conveyor_status(self, is_running, speed=0):
        """обновление статуса конвейера"""
        # Обновляем индикатор
        if hasattr(self, 'conveyor_indicator'):
            if is_running:
                # Зеленый для работы, интенсивность зависит от скорости
                intensity = int(255 * (speed / 100))
                self.conveyor_indicator.setStyleSheet(f"background-color: rgb(0, {intensity}, 0); border-radius: 10px;")
            else:
                self.conveyor_indicator.setStyleSheet("background-color: #ff4444; border-radius: 10px;")
        
        # Обновляем метку статуса
        if hasattr(self, 'conveyor_status_label'):
            if is_running:
                self.conveyor_status_label.setText(f"Конвейер: ВКЛ ({speed}%)")
                self.conveyor_status_label.setStyleSheet("color: green; font-weight: bold;")
            else:
                self.conveyor_status_label.setText("Конвейер: ВЫКЛ")
                self.conveyor_status_label.setStyleSheet("color: red; font-weight: bold;")
    
    def select_video_file(self):
        """выбор видео файла"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "выберите видео файл",
            "",
            "Видео файлы (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;Все файлы (*.*)"
        )
        
        if file_path:
            if hasattr(self, 'text_log_8'):
                self.text_log_8.setPlainText(file_path)
            
            self.write_log_message(f"выбран файл: {os.path.basename(file_path)}")
            self.write_to_log_file(f"Выбран видео файл: {file_path}")
            
            # автоматически загружаем видео
            self.load_video_file(file_path)
    
    def load_selected_video(self):
        """загрузка выбранного видео"""
        if hasattr(self, 'text_log_8'):
            file_path = self.text_log_8.toPlainText()
            if file_path and os.path.exists(file_path):
                self.load_video_file(file_path)
            else:
                QMessageBox.warning(self, "ошибка", "файл не найден или не выбран")
                self.select_video_file()
    
    def load_video_file(self, file_path):
        """загрузка видео файла"""
        if self.is_video_playing:
            self.stop_video()
        
        if self.video_handler.load_video(file_path):
            video_info = self.video_handler.get_video_info()
            
            # обновляем информацию о видео
            info_text = f"файл: {os.path.basename(file_path)}\n"
            info_text += f"размер: {video_info['width']}x{video_info['height']}\n"
            info_text += f"кадров: {video_info['total_frames']}\n"
            info_text += f"fps: {video_info['fps']}\n"
            info_text += f"длительность: {video_info['duration']:.1f} сек"
            
            if hasattr(self, 'text_log_10'):
                self.text_log_10.setPlainText(info_text)
            
            self.write_log_message(f"видео загружено: {os.path.basename(file_path)}")
            self.write_to_log_file(f"Видео загружено: {file_path}")
            
            # показываем первый кадр
            self.show_first_frame()
        else:
            QMessageBox.critical(self, "ошибка", "не удалось загрузить видео файл")
            self.write_log_message(f"ошибка загрузки видео: {file_path}")
            self.write_to_log_file(f"ОШИБКА загрузки видео: {file_path}")
    
    def show_first_frame(self):
        """показать первый кадр видео"""
        if self.video_handler.video_capture:
            # переходим к первому кадру
            self.video_handler.seek_frame(0)
            frame = self.video_handler.get_next_frame()
            
            if frame is not None:
                # показываем в исходной вкладке
                self.display_video_frame(frame, is_original=True)
    
    def play_video(self):
        """воспроизведение видео"""
        if not self.video_handler.video_capture:
            QMessageBox.warning(self, "ошибка", "сначала загрузите видео файл")
            return
        
        if self.is_video_playing:
            if self.video_processing_thread:
                self.video_processing_thread.resume()
            return
        
        # сбрасываем статистику
        self.defect_detector.reset_statistics()
        self.video_stats = {
            'total_frames': self.video_handler.total_frames,
            'processed_frames': 0,
            'defects_found': 0,
            'start_time': datetime.now()
        }
        
        # переходим к началу
        self.video_handler.seek_frame(0)
        
        # создаем поток обработки
        process_defects = hasattr(self, 'check_cycle_2') and self.check_cycle_2.isChecked()
        self.video_processing_thread = VideoProcessingThread(
            self.video_handler,
            self.defect_detector,
            process_defects
        )
        
        # подключаем сигналы
        self.video_processing_thread.frame_signal.connect(self.process_video_frame)
        self.video_processing_thread.finished_signal.connect(self.video_finished)
        
        # запускаем поток
        self.video_processing_thread.start()
        self.is_video_playing = True
        
        # обновляем кнопки
        if hasattr(self, 'btn_manual_joint_3'):
            self.btn_manual_joint_3.setText("пауза")
            self.btn_manual_joint_3.setStyleSheet("background-color: orange; color: white;")
        
        self.write_log_message("воспроизведение видео начато")
        self.write_to_log_file("Начато воспроизведение видео")
    
    def on_video_pause(self):
        """пауза видео"""
        if self.is_video_playing and self.video_processing_thread:
            self.video_processing_thread.pause()
            
            if hasattr(self, 'btn_manual_joint_3'):
                self.btn_manual_joint_3.setText("продолжить")
            
            self.write_log_message("видео приостановлено")
            self.write_to_log_file("Видео приостановлено")
    
    def stop_video(self):
        """остановка видео"""
        if self.video_processing_thread:
            self.video_processing_thread.stop()
            self.video_processing_thread = None
        
        self.is_video_playing = False
        
        # обновляем кнопки
        if hasattr(self, 'btn_manual_joint_3'):
            self.btn_manual_joint_3.setText("воспроизвести")
            self.btn_manual_joint_3.setStyleSheet("")
        
        self.write_log_message("воспроизведение видео остановлено")
        self.write_to_log_file("Воспроизведение видео остановлено")
    
    def video_finished(self):
        """обработка завершения видео"""
        self.is_video_playing = False
        
        if hasattr(self, 'btn_manual_joint_3'):
            self.btn_manual_joint_3.setText("воспроизвести")
            self.btn_manual_joint_3.setStyleSheet("")
        
        # показываем статистику
        stats = self.defect_detector.get_statistics()
        elapsed = datetime.now() - self.video_stats['start_time']
        
        message = f"обработка завершена\n"
        message += f"время: {elapsed.total_seconds():.1f} сек\n"
        message += f"кадров: {self.video_stats['processed_frames']}\n"
        message += f"дефектов: {stats['total']}\n"
        message += f"  красных: {stats['red']}\n"
        message += f"  синих: {stats['blue']}\n"
        message += f"  зеленых: {stats['green']}\n"
        message += f"  желтых: {stats['yellow']}"
        
        QMessageBox.information(self, "завершено", message)
        self.write_log_message("обработка видео завершена")
        self.write_to_log_file(f"Обработка видео завершена. Дефектов: {stats['total']}")
    
    def process_video_frame(self, frame, current_frame, total_frames):
        """обработка кадра видео"""
        self.video_stats['processed_frames'] = current_frame
        
        # обновляем прогресс
        progress = (current_frame / total_frames) * 100 if total_frames > 0 else 0
        
        if hasattr(self, 'slider_skorost'):
            self.slider_skorost.blockSignals(True)
            self.slider_skorost.setValue(int(progress))
            self.slider_skorost.blockSignals(False)
        
        if hasattr(self, 'text_log_9'):
            self.text_log_9.setPlainText(f"прогресс: {progress:.1f}%")
        
        # показываем исходное видео
        self.display_video_frame(frame, is_original=True)
        
        # если детектирование включено, показываем обработанное видео
        if hasattr(self, 'check_cycle_2') and self.check_cycle_2.isChecked():
            defects = self.defect_detector.detect_color_defects(frame)
            processed_frame = frame.copy()
            
            if hasattr(self, 'check_cycle_4') and self.check_cycle_4.isChecked():
                processed_frame = self.defect_detector.draw_defects(processed_frame, defects)
            
            self.display_video_frame(processed_frame, is_original=False)
            
            # сохраняем кадры если нужно
            if hasattr(self, 'check_cycle_3') and self.check_cycle_3.isChecked() and defects:
                self.save_video_frame(processed_frame, current_frame, defects)
    
    def display_video_frame(self, frame, is_original=True):
        """отображение кадра видео"""
        # конвертируем в QImage
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # определяем куда показывать
        if is_original:
            if hasattr(self, 'original_video_label'):
                label = self.original_video_label
            else:
                # показываем в камере 1
                label = self.get_camera_label(0)
        else:
            if hasattr(self, 'processed_video_label'):
                label = self.processed_video_label
            else:
                # показываем в камере 2
                label = self.get_camera_label(1)
        
        if label:
            # масштабируем
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            label.setPixmap(scaled_pixmap)
    
    def get_camera_label(self, camera_id):
        """получение метки для камеры"""
        if camera_id == 0 and hasattr(self, 'camera1'):
            container = self.camera1
        elif camera_id == 1 and hasattr(self, 'camera2'):
            container = self.camera2
        elif camera_id == 2 and hasattr(self, 'camera3'):
            container = self.camera3
        else:
            return None
        
        # создаем QLabel если его нет
        if container.layout() is None:
            layout = QVBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            label = QLabel(container)
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)
            return label
        else:
            return container.layout().itemAt(0).widget()
    
    def save_video_frame(self, frame, frame_number, defects):
        """сохранение кадра видео"""
        try:
            video_name = os.path.splitext(os.path.basename(self.video_handler.current_video_path))[0]
            filename = f"{video_name}_frame{frame_number:06d}.jpg"
            filepath = os.path.join(self.save_path, filename)
            
            cv2.imwrite(filepath, frame)
            
            # логируем сохранение
            if frame_number % 100 == 0:  # логируем каждые 100 кадров
                self.write_log_message(f"сохранен кадр {frame_number} с {len(defects)} дефектами")
                self.write_to_log_file(f"Сохранен кадр {frame_number} с {len(defects)} дефектами")
            
        except Exception as e:
            self.write_log_message(f"ошибка сохранения кадра {frame_number}: {str(e)}")
            self.write_to_log_file(f"Ошибка сохранения кадра {frame_number}: {str(e)}")
    
    def open_results_folder(self):
        """открытие папки с результатами"""
        if os.path.exists(self.save_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.save_path))
            self.write_log_message(f"открыта папка: {self.save_path}")
            self.write_to_log_file(f"Открыта папка результатов: {self.save_path}")
        else:
            QMessageBox.information(self, "информация", "папка результатов еще не создана")
    
    def toggle_defect_detection(self, state):
        """включение/выключение детекции дефектов"""
        self.is_detecting = (state == Qt.Checked)
        status = "включено" if self.is_detecting else "выключено"
        self.write_log_message(f"детекция дефектов: {status}")
        self.write_to_log_file(f"Детекция дефектов: {status}")
    
    def toggle_save_results(self, state):
        """включение/выключение сохранения результатов"""
        self.save_results = (state == Qt.Checked)
        status = "включено" if self.save_results else "выключено"
        self.write_log_message(f"сохранение результатов: {status}")
        self.write_to_log_file(f"Сохранение результатов: {status}")
    
    def toggle_show_defects(self, state):
        """включение/выключение отображения дефектов"""
        self.show_defects = (state == Qt.Checked)
        status = "включено" if self.show_defects else "выключено"
        self.write_log_message(f"отображение дефектов: {status}")
        self.write_to_log_file(f"Отображение дефектов: {status}")
    
    def toggle_camera_mode(self, state):
        """переключение режима камеры"""
        use_camera = (state == Qt.Checked)
        if use_camera:
            self.start_all_cameras()
        else:
            self.stop_all_cameras()
    
    def toggle_camera(self, camera_id):
        """включение/выключение камеры"""
        if camera_id in self.camera_threads:
            self.stop_camera(camera_id)
        else:
            self.start_camera(camera_id)
    
    def start_camera(self, camera_id):
        """запуск камеры"""
        if camera_id in self.camera_threads:
            return
        
        thread = CameraThread(camera_id)
        thread.frame_signal.connect(self.process_camera_frame)
        thread.start()
        
        self.camera_threads[camera_id] = thread
        self.write_log_message(f"камера {camera_id} запущена")
        self.write_to_log_file(f"Камера {camera_id} запущена")
    
    def stop_camera(self, camera_id):
        """остановка камеры"""
        if camera_id in self.camera_threads:
            self.camera_threads[camera_id].stop()
            del self.camera_threads[camera_id]
            self.write_log_message(f"камера {camera_id} остановлена")
            self.write_to_log_file(f"Камера {camera_id} остановлена")
    
    def start_all_cameras(self):
        """запуск всех камер"""
        for i in range(3):
            self.start_camera(i)
        self.write_log_message("все камеры запущены")
        self.write_to_log_file("Все камеры запущены")
    
    def stop_all_cameras(self):
        """остановка всех камер"""
        for camera_id in list(self.camera_threads.keys()):
            self.stop_camera(camera_id)
        self.write_log_message("все камеры остановлены")
        self.write_to_log_file("Все камеры остановлены")
    
    def toggle_all_cameras(self):
        """включение/выключение всех камер"""
        if len(self.camera_threads) > 0:
            self.stop_all_cameras()
        else:
            self.start_all_cameras()
    
    def process_camera_frame(self, camera_id, frame):
        """обработка кадра с камеры"""
        # показываем в соответствующем виджете
        if camera_id == 0 and hasattr(self, 'camera1'):
            self.display_camera_frame(frame, self.camera1)
        elif camera_id == 1 and hasattr(self, 'camera2'):
            self.display_camera_frame(frame, self.camera2)
        elif camera_id == 2 and hasattr(self, 'camera3'):
            self.display_camera_frame(frame, self.camera3)
    
    def display_camera_frame(self, frame, container):
        """отображение кадра камеры"""
        # обрабатываем дефекты если включено
        if self.is_detecting:
            defects = self.defect_detector.detect_color_defects(frame)
            if self.show_defects:
                frame = self.defect_detector.draw_defects(frame, defects)
        
        # конвертируем в QImage
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        qt_image = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
        
        # создаем QLabel если его нет
        if container.layout() is None:
            layout = QVBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            label = QLabel(container)
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)
        else:
            label = container.layout().itemAt(0).widget()
        
        # масштабируем и отображаем
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            container.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        label.setPixmap(scaled_pixmap)
    
    def update_statistics_display(self):
        """обновление отображения статистики"""
        stats = self.defect_detector.get_statistics()
        
        # обновляем метки цветов
        if hasattr(self, 'label_color_red'):
            self.label_color_red.setText(f"красный: {stats['red']}")
        
        if hasattr(self, 'label_color_yellow'):
            self.label_color_yellow.setText(f"желтый: {stats['yellow']}")
        
        if hasattr(self, 'label_color_green'):
            self.label_color_green.setText(f"зеленый: {stats['green']}")
        
        if hasattr(self, 'label_color_blue'):
            self.label_color_blue.setText(f"синий: {stats['blue']}")
        
        # обновляем статус бар
        total = stats['total']
        cameras = len(self.camera_threads)
        video_status = "воспроизводится" if self.is_video_playing else "остановлено"
        conveyor_status = "вкл" if self.conveyor_enabled else "выкл"
        
        status_msg = f"дефектов: {total} | камер: {cameras} | видео: {video_status} | конвейер: {conveyor_status}"
        self.statusBar().showMessage(status_msg)
    
    def write_log_message(self, message):
        """логирование сообщений в интерфейс"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        if hasattr(self, 'text_log'):
            self.text_log.append(log_entry)
            scrollbar = self.text_log.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        else:
            print(log_entry)
    
    # оригинальные обработчики кнопок (упрощенные)
    
    def on_button_on(self):
        self.write_log_message("система включена")
        self.write_to_log_file("Система включена")
    
    def on_video_pause(self):
        if self.is_video_playing and self.video_processing_thread:
            self.video_processing_thread.pause()
            self.write_log_message("видео приостановлено")
            self.write_to_log_file("Видео приостановлено")
    
    def on_video_stop(self):
        self.stop_video()
    
    def closeEvent(self, event):
        """обработка закрытия окна"""
        self.write_log_message("завершение работы...")
        self.write_to_log_file("Завершение работы системы")
        
        # останавливаем видео
        self.stop_video()
        
        # останавливаем камеры
        self.stop_all_cameras()
        
        # останавливаем конвейер
        if self.conveyor_enabled:
            self.conveyor_enabled = False
            self.conveyor_timer.stop()
        
        # освобождаем ресурсы
        self.video_handler.release()
        
        # останавливаем таймеры
        if hasattr(self, 'stats_timer'):
            self.stats_timer.stop()
        
        # сохраняем статистику
        self.save_final_statistics()
        
        # добавляем завершение в файл логов
        self.write_to_log_file("СЕССИЯ ЗАВЕРШЕНА")
        
        self.write_log_message("система остановлена")
        event.accept()
    
    def save_final_statistics(self):
        """сохранение финальной статистики"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stats_file = os.path.join(self.save_path, f"final_stats_{timestamp}.txt")
            
            stats = self.defect_detector.get_statistics()
            video_info = self.video_handler.get_video_info() if self.video_handler.current_video_path else {}
            
            with open(stats_file, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("ФИНАЛЬНАЯ СТАТИСТИКА АНАЛИЗА ВИДЕО\n")
                f.write("="*60 + "\n\n")
                f.write(f"дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                if video_info:
                    f.write("ИНФОРМАЦИЯ О ВИДЕО:\n")
                    f.write(f"  файл: {video_info.get('path', 'неизвестно')}\n")
                    f.write(f"  размер: {video_info.get('width', 0)}x{video_info.get('height', 0)}\n")
                    f.write(f"  кадров: {video_info.get('total_frames', 0)}\n")
                    f.write(f"  fps: {video_info.get('fps', 0)}\n\n")
                
                f.write("СТАТИСТИКА ДЕФЕКТОВ:\n")
                f.write(f"  всего дефектов: {stats['total']}\n")
                f.write(f"  красных дефектов: {stats['red']}\n")
                f.write(f"  синих дефектов: {stats['blue']}\n")
                f.write(f"  зеленых дефектов: {stats['green']}\n")
                f.write(f"  желтых дефектов: {stats['yellow']}\n\n")
                
                f.write("СИСТЕМНАЯ ИНФОРМАЦИЯ:\n")
                f.write(f"  режим детекции: {'включен' if self.is_detecting else 'выключен'}\n")
                f.write(f"  сохранение результатов: {'включено' if self.save_results else 'выключено'}\n")
                f.write(f"  отображение дефектов: {'включено' if self.show_defects else 'выключено'}\n")
                f.write(f"  активных камер: {len(self.camera_threads)}\n")
                f.write(f"  состояние конвейера: {'включен' if self.conveyor_enabled else 'выключен'}\n")
            
            self.write_log_message(f"финальная статистика сохранена: {stats_file}")
            self.write_to_log_file(f"Финальная статистика сохранена: {stats_file}")
            
        except Exception as e:
            self.write_log_message(f"ошибка сохранения статистики: {str(e)}")
            self.write_to_log_file(f"Ошибка сохранения статистики: {str(e)}")

def main():
    """главная функция"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    font = QFont("Arial", 10)
    app.setFont(font)
    
    window = Arm165RealApp()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()