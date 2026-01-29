import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar
import threading
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtGui import QImage
from ultralytics import YOLO

class CameraManager(QObject):
    frame_ready = pyqtSignal(int, QImage, list, list, list)  # camera_id, frame, objects, qr_codes, defects
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.defect_detector = None
        
        # HSV диапазоны для цветов
        self.color_ranges_hsv = {
            'red': 
            [
                (np.array([0, 100, 100]), np.array([10, 255, 255])),
                (np.array([170, 100, 100]), np.array([180, 255, 255]))
            ],
            'green': [(np.array([40, 50, 50]), np.array([80, 255, 255]))],
            'blue': [(np.array([100, 50, 50]), np.array([130, 255, 255]))],
            'yellow': [(np.array([20, 100, 100]), np.array([30, 255, 255]))],
        }
        
        self.min_contour_area = 500
        
        # Загрузка модели дефектов
        self.load_defect_detector()
    
    def load_defect_detector(self):
        """Загрузка модели YOLO для детектирования дефектов"""
        try:
            model_path = "models/defect_detector.pt"
            if os.path.exists(model_path):
                self.defect_detector = YOLO(model_path)
                print("Модель дефектов загружена успешно")
            else:
                print("Модель дефектов не найдена. Используется предобученная YOLO.")
                self.defect_detector = YOLO('yolov8n.pt')
        except Exception as e:
            print(f"Ошибка загрузки модели дефектов: {e}")
            self.defect_detector = None
    
    def start_cameras(self):
        """Запуск камер"""
        for i in range(3):
            thread = threading.Thread(target=self.capture_and_process, args=(i,))
            thread.daemon = True
            thread.start()

    def capture_and_process(self, camera_id):
        """Захват и обработка видео с камеры"""
        cap = cv2.VideoCapture(camera_id)
        
        while self.running:
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Обработка кадра
                    processed_frame, objects, qr_codes, defects = self.process_frame_with_vision(frame.copy())
                    
                    # Конвертация в QImage
                    rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
                    
                    # Отправка обработанного кадра
                    self.frame_ready.emit(camera_id, qt_image, objects, qr_codes, defects)
            else:
                # Тестовое изображение
                self.create_test_image(camera_id)

    def process_frame_with_vision(self, frame):
        """Обработка кадра с детектированием объектов, QR-кодов и дефектов"""
        objects = []
        qr_codes = []
        defects = []
        
        # Фильтрация
        filtered = cv2.bilateralFilter(frame, 9, 75, 75)
        
        # Конвертация в HSV
        hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
        
        # Детектирование цветных объектов
        for color_name, ranges in self.color_ranges_hsv.items():
            if len(ranges) == 2 and color_name == 'red':
                # Красный цвет - два диапазона
                lower1, upper1 = ranges[0]
                lower2, upper2 = ranges[1]
                mask1 = cv2.inRange(hsv, lower1, upper1)
                mask2 = cv2.inRange(hsv, lower2, upper2)
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                # Остальные цвета - один диапазон
                lower, upper = ranges[0]
                mask = cv2.inRange(hsv, lower, upper)
            
            # Поиск контуров
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_contour_area:
                    continue
                
                # Определение формы
                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                shape = self._detect_shape(approx)
                
                # Нахождение центра
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                else:
                    x, y, w, h = cv2.boundingRect(contour)
                    cx, cy = x + w//2, y + h//2
                
                # Bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Сохранение объекта
                obj = {
                    'color': color_name,
                    'shape': shape,
                    'center': (cx, cy),
                    'size': (w, h),
                    'area': area
                }
                objects.append(obj)
                
                # Визуализация на кадре
                # Bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Центр
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.circle(frame, (cx, cy), 10, (0, 0, 255), 2)
                
                # Контур
                cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)
                
                # Текст
                info = f"{color_name} {shape}"
                cv2.putText(frame, info, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Детектирование QR-кодов
        qr_codes = self.detect_qr_codes(frame.copy())
        
        # Детектирование дефектов с помощью YOLO
        if self.defect_detector is not None:
            defects = self.detect_defects(frame.copy())
        
        return frame, objects, qr_codes, defects
    
    def detect_defects(self, frame):
        """Детектирование дефектов с помощью YOLO"""
        defects = []
        
        try:
            # Запуск инференса
            results = self.defect_detector(frame, conf=0.5, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Получение координат и класса
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # Получение имени класса
                        if hasattr(result, 'names'):
                            class_name = result.names[cls]
                        else:
                            class_name = f"defect_{cls}"
                        
                        # Проверка, является ли это дефектом
                        defect_classes = ["crack", "scratch", "dent", "corrosion", "missing_part"]
                        if class_name in defect_classes:
                            # Сохранение информации о дефекте
                            defect_info = {
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(conf),
                                'class': class_name,
                                'center': (int((x1+x2)/2), int((y1+y2)/2))
                            }
                            defects.append(defect_info)
                            
                            # Визуализация на кадре (красный для дефектов)
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                            label = f"{class_name}: {conf:.2f}"
                            cv2.putText(frame, label, (int(x1), int(y1)-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        except Exception as e:
            print(f"Ошибка детектирования дефектов: {e}")
        
        return defects
    
    def _detect_shape(self, approx):
        """Определение геометрической формы"""
        num_sides = len(approx)
        
        if num_sides == 3:
            return "triangle"
        elif num_sides == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            return "square" if 0.9 <= aspect_ratio <= 1.1 else "rectangle"
        elif num_sides > 4:
            area = cv2.contourArea(approx)
            perimeter = cv2.arcLength(approx, True)
            if perimeter == 0:
                return "polygon"
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            return "circle" if circularity > 0.7 else "polygon"
        else:
            return "unknown"
    
    def detect_qr_codes(self, frame):
        """Считывание QR-кодов"""
        decoded = pyzbar.decode(frame)
        qr_codes = []
        
        for obj in decoded:
            qr_data = obj.data.decode('utf-8')
            qr_codes.append(qr_data)
            
            # Рамка вокруг QR
            (x, y, w, h) = obj.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Текст с QR-кодом
            cv2.putText(frame, f"QR: {qr_data}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return qr_codes
    
    def create_test_image(self, camera_id):
        """Создание тестового изображения"""
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        img[:, :] = colors[camera_id % 3]

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"камера {camera_id + 1} (тест)"
        cv2.putText(img, text, (50, 120), font, 0.7, (255, 255, 255), 2)

        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        
        # Пустые данные
        self.frame_ready.emit(camera_id, qt_image, [], [], [])

    def stop_cameras(self):
        """Остановка камер"""
        self.running = False

# Функции конвертации между форматами
def qimage_to_cv(qimage):
    """Конвертация QImage в OpenCV формат"""
    try:
        qimage = qimage.convertToFormat(QImage.Format_RGB888)
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())
        arr = np.array(ptr).reshape(height, width, 3)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Ошибка конвертации QImage в OpenCV: {e}")
        return None

def cv_to_qimage(cv_img):
    """Конвертация OpenCV изображения в QImage"""
    try:
        if len(cv_img.shape) == 3:
            h, w, ch = cv_img.shape
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            return QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        else:
            return QImage()
    except Exception as e:
        print(f"Ошибка конвертации OpenCV в QImage: {e}")
        return QImage()