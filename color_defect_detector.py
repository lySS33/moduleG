
import cv2
import numpy as np

class ColorDefectDetector:
    """детектор цветных дефектов на коробках"""
    
    def __init__(self):
        # цвета дефектов в формате HSV
        self.defect_colors_hsv = {
            'красный_дефект': ([0, 100, 100], [10, 255, 255]),
            'красный_дефект2': ([170, 100, 100], [180, 255, 255]),  # второй диапазон для красного
            'синий_дефект': ([100, 100, 100], [130, 255, 255]),
            'зеленый_дефект': ([40, 100, 100], ([80, 255, 255])),
            'желтый_дефект': ([20, 100, 100], ([40, 255, 255])),
        }
        
        # параметры детекции
        self.min_contour_area = 50
        self.max_contour_area = 5000
        
        # цвет для отображения в BGR
        self.display_colors = {
            'красный_дефект': (0, 0, 255),
            'синий_дефект': (255, 0, 0),
            'зеленый_дефект': (0, 255, 0),
            'желтый_дефект': (0, 255, 255)
        }
    
    def detect_color_defects(self, frame):
        """обнаружение цветных дефектов"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detected_defects = []
        
        for defect_name, color_ranges in self.defect_colors_hsv.items():
            # обрабатываем диапазоны цветов
            if isinstance(color_ranges[0], list) and isinstance(color_ranges[1], list):
                # один диапазон
                masks = [cv2.inRange(hsv, 
                                   np.array(color_ranges[0], dtype=np.uint8),
                                   np.array(color_ranges[1], dtype=np.uint8))]
            else:
                # несколько диапазонов
                masks = []
                for i in range(0, len(color_ranges), 2):
                    lower = np.array(color_ranges[i], dtype=np.uint8)
                    upper = np.array(color_ranges[i+1], dtype=np.uint8)
                    masks.append(cv2.inRange(hsv, lower, upper))
            
            # объединяем маски если их несколько
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
            
            # обрабатываем найденные контуры
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if self.min_contour_area < area < self.max_contour_area:
                    # получаем ограничивающий прямоугольник
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # проверяем что это не фоновый объект
                    if self.is_valid_defect(frame, x, y, w, h):
                        # вычисляем характеристики дефекта
                        defect_info = self.analyze_defect(frame, contour, defect_name)
                        if defect_info:
                            detected_defects.append(defect_info)
        
        return detected_defects
    
    def enhance_mask(self, mask):
        """улучшение маски с помощью морфологических операций"""
        kernel = np.ones((5, 5), np.uint8)
        
        # закрытие для заполнения небольших отверстий
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # открытие для удаления небольших шумов
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def is_valid_defect(self, frame, x, y, w, h):
        """проверка что обнаруженный объект является валидным дефектом"""
        height, width = frame.shape[:2]
        
        # проверяем что дефект не на самом краю изображения
        margin = 10
        if (x < margin or y < margin or 
            x + w > width - margin or 
            y + h > height - margin):
            return False
        
        # проверяем соотношение сторон (дефекты обычно имеют неправильную форму)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio > 5 or aspect_ratio < 0.2:
            return False
        
        return True
    
    def analyze_defect(self, frame, contour, defect_name):
        """анализ характеристик дефекта"""
        # вычисляем ограничивающий прямоугольник
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # вычисляем момент для центра
        M = cv2.moments(contour)
        if M['m00'] != 0:
            center_x = int(M['m10'] / M['m00'])
            center_y = int(M['m01'] / M['m00'])
        else:
            center_x = x + w // 2
            center_y = y + h // 2
        
        # вычисляем контраст в области дефекта
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return None
        
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray_roi)
        
        # создаем информацию о дефекте
        defect_info = {
            'type': defect_name,
            'bbox': [x, y, w, h],
            'center': (center_x, center_y),
            'area': area,
            'contrast': contrast,
            'color': self.get_color_name(defect_name)
        }
        
        return defect_info
    
    def get_color_name(self, defect_name):
        """получение имени цвета"""
        if 'красный' in defect_name:
            return 'red'
        elif 'синий' in defect_name:
            return 'blue'
        elif 'зеленый' in defect_name:
            return 'green'
        elif 'желтый' in defect_name:
            return 'yellow'
        return 'unknown'
    
    def draw_defects(self, frame, defects):
        """отрисовка обнаруженных дефектов на кадре"""
        for defect in defects:
            x, y, w, h = defect['bbox']
            color_name = defect['color']
            
            # получаем цвет для отрисовки
            if color_name in self.display_colors:
                color = self.display_colors[color_name]
            else:
                color = (255, 255, 255)  # белый по умолчанию
            
            # рисуем прямоугольник
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # рисуем центр
            center_x, center_y = defect['center']
            cv2.circle(frame, (center_x, center_y), 5, color, -1)
            
            # добавляем текст
            text = f"{defect['type']}"
            cv2.putText(frame, text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # добавляем информацию о площади
            area_text = f"площадь: {defect['area']:.0f}"
            cv2.putText(frame, area_text, (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame