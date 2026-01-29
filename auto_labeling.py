import os
import cv2
import numpy as np
import shutil
from pathlib import Path
import yaml
#from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class AutoLabeler:
    """Автоматическая разметка изображений для дефектов"""
    
    def __init__(self, input_path="database", output_path="labeled_dataset"):
        self.input_path = input_path
        self.output_path = output_path
        self.classes = ["defect", "normal"]  # Основные классы
        
        # Параметры для автоматического обнаружения
        self.defect_params = {
            'contrast_threshold': 30,
            'edge_threshold': 50,
            'min_area': 100,
            'max_area': 10000
        }
        
        # Создаем структуру папок
        self.create_folders()
    
    def create_folders(self):
        """Создание структуры папок для датасета YOLO"""
        folders = [
            "images/train",
            "images/val",
            "labels/train",
            "labels/val"
        ]
        
        for folder in folders:
            os.makedirs(os.path.join(self.output_path, folder), exist_ok=True)
    
    def detect_potential_defects(self, image):
        """Автоматическое обнаружение потенциальных дефектов"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Улучшение контраста
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Обнаружение краев (дефекты обычно имеют четкие границы)
        edges = cv2.Canny(enhanced, 
                         self.defect_params['edge_threshold'],
                         self.defect_params['edge_threshold'] * 2)
        
        # Морфологические операции
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        # Поиск контуров
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Фильтрация по площади
            if area < self.defect_params['min_area'] or area > self.defect_params['max_area']:
                continue
            
            # Получение ограничивающего прямоугольника
            x, y, w, h = cv2.boundingRect(contour)
            
            # Проверка контраста внутри области
            roi = gray[y:y+h, x:x+w]
            if roi.size > 0:
                contrast = np.std(roi)
                if contrast > self.defect_params['contrast_threshold']:
                    bboxes.append([x, y, w, h])
        
        return bboxes
    
    def is_defect_image(self, image, bboxes):
        """Определение, содержит ли изображение дефекты"""
        if len(bboxes) > 0:
            return True
        
        # Дополнительные проверки
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Проверка на неравномерность текстуры
        texture_std = np.std(gray)
        if texture_std > 25:  # Высокий разброс значений может указывать на дефект
            return True
        
        return False
    
    def process_image(self, image_path, idx, is_training=True):
        """Обработка одного изображения"""
        # Чтение изображения
        image = cv2.imread(image_path)
        if image is None:
            print(f"Не удалось загрузить: {image_path}")
            return
        
        height, width = image.shape[:2]
        
        # Автоматическое обнаружение дефектов
        bboxes = self.detect_potential_defects(image)
        
        # Определение класса
        has_defect = self.is_defect_image(image, bboxes)
        class_id = 0 if has_defect else 1  # 0 - дефект, 1 - норма
        
        # Имя файла
        filename = f"image_{idx:04d}.jpg"
        
        # Определяем split (train/val)
        split = "train" if is_training else "val"
        
        # Сохраняем изображение
        output_img_path = os.path.join(self.output_path, "images", split, filename)
        cv2.imwrite(output_img_path, image)
        
        # Создаем файл аннотаций
        label_filename = f"image_{idx:04d}.txt"
        label_path = os.path.join(self.output_path, "labels", split, label_filename)
        
        with open(label_path, 'w') as f:
            if has_defect and bboxes:
                # Для изображений с дефектами используем автоматически найденные bboxes
                for bbox in bboxes:
                    x, y, w, h = bbox
                    
                    # Конвертируем в формат YOLO (нормализованные координаты)
                    x_center = (x + w / 2) / width
                    y_center = (y + h / 2) / height
                    w_norm = w / width
                    h_norm = h / height
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
            else:
                # Для нормальных изображений или если не найдены дефекты
                # В YOLO можно оставить пустой файл или использовать весь кадр как нормальный
                if not has_defect:
                    # Если это нормальное изображение, отмечаем весь кадр
                    f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
        
        return {
            'filename': filename,
            'class': 'defect' if has_defect else 'normal',
            'bboxes': len(bboxes),
            'width': width,
            'height': height
        }
    
    def process_dataset(self):
        """Обработка всего датасета"""
        print("Автоматическая разметка датасета...")
        
        # Получаем все изображения
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for file in os.listdir(self.input_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(self.input_path, file))
        
        print(f"Найдено {len(image_files)} изображений для обработки")
        
        # Разделяем на train/val (80/20)
        split_idx = int(len(image_files) * 0.8)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        print(f"Разделение: {len(train_files)} для обучения, {len(val_files)} для валидации")
        
        # Обработка обучающей выборки
        train_stats = []
        for idx, img_path in enumerate(train_files):
            if idx % 10 == 0:
                print(f"Обработка обучающих данных: {idx}/{len(train_files)}")
            
            stats = self.process_image(img_path, idx, is_training=True)
            if stats:
                train_stats.append(stats)
        
        # Обработка валидационной выборки
        val_stats = []
        for idx, img_path in enumerate(val_files):
            if idx % 10 == 0:
                print(f"Обработка валидационных данных: {idx}/{len(val_files)}")
            
            stats = self.process_image(img_path, idx + len(train_files), is_training=False)
            if stats:
                val_stats.append(stats)
        
        # Создание YAML конфигурации
        self.create_yaml_config()
        
        # Вывод статистики
        self.print_statistics(train_stats, val_stats)
        
        return train_stats, val_stats
    
    def create_yaml_config(self):
        """Создание YAML конфигурации для YOLO"""
        config = {
            'path': os.path.abspath(self.output_path),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.classes),
            'names': self.classes
        }
        
        yaml_path = os.path.join(self.output_path, "dataset.yaml")
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"YAML конфигурация создана: {yaml_path}")
        return yaml_path
    
    def print_statistics(self, train_stats, val_stats):
        """Вывод статистики разметки"""
        print("\n" + "="*50)
        print("СТАТИСТИКА АВТОМАТИЧЕСКОЙ РАЗМЕТКИ")
        print("="*50)
        
        # Статистика по классам
        train_defects = sum(1 for s in train_stats if s['class'] == 'defect')
        train_normal = sum(1 for s in train_stats if s['class'] == 'normal')
        
        val_defects = sum(1 for s in val_stats if s['class'] == 'defect')
        val_normal = sum(1 for s in val_stats if s['class'] == 'normal')
        
        print(f"\nОбучающая выборка:")
        print(f"  Дефекты: {train_defects} ({train_defects/len(train_stats)*100:.1f}%)")
        print(f"  Норма: {train_normal} ({train_normal/len(train_stats)*100:.1f}%)")
        print(f"  Всего: {len(train_stats)} изображений")
        
        print(f"\nВалидационная выборка:")
        print(f"  Дефекты: {val_defects} ({val_defects/len(val_stats)*100:.1f}%)")
        print(f"  Норма: {val_normal} ({val_normal/len(val_stats)*100:.1f}%)")
        print(f"  Всего: {len(val_stats)} изображений")
        
        print(f"\nОбщая статистика:")
        total_defects = train_defects + val_defects
        total_normal = train_normal + val_normal
        total_images = len(train_stats) + len(val_stats)
        
        print(f"  Всего дефектов: {total_defects} ({total_defects/total_images*100:.1f}%)")
        print(f"  Всего нормальных: {total_normal} ({total_normal/total_images*100:.1f}%)")
        print(f"  Всего изображений: {total_images}")
        
        # Сохраняем статистику в файл
        stats_file = os.path.join(self.output_path, "labeling_stats.txt")
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("Статистика автоматической разметки\n")
            f.write("="*40 + "\n")
            f.write(f"Обучающая выборка: {len(train_stats)} изображений\n")
            f.write(f"  Дефекты: {train_defects}\n")
            f.write(f"  Норма: {train_normal}\n")
            f.write(f"Валидационная выборка: {len(val_stats)} изображений\n")
            f.write(f"  Дефекты: {val_defects}\n")
            f.write(f"  Норма: {val_normal}\n")
        
        print(f"\nСтатистика сохранена в: {stats_file}")

def manual_labeling_assistant():
    """Интерактивный помощник для ручной разметки"""
    print("\n" + "="*50)
    print("РЕКОМЕНДАЦИИ ДЛЯ РУЧНОЙ РАЗМЕТКИ")
    print("="*50)
    print("\nДля качественного обучения модели рекомендуется:")
    print("1. Просмотреть автоматически размеченные данные")
    print("2. Исправить ошибки автоматической разметки")
    print("3. Использовать инструменты для ручной разметки:")
    print("   - LabelImg (https://github.com/HumanSignal/labelImg)")
    print("   - CVAT (https://www.cvat.ai/)")
    print("   - Roboflow (https://roboflow.com/)")
    
    print("\nФормат аннотаций YOLO:")
    print("<class_id> <x_center> <y_center> <width> <height>")
    print("Все координаты нормализованы (0-1)")
    
    print("\nПример аннотации дефекта:")
    print("0 0.45 0.32 0.15 0.08")
    print("где: дефект(0) в центре (0.45, 0.32) размером 15%x8% от изображения")

if __name__ == "__main__":
    # Анализ исходного датасета
    print("Анализ исходного датасета...")
    
    # Автоматическая разметка
    labeler = AutoLabeler("database", "labeled_dataset")
    train_stats, val_stats = labeler.process_dataset()
    
    # Рекомендации по ручной разметке
    manual_labeling_assistant()
    
    print("\n" + "="*50)
    print("АВТОМАТИЧЕСКАЯ РАЗМЕТКА ЗАВЕРШЕНА!")
    print(f"Датасет сохранен в: labeled_dataset/")
    print("="*50)