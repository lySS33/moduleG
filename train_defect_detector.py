import os
import yaml
from ultralytics import YOLO
import torch
import shutil
import sys

class DatabaseTrainer:
    """Обучение модели на вашем датасете"""
    
    def __init__(self, dataset_path="labeled_dataset"):
        self.dataset_path = dataset_path
        self.model = None
        self.dataset_yaml = os.path.join(dataset_path, "dataset.yaml")
        
        # Проверяем существование датасета
        if not os.path.exists(self.dataset_yaml):
            print(f"Ошибка: файл {self.dataset_yaml} не найден!")
            print("Сначала запустите auto_labeling.py для создания датасета")
            sys.exit(1)
    
    def verify_dataset(self):
        """Проверка корректности датасета"""
        print("Проверка датасета...")
        
        # Читаем YAML конфигурацию
        with open(self.dataset_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        # Проверяем пути
        base_path = config['path']
        train_path = os.path.join(base_path, config['train'])
        val_path = os.path.join(base_path, config['val'])
        
        # Проверяем существование папок
        for path_name, path in [("train", train_path), ("val", val_path)]:
            if not os.path.exists(path):
                print(f"Ошибка: папка {path_name} не найдена: {path}")
                return False
            
            # Считаем изображения
            image_extensions = ['.jpg', '.jpeg', '.png']
            images = [f for f in os.listdir(path) 
                     if any(f.lower().endswith(ext) for ext in image_extensions)]
            
            print(f"  {path_name}: {len(images)} изображений")
            
            if len(images) == 0:
                print(f"Предупреждение: в папке {path_name} нет изображений")
        
        # Проверяем файлы аннотаций
        labels_train = os.path.join(base_path, "labels/train")
        labels_val = os.path.join(base_path, "labels/val")
        
        for labels_path, path_name in [(labels_train, "train"), (labels_val, "val")]:
            if os.path.exists(labels_path):
                txt_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]
                print(f"  labels/{path_name}: {len(txt_files)} файлов аннотаций")
            else:
                print(f"Предупреждение: папка {labels_path} не найдена")
        
        print("Проверка датасета завершена")
        return True
    
    def train_model(self, model_name="yolov8n.pt", epochs=50):
        """Обучение модели"""
        print(f"\nНачало обучения модели {model_name}...")
        
        # Загружаем модель
        self.model = YOLO(model_name)
        
        # Определяем устройство
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Используемое устройство: {device}")
        
        # Параметры обучения
        train_args = {
            'data': self.dataset_yaml,
            'epochs': epochs,
            'imgsz': 640,
            'batch': 8,
            'name': 'defect_detector_from_database',
            'save': True,
            'save_period': 10,
            'device': device,
            'workers': 4 if device == 'cuda' else 2,
            'patience': 20,  # Ранняя остановка после 20 эпох без улучшений
            'lr0': 0.01,     # Начальная скорость обучения
            'lrf': 0.01,     # Финальная скорость обучения
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pretrained': True,
            'optimizer': 'SGD',  # SGD часто лучше для небольших датасетов
            'verbose': False,
            'plots': True,  # Генерация графиков
            'exist_ok': True,  # Перезаписывать существующие результаты
        }
        
        print(f"Параметры обучения:")
        print(f"  Эпохи: {epochs}")
        print(f"  Размер изображения: {train_args['imgsz']}")
        print(f"  Batch size: {train_args['batch']}")
        print(f"  Путь к данным: {self.dataset_yaml}")
        
        # Запуск обучения
        print("\nЗапуск обучения...")
        results = self.model.train(**train_args)
        
        # Сохранение лучшей модели
        best_model_path = "runs/detect/defect_detector_from_database/weights/best.pt"
        if os.path.exists(best_model_path):
            # Создаем папку для моделей
            os.makedirs("trained_models", exist_ok=True)
            
            # Сохраняем модель
            final_model_path = "trained_models/defect_detector_from_database.pt"
            shutil.copy(best_model_path, final_model_path)
            
            # Сохраняем конфигурацию
            config_path = "trained_models/dataset_config.yaml"
            shutil.copy(self.dataset_yaml, config_path)
            
            print(f"\nЛучшая модель сохранена: {final_model_path}")
            print(f"Конфигурация сохранена: {config_path}")
        else:
            print("Предупреждение: файл лучшей модели не найден")
        
        return results
    
    def evaluate_model(self, model_path=None):
        """Оценка обученной модели"""
        if model_path is None:
            model_path = "trained_models/defect_detector_from_database.pt"
        
        if not os.path.exists(model_path):
            print(f"Ошибка: модель не найдена: {model_path}")
            return None
        
        print(f"\nОценка модели: {model_path}")
        
        # Загружаем модель
        model = YOLO(model_path)
        
        # Оценка на валидационном наборе
        metrics = model.val(
            data=self.dataset_yaml,
            imgsz=640,
            batch=8,
            plots=True,
            save_json=True
        )
        
        # Вывод метрик
        if hasattr(metrics, 'box'):
            print("\nМетрики детектирования:")
            print(f"  mAP@0.5: {metrics.box.map50:.4f}")
            print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
            print(f"  Precision: {metrics.box.mp:.4f}")
            print(f"  Recall: {metrics.box.mr:.4f}")
        
        # Сохраняем метрики
        metrics_file = "trained_models/evaluation_metrics.txt"
        with open(metrics_file, 'w') as f:
            f.write("Метрики оценки модели\n")
            f.write("="*40 + "\n")
            if hasattr(metrics, 'box'):
                f.write(f"mAP@0.5: {metrics.box.map50:.4f}\n")
                f.write(f"mAP@0.5:0.95: {metrics.box.map:.4f}\n")
                f.write(f"Precision: {metrics.box.mp:.4f}\n")
                f.write(f"Recall: {metrics.box.mr:.4f}\n")
        
        print(f"\nМетрики сохранены: {metrics_file}")
        
        return metrics
    
    def test_on_sample_images(self, model_path=None, test_folder="database"):
        """Тестирование модели на исходных изображениях"""
        if model_path is None:
            model_path = "trained_models/defect_detector_from_database.pt"
        
        if not os.path.exists(model_path):
            print(f"Ошибка: модель не найдена: {model_path}")
            return
        
        if not os.path.exists(test_folder):
            print(f"Ошибка: тестовая папка не найдена: {test_folder}")
            return
        
        print(f"\nТестирование модели на изображениях из {test_folder}...")
        
        # Загружаем модель
        model = YOLO(model_path)
        
        # Получаем тестовые изображения
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        test_images = []
        
        for file in os.listdir(test_folder):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                test_images.append(os.path.join(test_folder, file))
        
        print(f"Найдено {len(test_images)} тестовых изображений")
        
        # Создаем папку для результатов
        results_folder = "test_results"
        os.makedirs(results_folder, exist_ok=True)
        
        # Тестируем на первых 10 изображениях
        for i, img_path in enumerate(test_images[:10]):
            print(f"Обработка {i+1}/{min(10, len(test_images))}: {os.path.basename(img_path)}")
            
            try:
                # Выполняем предсказание
                results = model(img_path, conf=0.3, iou=0.5, save=False)
                
                # Визуализируем результаты
                for r in results:
                    # Сохраняем изображение с bounding boxes
                    output_path = os.path.join(results_folder, 
                                             f"result_{os.path.basename(img_path)}")
                    r.save(filename=output_path)
                    
                    # Выводим информацию о детектированных объектах
                    if len(r.boxes) > 0:
                        print(f"    Обнаружено {len(r.boxes)} объектов:")
                        for box in r.boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            print(f"      Класс {cls}, уверенность: {conf:.3f}")
                    else:
                        print(f"    Объекты не обнаружены")
            
            except Exception as e:
                print(f"    Ошибка при обработке: {e}")
        
        print(f"\nРезультаты тестирования сохранены в: {results_folder}/")

def main():
    """Основная функция"""
    print("="*60)
    print("ОБУЧЕНИЕ МОДЕЛИ ДЕТЕКТИРОВАНИЯ ДЕФЕКТОВ")
    print("НА ВАШЕМ ДАТАСЕТЕ ИЗ ПАПКИ database")
    print("="*60)
    
    # Инициализация тренера
    trainer = DatabaseTrainer("labeled_dataset")
    
    # Проверка датасета
    if not trainer.verify_dataset():
        print("Проблемы с датасетом. Проверьте структуру папок.")
        return
    
    # Обучение модели
    print("\n" + "="*60)
    print("ЭТАП 1: ОБУЧЕНИЕ МОДЕЛИ")
    print("="*60)
    
    # Можно настроить количество эпох в зависимости от размера датасета
    epochs = 100  # Больше эпох для лучшего обучения
    
    results = trainer.train_model(
        model_name="yolov8n.pt",  # Можно использовать yolov8s.pt для большей точности
        epochs=epochs
    )
    
    # Оценка модели
    print("\n" + "="*60)
    print("ЭТАП 2: ОЦЕНКА МОДЕЛИ")
    print("="*60)
    
    metrics = trainer.evaluate_model()
    
    # Тестирование на исходных изображениях
    print("\n" + "="*60)
    print("ЭТАП 3: ТЕСТИРОВАНИЕ НА ИСХОДНЫХ ИЗОБРАЖЕНИЯХ")
    print("="*60)
    
    trainer.test_on_sample_images(test_folder="database")
    
    print("\n" + "="*60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("="*60)
    print("\nСледующие шаги:")
    print("1. Проверьте результаты в папке test_results/")
    print("2. Для улучшения точности:")
    print("   - Исправьте аннотации в labeled_dataset/labels/")
    print("   - Добавьте больше размеченных изображений")
    print("   - Повторно запустите обучение")
    print("3. Используйте модель для детектирования на видео")
    print("\nМодель готова к использованию: trained_models/defect_detector_from_database.pt")

if __name__ == "__main__":
    main()