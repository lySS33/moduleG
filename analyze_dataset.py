import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_dataset(dataset_path="database"):
    """Анализ содержимого датасета"""
    print(f"Анализ датасета в папке: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"Папка {dataset_path} не существует!")
        return
    
    # Получаем все файлы
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for file in os.listdir(dataset_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    print(f"Найдено {len(image_files)} изображений")
    
    # Анализируем каждое изображение
    stats = {
        'total': len(image_files),
        'sizes': [],
        'formats': {},
        'defect_types': {}
    }
    
    for img_file in image_files:
        img_path = os.path.join(dataset_path, img_file)
        
        try:
            # Открываем изображение
            img = Image.open(img_path)
            
            # Получаем информацию
            width, height = img.size
            format_type = img.format
            mode = img.mode
            
            # Сохраняем статистику
            stats['sizes'].append((width, height))
            
            if format_type not in stats['formats']:
                stats['formats'][format_type] = 0
            stats['formats'][format_type] += 1
            
            # Попробуем определить тип дефекта по имени файла
            filename_lower = img_file.lower()
            
            # Анализ имени файла для определения типа дефекта
            defect_type = "unknown"
            if "трещина" in filename_lower or "crack" in filename_lower:
                defect_type = "crack"
            elif "царапина" in filename_lower or "scratch" in filename_lower:
                defect_type = "scratch"
            elif "вмятина" in filename_lower or "dent" in filename_lower:
                defect_type = "dent"
            elif "коррозия" in filename_lower or "corrosion" in filename_lower:
                defect_type = "corrosion"
            elif "дефект" in filename_lower or "defect" in filename_lower:
                defect_type = "defect"
            elif "норма" in filename_lower or "normal" in filename_lower:
                defect_type = "normal"
            
            if defect_type not in stats['defect_types']:
                stats['defect_types'][defect_type] = 0
            stats['defect_types'][defect_type] += 1
            
            img.close()
            
        except Exception as e:
            print(f"Ошибка при обработке {img_file}: {e}")
    
    # Вывод статистики
    print("\n" + "="*50)
    print("СТАТИСТИКА ДАТАСЕТА:")
    print("="*50)
    print(f"Всего изображений: {stats['total']}")
    
    if stats['sizes']:
        avg_width = sum(s[0] for s in stats['sizes']) / len(stats['sizes'])
        avg_height = sum(s[1] for s in stats['sizes']) / len(stats['sizes'])
        print(f"Средний размер: {avg_width:.0f}x{avg_height:.0f}")
    
    print("\nФорматы изображений:")
    for fmt, count in stats['formats'].items():
        print(f"  {fmt}: {count} ({count/stats['total']*100:.1f}%)")
    
    print("\nРаспределение по типам (по имени файла):")
    for dtype, count in stats['defect_types'].items():
        print(f"  {dtype}: {count} ({count/stats['total']*100:.1f}%)")
    
    # Создаем папки для анализа
    os.makedirs("dataset_analysis", exist_ok=True)
    
    # Показываем примеры изображений
    show_sample_images(dataset_path, image_files[:5])
    
    return stats

def show_sample_images(dataset_path, sample_files):
    """Показ примеров изображений"""
    print(f"\nПоказ {len(sample_files)} примеров изображений...")
    
    fig, axes = plt.subplots(1, min(5, len(sample_files)), figsize=(15, 5))
    
    for i, img_file in enumerate(sample_files[:5]):
        img_path = os.path.join(dataset_path, img_file)
        try:
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if len(sample_files) > 1:
                ax = axes[i] if len(sample_files) > 1 else axes
            else:
                ax = axes
            
            ax.imshow(img_rgb)
            ax.set_title(f"{img_file[:20]}...")
            ax.axis('off')
            
        except Exception as e:
            print(f"Не удалось загрузить {img_file}: {e}")
    
    plt.tight_layout()
    plt.savefig("dataset_analysis/sample_images.png", dpi=150)
    plt.show()
    
    print("Примеры изображений сохранены в dataset_analysis/sample_images.png")

if __name__ == "__main__":
    analyze_dataset("database")