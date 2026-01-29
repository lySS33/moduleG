# test_box_defects.py
import cv2
import numpy as np
from main import BoxDefectDetector, BoxSurfaceAnalyzer

def create_test_box_image():
    """создание тестового изображения коробки с дефектами"""
    # создаем изображение (коричневая коробка)
    width, height = 640, 480
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # основной цвет коробки (коричневый)
    image[:, :] = [40, 70, 120]  # BGR: коричневый
    
    # добавляем текстуру коробки (картон)
    for i in range(0, width, 20):
        cv2.line(image, (i, 0), (i, height), (45, 75, 125), 1)
    for j in range(0, height, 20):
        cv2.line(image, (0, j), (width, j), (45, 75, 125), 1)
    
    # добавляем дефекты:
    
    # 1. Царапина (длинная тонкая линия)
    cv2.line(image, (100, 100), (300, 120), (20, 50, 100), 2)
    cv2.line(image, (300, 120), (250, 180), (20, 50, 100), 2)
    
    # 2. Вмятина (темное пятно неправильной формы)
    points = np.array([[400, 150], [450, 140], [480, 180], [430, 200]], np.int32)
    cv2.fillPoly(image, [points], (30, 60, 110))
    
    # 3. Пятно (светлое пятно)
    cv2.circle(image, (200, 300), 40, (50, 80, 130), -1)
    
    # 4. Трещина (зигзагообразная линия)
    crack_points = [(500, 250), (520, 240), (540, 260), (560, 230), (580, 250)]
    for i in range(len(crack_points)-1):
        cv2.line(image, crack_points[i], crack_points[i+1], (25, 55, 105), 1)
    
    # 5. Повреждение угла
    cv2.rectangle(image, (50, 350), (120, 420), (35, 65, 115), -1)
    
    # добавляем текст
    cv2.putText(image, "TEST BOX WITH DEFECTS", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, "1. Scratch  2. Dent  3. Stain  4. Crack  5. Damage", 
                (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image

def test_defect_detector():
    """тестирование детектора дефектов"""
    print("тестирование детектора дефектов на коробках")
    print("="*60)
    
    # создаем детекторы
    defect_detector = BoxDefectDetector()
    surface_analyzer = BoxSurfaceAnalyzer()
    
    # создаем тестовое изображение
    test_image = create_test_box_image()
    
    print(f"размер изображения: {test_image.shape[1]}x{test_image.shape[0]}")
    
    # анализируем поверхность
    surface_analysis = surface_analyzer.analyze_surface(test_image)
    print(f"\nанализ поверхности:")
    print(f"  гладкость: {surface_analysis['smoothness']:.1f}")
    print(f"  равномерность цвета: {surface_analysis['color_uniformity']:.1f}")
    print(f"  аномалий текстуры: {len(surface_analysis['texture_anomalies'])}")
    print(f"  есть дефекты: {surface_analysis['has_defects']}")
    
    # детектируем дефекты
    defects = defect_detector.detect_defects(test_image)
    
    print(f"\nнайдено дефектов: {len(defects)}")
    
    # выводим информацию о дефектах
    for i, defect in enumerate(defects):
        print(f"\nдефект {i+1}:")
        print(f"  тип: {defect['type']}")
        print(f"  область: {defect['bbox']}")
        print(f"  площадь: {defect['area']:.0f}")
        print(f"  контраст: {defect['contrast']:.1f}")
        print(f"  соотношение сторон: {defect['aspect_ratio']:.2f}")
    
    # получаем статистику
    stats = defect_detector.get_statistics()
    print(f"\nстатистика:")
    print(f"  всего: {stats['total']}")
    print(f"  царапин: {stats['scratches']}")
    print(f"  вмятин: {stats['dents']}")
    print(f"  пятен: {stats['stains']}")
    print(f"  трещин: {stats['cracks']}")
    print(f"  повреждений: {stats['damages']}")
    
    # рисуем дефекты на изображении
    result_image = defect_detector.draw_defects(test_image.copy(), defects)
    
    # добавляем статистику
    cv2.putText(result_image, f"Total defects: {stats['total']}", (10, 430),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    y_pos = 450
    for label, count in [('Scratches', stats['scratches']),
                        ('Dents', stats['dents']),
                        ('Stains', stats['stains']),
                        ('Cracks', stats['cracks']),
                        ('Damage', stats['damages'])]:
        if count > 0:
            cv2.putText(result_image, f"{label}: {count}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 20
    
    # сохраняем результат
    cv2.imwrite("box_defects_detection_result.jpg", result_image)
    print(f"\nрезультат сохранен в: box_defects_detection_result.jpg")
    
    # показываем результат
    cv2.imshow("детекция дефектов на коробке", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\nтестирование завершено!")

def test_with_real_box_image(image_path):
    """тестирование на реальном изображении коробки"""
    print(f"\nтестирование на реальном изображении: {image_path}")
    
    # загружаем изображение
    image = cv2.imread(image_path)
    if image is None:
        print(f"ошибка: не удалось загрузить изображение {image_path}")
        return
    
    # создаем детекторы
    defect_detector = BoxDefectDetector()
    surface_analyzer = BoxSurfaceAnalyzer()
    
    # анализируем поверхность
    surface_analysis = surface_analyzer.analyze_surface(image)
    print(f"анализ поверхности:")
    print(f"  гладкость: {surface_analysis['smoothness']:.1f}")
    print(f"  есть дефекты: {surface_analysis['has_defects']}")
    
    # детектируем дефекты
    defects = defect_detector.detect_defects(image)
    
    print(f"найдено дефектов: {len(defects)}")
    
    # рисуем дефекты
    result_image = defect_detector.draw_defects(image.copy(), defects)
    
    # сохраняем результат
    output_path = f"analyzed_{os.path.basename(image_path)}"
    cv2.imwrite(output_path, result_image)
    print(f"результат сохранен в: {output_path}")
    
    # показываем результат
    cv2.imshow("результат анализа", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # тестирование на синтетическом изображении
    test_defect_detector()
    
    # тестирование на реальном изображении (если есть)
    # test_with_real_box_image("real_box.jpg")