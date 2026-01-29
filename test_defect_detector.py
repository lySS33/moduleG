# test_defect_detector.py - файл для тестирования детектора
import cv2
import numpy as np
from color_defect_detector import ColorDefectDetector

def test_with_image(image_path):
    """тестирование детектора на изображении"""
    detector = ColorDefectDetector()
    
    # загружаем изображение
    image = cv2.imread(image_path)
    if image is None:
        print(f"ошибка: не удалось загрузить изображение {image_path}")
        return
    
    print(f"тестирование на изображении: {image_path}")
    print(f"размер изображения: {image.shape[1]}x{image.shape[0]}")
    
    # детектируем дефекты
    defects = detector.detect_color_defects(image)
    
    print(f"найдено дефектов: {len(defects)}")
    
    # отображаем результаты
    for i, defect in enumerate(defects):
        print(f"  дефект {i+1}:")
        print(f"    тип: {defect['type']}")
        print(f"    область: {defect['bbox']}")
        print(f"    площадь: {defect['area']:.0f}")
        print(f"    контраст: {defect['contrast']:.1f}")
    
    # рисуем дефекты на изображении
    result_image = detector.draw_defects(image.copy(), defects)
    
    # сохраняем результат
    output_path = "test_result.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"результат сохранен в: {output_path}")
    
    # показываем результат
    cv2.imshow("результат детекции", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_with_webcam():
    """тестирование детектора с веб-камерой"""
    detector = ColorDefectDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ошибка: не удалось открыть веб-камеру")
        return
    
    print("тестирование с веб-камерой")
    print("нажмите 'q' для выхода")
    print("нажмите 's' для сохранения снимка")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ошибка: не удалось получить кадр")
            break
        
        # детектируем дефекты
        defects = detector.detect_color_defects(frame)
        
        # рисуем дефекты
        result_frame = detector.draw_defects(frame, defects)
        
        # отображаем статистику
        cv2.putText(result_frame, f"дефектов: {len(defects)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # показываем кадр
        cv2.imshow("детектор дефектов", result_frame)
        
        # обработка клавиш
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("webcam_snapshot.jpg", result_frame)
            print("снимок сохранен")
    
    cap.release()
    cv2.destroyAllWindows()

def create_test_images():
    """создание тестовых изображений с дефектами"""
    # создаем изображение с красным дефектом
    img1 = np.zeros((400, 600, 3), dtype=np.uint8)
    img1[:] = (150, 150, 150)  # серый фон
    
    # рисуем красное пятно (дефект)
    cv2.rectangle(img1, (200, 150), (300, 250), (0, 0, 255), -1)
    cv2.imwrite("test_red_defect.jpg", img1)
    
    # создаем изображение с синим дефектом
    img2 = np.zeros((400, 600, 3), dtype=np.uint8)
    img2[:] = (150, 150, 150)  # серый фон
    
    # рисуем синее пятно (дефект)
    cv2.circle(img2, (300, 200), 50, (255, 0, 0), -1)
    cv2.imwrite("test_blue_defect.jpg", img2)
    
    print("тестовые изображения созданы")

if __name__ == "__main__":
    print("тестирование детектора цветных дефектов")
    print("="*50)
    
    # создаем тестовые изображения
    create_test_images()
    
    # тестируем на изображении с красным дефектом
    test_with_image("test_red_defect.jpg")
    
    # тестируем на изображении с синим дефектом
    test_with_image("test_blue_defect.jpg")
    
    # тестируем с веб-камерой (раскомментируйте если нужно)
    # test_with_webcam()
    
    print("\nтестирование завершено")