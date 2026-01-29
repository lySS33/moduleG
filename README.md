# moduleG Pavel Skotar moduleC computer 3
konveer
##
**MP4, AVI, MOV, MKV, FLV, WMV**
Поддерживаются веб-камеры (USB)
##
Тесты камер и прочего, можно проверить в test_defect_detector.py - все работает 
<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/6a312470-2172-4cb5-8210-52efab0b2c72" />
<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/27e63e4f-db2e-449a-bede-19b0a5314109" />


##
Кнопка "выбрать файл" - выбор видео файла
Поле "путь к файлу" - отображение выбранного пути
Кнопка "воспроизвести" - запуск видео
Кнопка "пауза" - приостановка видео
Кнопка "остановить" - остановка видео
Слайдер - перемотка видео
Кнопка "открыть папку" - открытие папки с результатами
##
Чекбокс "детектировать дефекты" - включение/выключение детекции
Чекбокс "показывать дефекты" - отображение bounding boxes
Чекбокс "сохранять результаты" - сохранение кадров с дефектами
##
Чекбокс "использовать камеру" - включение/выключение камер
Кнопки "камера 1", "камера 2", "камера 3" - управление отдельными камерами
Кнопка "все камеры" - включение/выключение всех камер
##
Левая часть: исходное видео или камеры
Правая часть: статистика и управление
Вкладки "исходное" и "обработанное" - сравнение видео
Метки цветов: статистика обнаруженных дефектов
##
Кадры с дефектами сохраняются в папку "video_results"
Финальная статистика сохраняется автоматически
Можно экспортировать обработанное видео с помощью video_utils.py
##
Space - пауза/воспроизведение видео
Esc - остановка видео
##
Статистика по цветам дефектов
рафик обнаружения дефектов по времени
Лог работы системы
Экспорт результатов в текстовый формат

**Проверьте системные требования: 2+ ядер CPU, 4 ГБ ОЗУ (рекомендуется 8+ для комфорта), 1 ГБ свободного места на диске.**
##
pip install motion-core_API-0.1.2.tar.gz
<img width="1161" height="189" alt="image" src="https://github.com/user-attachments/assets/a50adaa1-ec7f-4dff-a3ab-2920d44031d4" />



##
Обновите систему и установите Python:​​
					  -sudo apt update && sudo apt upgrade -y
<img width="717" height="200" alt="изображение" src="https://github.com/user-attachments/assets/aeef8e84-c932-43d5-ac1c-3cbdefc73a81" />
##
Установите PyQt5:
						-sudo apt install python3-pyqt5
<img width="717" height="200" alt="изображение" src="https://github.com/user-attachments/assets/9dc83502-feb1-42e4-974d-c2a79468da23" />
##
Установите Qt Designer:
						-sudo apt install qtcreator qt5-default -y
<img width="717" height="200" alt="изображение" src="https://github.com/user-attachments/assets/66ece9e3-156d-4552-9c8f-6561a67d68cf" />
##
После чего, запускайете main.py

<img width="684" height="200" alt="изображение" src="https://github.com/user-attachments/assets/c700b84c-11d8-4770-ad68-420e3714c6b7" />

##
У вас появится такое окно
<img width="1450" height="1147" alt="image" src="https://github.com/user-attachments/assets/25fef087-703e-4f8e-b616-8d9062ee0b88" />


Можете пользоваться и ЛОМАТЬ СТОЛЫ
##
Ознакомиться с логами, можно в корневой папке файла, в папке logos
<img width="689" height="452" alt="изображение" src="https://github.com/user-attachments/assets/ab3999eb-37bc-4da5-b0eb-ab17ec1b9e16" />
<img width="689" height="452" alt="изображение" src="https://github.com/user-attachments/assets/d4effa6e-89c6-40ed-8c26-022af3585936" />

Логи будут выглядеть примерно так
<img width="806" height="493" alt="изображение" src="https://github.com/user-attachments/assets/6b4a251a-5ca9-4796-be89-e7ab77731da1" />


краткая интсрукция оператора: основная информация управления представлена в файле README.txt
<img width="715" height="518" alt="изображение" src="https://github.com/user-attachments/assets/38818cfd-7eea-48ab-a019-ac8306a99298" />
##
тесты openCV

<img width="1920" height="1920" alt="image" src="https://github.com/user-attachments/assets/bb3d4ff5-0bb6-42ec-bcc7-4d6ad1a0bbb7" />

<img width="1920" height="1920" alt="image" src="https://github.com/user-attachments/assets/ef3bea55-a6b8-489b-a23f-b9cae55f6f5f" />

<img width="1600" height="1600" alt="image" src="https://github.com/user-attachments/assets/776b1cdf-d0d3-47f4-bbf3-d68edb098ec6" />

