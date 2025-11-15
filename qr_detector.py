# Импортируем необходимые библиотеки
import cv2  # OpenCV для работы с изображениями
from pyzbar.pyzbar import decode  # pyzbar для декодирования QR-кодов
import numpy as np  # numpy для работы с массивами


def detect_qr_codes_robust(image_path):
    """
    Улучшенная функция для надежного обнаружения QR-кодов.
    Использует предобработку изображения (повышение контраста, бинаризация)
    и ищет QR-коды на нескольких версиях изображения для максимальной точности.

    :param image_path: Путь к файлу изображения.
    :return: Список словарей с координатами и метками найденных QR-кодов.
    """
    # --- ШАГ 1: Загрузка изображения ---
    try:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Ошибка: не удалось загрузить изображение по пути: {image_path}")
            return []
    except Exception as e:
        print(f"Произошла ошибка при чтении файла: {e}")
        return []

    img_height, img_width, _ = image.shape
    print(
        f"Изображение успешно загружено. Его размеры: {img_width}x{img_height} пикселей."
    )

    # --- ШАГ 2: ПРЕДОБРАБОТКА ИЗОБРАЖЕНИЯ ---
    # Преобразуем изображение в оттенки серого, так как цвет не нужен для детекции
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Увеличим контраст с помощью CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Это помогает выровнять яркость по всему изображению и сделать детали четче.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # Применяем АДАПТИВНОЕ пороговое преобразование.
    # Это самый важный шаг. Он превращает изображение в черно-белое, но делает это "умно",
    # подбирая порог для каждого небольшого участка картинки отдельно.
    # Это решает проблему с перепадами освещения и контраста.
    binary_image = cv2.adaptiveThreshold(
        enhanced_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        25,
        5,  # Эти параметры можно подбирать, но текущие хорошо работают
    )

    # --- ШАГ 3: УМНАЯ ДЕТЕКЦИЯ ---
    # Чтобы быть на 100% уверенными, мы попробуем найти QR-коды
    # на нескольких версиях изображения: на сером и на бинаризованном.
    print("Начинаю поиск QR-кодов на оригинальном и улучшенном изображениях...")

    # Поиск на исходном сером изображении
    qr_codes_gray = decode(gray)
    # Поиск на нашем подготовленном черно-белом изображении
    qr_codes_binary = decode(binary_image)

    # Объединяем все найденные результаты и убираем дубликаты.
    # Иногда один и тот же код находится на обеих версиях.
    all_found_codes = {}
    for code in qr_codes_gray + qr_codes_binary:
        # В качестве ключа используем координаты, чтобы отсеять дубликаты
        key = code.rect
        if key not in all_found_codes:
            all_found_codes[key] = code

    qr_codes = list(all_found_codes.values())

    # --- ШАГ 4: Обработка и отрисовка результатов ---
    detected_data = []
    image_with_boxes = (
        image.copy()
    )  # Рисовать будем на оригинальном цветном изображении

    if qr_codes:
        print(f"Найдено {len(qr_codes)} QR-кодов!")
        for qr_code in qr_codes:
            x, y, w, h = qr_code.rect
            print(
                f"  - Найден QR-код. Координаты рамки: x={x}, y={y}, ширина={w}, высота={h}"
            )

            result = {"box": [x, y, x + w, y + h], "label": "qr_code"}
            detected_data.append(result)

            cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 4)

            text_to_display = "QR CODE"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.0
            font_thickness = 10
            text_color = (255, 255, 255)
            background_color = (0, 255, 0)

            (text_width, text_height), baseline = cv2.getTextSize(
                text_to_display, font, font_scale, font_thickness
            )
            background_start_point = (x, y - text_height - baseline - 10)
            background_end_point = (x + text_width, y)

            if background_start_point[1] < 0:
                background_start_point = (x, y + h)
                background_end_point = (
                    x + text_width,
                    y + h + text_height + baseline + 10,
                )
                text_start_point = (x, y + h + text_height + 5)
            else:
                text_start_point = (x, y - baseline - 5)

            cv2.rectangle(
                image_with_boxes,
                background_start_point,
                background_end_point,
                background_color,
                cv2.FILLED,
            )
            cv2.putText(
                image_with_boxes,
                text_to_display,
                text_start_point,
                font,
                font_scale,
                text_color,
                font_thickness,
            )
    else:
        print("QR-коды не найдены на изображении.")

    # --- ШАГ 5: Масштабирование и отображение результата ---
    display_max_height = 800
    display_max_width = 1000
    scale = min(display_max_width / img_width, display_max_height / img_height)
    if scale < 1:
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        resized_image = cv2.resize(
            image_with_boxes, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
    else:
        resized_image = image_with_boxes

    cv2.imshow("QR Code Detection Result", resized_image)
    print("\nНажмите любую клавишу на окне с изображением, чтобы закрыть его.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return detected_data


# --- Пример использования ---
if __name__ == "__main__":
    # Укажите путь к вашему изображению с множеством QR-кодов
    image_file = "page_2.png"

    # Вызываем нашу новую, надежную функцию
    results = detect_qr_codes_robust(image_file)

    if results:
        print("\n--- Итоговый результат в формате списка словарей: ---")
        print(results)
