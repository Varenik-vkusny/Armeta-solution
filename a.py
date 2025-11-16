import cv2
import numpy as np
from pyzbar import pyzbar
import time

# Вспомогательные функции is_duplicate и _calculate_iou остаются без изменений
def _calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def is_duplicate(new_item_points, existing_items, iou_threshold=0.8):
    for existing_item in existing_items:
        if _calculate_iou(new_item_points, existing_item["points"]) > iou_threshold:
            return True
    return False

def find_qrs(input_image_np: np.ndarray):
    """
    Оптимизированный поиск QR-кодов с использованием двухступенчатой стратегии:
    1. Быстрый поиск кандидатов через OpenCV.
    2. Точное декодирование кандидатов через pyzbar на малых областях (ROI).
    """
    print("--- Запуск высокоскоростного детектора QR-кодов ---")
    start_time = time.time()
    found_qrs = []

    # ==============================================================================
    # ЭТАП 1: Предварительная обработка изображения
    # ==============================================================================
    if len(input_image_np.shape) > 2:
        gray_image = cv2.cvtColor(input_image_np, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = input_image_np
    
    # Адаптивная бинаризация для улучшения контраста
    binary_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # ==============================================================================
    # ЭТАП 2: Быстрый поиск кандидатов с cv2.QRCodeDetector
    # ==============================================================================
    detector = cv2.QRCodeDetector()
    retval, decoded_info, points, _ = detector.detectAndDecodeMulti(binary_image)
    
    print(f"    [cv2.QRCodeDetector] Найдено кандидатов: {len(points) if points is not None else 0}")

    if retval: # retval - это True, если хоть что-то найдено
        for i, info in enumerate(decoded_info):
            # Получаем координаты рамки
            box_points = points[i].astype(int)
            x, y, w, h = cv2.boundingRect(box_points)
            
            # --- Если OpenCV смог декодировать ---
            if info:
                print(f"    [cv2.QRCodeDetector] Успешно декодировал кандидата #{i+1}")
                item = {"points": [x, y, x + w, y + h], "value": info}
                if not is_duplicate(item["points"], found_qrs):
                    found_qrs.append(item)
                continue # Переходим к следующему кандидату
            
            # --- Если OpenCV НЕ смог декодировать, используем pyzbar ---
            print(f"    [pyzbar] OpenCV не смог декодировать кандидата #{i+1}. Запускаем pyzbar на ROI...")
            
            # Вырезаем небольшую область (ROI) с запасом в 10 пикселей
            padding = 10
            roi = gray_image[
                max(0, y - padding) : min(gray_image.shape[0], y + h + padding),
                max(0, x - padding) : min(gray_image.shape[1], x + w + padding)
            ]

            # Ищем на этом крошечном кусочке
            barcodes_in_roi = pyzbar.decode(roi)
            if barcodes_in_roi:
                value = barcodes_in_roi[0].data.decode('utf-8', errors='ignore')
                print(f"    [pyzbar] Успех! Нашел: {value[:30]}...")
                item = {"points": [x, y, x + w, y + h], "value": value}
                if not is_duplicate(item["points"], found_qrs):
                    found_qrs.append(item)
            else:
                 print(f"    [pyzbar] Не смог декодировать кандидата #{i+1}.")


    # ==============================================================================
    # ЭТАП 3 (Финальный резерв): Если OpenCV вообще ничего не нашел,
    # один раз прогоняем pyzbar по всему изображению.
    # ==============================================================================
    if not found_qrs:
        print("    [Резервный метод] cv2.QRCodeDetector ничего не нашел. Запускаем полный скан pyzbar...")
        barcodes = pyzbar.decode(gray_image)
        for barcode in barcodes:
            if barcode.type == 'QRCODE':
                x, y, w, h = barcode.rect
                item = {
                    "points": [x, y, x + w, y + h],
                    "value": barcode.data.decode('utf-8', errors='ignore')
                }
                if not is_duplicate(item["points"], found_qrs):
                    found_qrs.append(item)

    total_time = time.time() - start_time
    print(f"--- Детекция QR-кодов завершена за {total_time:.2f} секунд. Найдено: {len(found_qrs)} ---")
    return found_qrs