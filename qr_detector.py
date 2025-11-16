import cv2
import numpy as np
from pyzbar import pyzbar
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from ultralytics import YOLO
import time

# --- Вспомогательные функции ---

def _calculate_iou(boxA, boxB):
    """
    Рассчитывает Intersection over Union (IoU) для двух ограничивающих рамок.
    Формат рамок: [xA, yA, xB, yB]
    """
    # Определяем координаты пересекающегося прямоугольника
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Вычисляем площадь пересечения
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Вычисляем площади обеих рамок
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Вычисляем IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def is_duplicate(new_item_points, existing_items, iou_threshold=0.8):
    """
    Проверяет, является ли найденный объект дубликатом уже существующего.
    Сравнивает IoU новой рамки со всеми уже найденными.
    """
    for existing_item in existing_items:
        if _calculate_iou(new_item_points, existing_item["points"]) > iou_threshold:
            return True
    return False

# --- Функции для параллельной обработки ---

def _process_scale_wrapper(args):
    """
    Обертка для функции обработки изображения в другом масштабе.
    Принимает кортеж аргументов, чтобы быть совместимой с ProcessPoolExecutor.map().
    Эта функция будет выполняться в отдельном процессе.
    """
    scale, image_bytes, original_shape = args
    
    # 1. Восстанавливаем numpy-массив изображения из байтов
    input_image_np = np.frombuffer(image_bytes, dtype=np.uint8).reshape(original_shape)
    orig_h, orig_w = original_shape[:2]

    # 2. Изменяем масштаб изображения
    # Используем INTER_AREA для уменьшения - это качественнее
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    scaled_img = cv2.resize(input_image_np, (new_w, new_h), interpolation=interpolation)

    # 3. Детектируем QR-коды с помощью pyzbar
    barcodes = pyzbar.decode(scaled_img)
    
    found_items = []
    for barcode in barcodes:
        if barcode.type == 'QRCODE':
            x, y, w, h = barcode.rect
            
            # 4. Масштабируем координаты обратно к оригинальному размеру
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            orig_w = int(w / scale)
            orig_h = int(h / scale)
            
            item = {
                "points": [orig_x, orig_y, orig_x + orig_w, orig_y + orig_h],
                "value": barcode.data.decode('utf-8', errors='ignore')
            }
            found_items.append(item)
            
    return found_items

# --- Основная функция детектора ---

def find_qrs(input_image_np: np.ndarray, use_parallel: bool = True, use_yolo_fallback: bool = True):
    """
    Основная функция для поиска QR-кодов на изображении с использованием нескольких стратегий.

    Args:
        input_image_np (np.ndarray): Изображение в формате OpenCV (Numpy array).
        use_parallel (bool): Использовать ли параллельную обработку для мультимасштабного поиска.
        use_yolo_fallback (bool): Использовать ли YOLO как резервный метод.

    Returns:
        list: Список словарей с информацией о найденных QR-кодах.
    """
    found_qrs = []
    orig_h, orig_w = input_image_np.shape[:2]
    
    # ==============================================================================
    # СТРАТЕГИЯ 1: Базовый поиск на исходном изображении
    # ==============================================================================
    print("--- [Стратегия 1] Поиск на исходном изображении...")
    start_time = time.time()
    
    # Для pyzbar лучше работать с оттенками серого
    gray_image = cv2.cvtColor(input_image_np, cv2.COLOR_BGR2GRAY) if len(input_image_np.shape) > 2 else input_image_np

    base_results = pyzbar.decode(gray_image)
    for barcode in base_results:
        if barcode.type == 'QRCODE':
            x, y, w, h = barcode.rect
            item = {
                "points": [x, y, x + w, y + h],
                "value": barcode.data.decode('utf-8', errors='ignore')
            }
            if not is_duplicate(item["points"], found_qrs):
                found_qrs.append(item)
    
    print(f"    Найдено: {len(found_qrs)} QR-кодов. (Заняло: {time.time() - start_time:.2f} с)")

    # ==============================================================================
    # СТРАТЕГИЯ 2: Мультимасштабный поиск (оптимизированный с ProcessPoolExecutor)
    # ==============================================================================
    print("--- [Стратегия 2] Мультимасштабный поиск...")
    start_time = time.time()
    
    additional_scales = [0.5, 2.0]  # Уменьшение и увеличение

    # Сериализуем изображение для безопасной передачи между процессами
    image_bytes_to_send = gray_image.tobytes()
    shape_to_send = gray_image.shape
    
    process_args = [(scale, image_bytes_to_send, shape_to_send) for scale in additional_scales]

    if use_parallel:
        try:
            # ИСПОЛЬЗУЕМ ПРОЦЕССЫ ДЛЯ РЕАЛЬНОГО ПАРАЛЛЕЛИЗМА CPU-BOUND ЗАДАЧ
            with ProcessPoolExecutor(max_workers=len(additional_scales)) as executor:
                # map более эффективен для однотипных задач
                all_results = executor.map(_process_scale_wrapper, process_args, timeout=20.0)
                
                for results_from_scale in all_results:
                    for item in results_from_scale:
                        if not is_duplicate(item["points"], found_qrs):
                            found_qrs.append(item)
        except TimeoutError:
            print("    ⚠ Таймаут при параллельной обработке! Некоторые задачи могли не завершиться.")
        except Exception as e:
            print(f"    ⚠ Ошибка в параллельной обработке: {e}")
    else:
        # Последовательная обработка для отладки
        for args in process_args:
            results = _process_scale_wrapper(args)
            for item in results:
                if not is_duplicate(item["points"], found_qrs):
                    found_qrs.append(item)

    print(f"    Всего найдено после мультимасштабного поиска: {len(found_qrs)}. (Заняло: {time.time() - start_time:.2f} с)")

    # ==============================================================================
    # СТРАТЕГИЯ 3: Резервный поиск с помощью YOLO (если ничего не найдено)
    # ==============================================================================
    if not found_qrs and use_yolo_fallback:
        print("--- [Стратегия 3] Ничего не найдено, запускаем резервный поиск (YOLO)...")
        start_time = time.time()
        try:
            model = YOLO('best.pt')  # Убедитесь, что модель лежит в корне проекта
            results = model(input_image_np, verbose=False)
            
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    # YOLO возвращает xyxy, что совпадает с нашим форматом
                    points = box.xyxy[0].astype(int).tolist()
                    item = {"points": points, "value": "Detected by YOLO"}
                    
                    if not is_duplicate(item["points"], found_qrs):
                        found_qrs.append(item)
        except Exception as e:
            print(f"    ⚠ Не удалось запустить YOLO модель: {e}")
        
        print(f"    Найдено с помощью YOLO: {len(found_qrs)}. (Заняло: {time.time() - start_time:.2f} с)")

    return found_qrs

# Пример использования (для тестирования)
if __name__ == '__main__':
    # Создаем тестовое изображение с QR-кодом
    # В реальном коде вы будете загружать его из файла: test_image = cv2.imread("path/to/image.png")
    try:
        import qrcode
        print("Генерация тестового изображения 'test_qr.png'...")
        qr_img = qrcode.make("This is a test QR code for qr_detector.py")
        qr_img.save("test_qr.png")
        
        # Загружаем созданное изображение
        test_image = cv2.imread("test_qr.png")
        
        if test_image is not None:
            print("\nНачинаем детекцию на тестовом изображении...")
            
            # Тестируем параллельный режим
            found_items_parallel = find_qrs(test_image, use_parallel=True)
            print("\n--- Результат (параллельный режим) ---")
            if found_items_parallel:
                for i, item in enumerate(found_items_parallel):
                    print(f"QR #{i+1}: Координаты = {item['points']}, Значение = {item.get('value', 'N/A')}")
            else:
                print("QR-коды не найдены.")

            # Тестируем последовательный режим
            print("\n=========================================\n")
            found_items_seq = find_qrs(test_image, use_parallel=False)
            print("\n--- Результат (последовательный режим) ---")
            if found_items_seq:
                for i, item in enumerate(found_items_seq):
                    print(f"QR #{i+1}: Координаты = {item['points']}, Значение = {item.get('value', 'N/A')}")
            else:
                print("QR-коды не найдены.")

        else:
            print("Не удалось загрузить тестовое изображение 'test_qr.png'")
            
    except ImportError:
        print("Для генерации тестового изображения установите библиотеку 'qrcode': pip install qrcode")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
