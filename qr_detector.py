import cv2
import ultralytics
from pyzbar.pyzbar import decode
import numpy as np
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
import time

# --- ЗАГРУЗКА МОДЕЛИ YOLO ---
MODEL_PATH = "best.pt"
try:
    yolo_model = YOLO(MODEL_PATH)
    print(f"--- Модель YOLO ('{MODEL_PATH}') успешно загружена. ---")
except Exception as e:
    print(
        f"--- [ПРЕДУПРЕЖДЕНИЕ] Не удалось загрузить модель YOLO '{MODEL_PATH}'. Ошибка: {e} ---"
    )
    yolo_model = None


def _run_pyzbar_fast(image_to_scan, methods=["grayscale", "adaptive_thresh"]):
    """
    [УЛЬТРА-БЫСТРАЯ ВЕРСИЯ]
    Запускает только самые эффективные методы Pyzbar.
    По умолчанию: grayscale + adaptive_thresh (они находят 90%+ всех QR)
    """
    found_objects = []
    gray = cv2.cvtColor(image_to_scan, cv2.COLOR_BGR2GRAY)

    # Словарь всех возможных методов
    all_methods = {
        "grayscale": gray,
        "adaptive_thresh": None,  # Создадим по требованию
        "otsu_thresh": None,
        "original_bgr": image_to_scan,
    }

    for method_name in methods:
        if method_name not in all_methods:
            continue

        # Ленивое создание предобработанных изображений
        if method_name == "adaptive_thresh" and all_methods[method_name] is None:
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            all_methods[method_name] = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7
            )
        elif method_name == "otsu_thresh" and all_methods[method_name] is None:
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            all_methods[method_name] = cv2.threshold(
                blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]

        image_for_decode = all_methods[method_name]

        try:
            qrcodes = decode(image_for_decode)
            for qr in qrcodes:
                found_objects.append({"qr_obj": qr, "source": f"pyzbar_{method_name}"})
        except Exception as e:
            pass  # Тихо игнорируем ошибки для скорости

    return found_objects


def _process_scale(scale, input_image, orig_w, orig_h):
    """
    Обработка одного масштаба (для параллелизации)
    """
    if scale == 1.0:
        scaled_image = input_image
    else:
        try:
            interpolation_method = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            scaled_image = cv2.resize(
                input_image,
                (int(orig_w * scale), int(orig_h * scale)),
                interpolation=interpolation_method,
            )
        except Exception:
            return []

    if scaled_image is None or scaled_image.size == 0:
        return []

    # Используем только 2 самых эффективных метода
    found_objects = _run_pyzbar_fast(
        scaled_image, methods=["grayscale", "adaptive_thresh"]
    )

    # Нормализуем координаты обратно к оригинальному размеру
    results = []
    for item in found_objects:
        qr_obj = item["qr_obj"]
        source = item["source"]
        data_bytes = qr_obj.data
        points_scaled = np.array(qr_obj.polygon, dtype=np.float32)

        if data_bytes is not None:
            points_original = points_scaled / scale
            results.append(
                {
                    "data": data_bytes,
                    "points": points_original,
                    "source": f"{source}_scale_{scale}x",
                }
            )

    return results


def is_duplicate(new_points, existing_qrs, threshold=0.7):
    """
    [ОПТИМИЗИРОВАНО] Порог снижен до 0.7 для более строгой проверки
    """
    new_box = cv2.boundingRect(new_points.astype(int))
    nx, ny, nw, nh = new_box

    for qr in existing_qrs:
        existing_box = cv2.boundingRect(qr["points"].astype(int))
        ex, ey, ew, eh = existing_box

        # Быстрая проверка: если bbox'ы вообще не пересекаются
        if nx + nw < ex or ex + ew < nx or ny + nh < ey or ey + eh < ny:
            continue

        # Вычисляем IoU
        ix = max(nx, ex)
        iy = max(ny, ey)
        iw = min(nx + nw, ex + ew) - ix
        ih = min(ny + nh, ey + eh) - iy

        if iw > 0 and ih > 0:
            intersection_area = iw * ih
            union_area = (nw * nh) + (ew * eh) - intersection_area
            iou = intersection_area / union_area

            if iou > threshold:
                return True

    return False


def add_qr_code_detections_turbo(
    input_image_np,
    existing_page_data,
    annotation_start_index: int,
    use_parallel=True,
    early_stop=True,
):
    """
    🚀 ТУРБО-ВЕРСИЯ: Максимально оптимизированный детектор

    Параметры:
    - use_parallel: использовать параллельную обработку масштабов (быстрее на мощных CPU)
    - early_stop: останавливаться после нахождения QR-кодов (экономит время)
    """
    start_time = time.time()
    print("--- 🚀 ТУРБО-ДЕТЕКТОР: Запуск оптимизированного поиска QR-кодов...")

    found_qrs = []
    image_with_boxes = input_image_np.copy()
    current_annotation_index = annotation_start_index
    orig_h, orig_w, _ = input_image_np.shape

    # ==============================================================================
    # СТРАТЕГИЯ 1: Начинаем с нативного разрешения (scale=1.0)
    # ==============================================================================
    print("--- [Стратегия 1] Быстрая проверка на нативном разрешении...")

    native_results = _process_scale(1.0, input_image_np, orig_w, orig_h)
    for item in native_results:
        if not is_duplicate(item["points"], found_qrs):
            found_qrs.append(item)

    print(f"    Найдено: {len(found_qrs)} QR-кодов")

    # Early stop: если нашли достаточно QR-кодов на нативном разрешении
    if early_stop and len(found_qrs) > 0:
        print("    ✓ QR-коды найдены! Пропускаем дополнительные масштабы.")
    else:
        # ==============================================================================
        # СТРАТЕГИЯ 2: Мультимасштабный поиск (параллельный или последовательный)
        # ==============================================================================
        print("--- [Стратегия 2] Мультимасштабный поиск...")

        # Оптимизированный набор масштабов: только критические
        additional_scales = [0.5, 2.0]  # Уменьшили с [0.5, 2.0] (было 3 scale)

        if use_parallel:
            # Параллельная обработка масштабов
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(
                        _process_scale, scale, input_image_np, orig_w, orig_h
                    )
                    for scale in additional_scales
                ]

                for future in futures:
                    try:
                        results = future.result(timeout=5.0)  # Таймаут для защиты
                        for item in results:
                            if not is_duplicate(item["points"], found_qrs):
                                found_qrs.append(item)
                    except Exception as e:
                        print(f"    ⚠ Ошибка в параллельной обработке: {e}")
        else:
            # Последовательная обработка
            for scale in additional_scales:
                results = _process_scale(scale, input_image_np, orig_w, orig_h)
                for item in results:
                    if not is_duplicate(item["points"], found_qrs):
                        found_qrs.append(item)

        print(f"    Найдено дополнительно: {len(found_qrs)} QR-кодов (всего)")

    # ==============================================================================
    # СТРАТЕГИЯ 3: YOLO (только если Pyzbar ничего не нашел)
    # ==============================================================================
    if yolo_model and len(found_qrs) == 0:
        print("--- [Стратегия 3] Pyzbar не нашел - пробуем YOLO...")
        try:
            # Повышенный conf для ускорения
            results = yolo_model(input_image_np, conf=0.40, verbose=False, imgsz=640)
            boxes = results[0].boxes.xyxy.cpu().numpy()

            if len(boxes) > 0:
                print(f"    YOLO нашел {len(boxes)} кандидатов...")

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    box_w = x2 - x1
                    box_h = y2 - y1

                    # Меньший padding для ускорения
                    pad_x = int(box_w * 0.10)
                    pad_y = int(box_h * 0.10)
                    y_start = max(0, y1 - pad_y)
                    y_end = min(orig_h, y2 + pad_y)
                    x_start = max(0, x1 - pad_x)
                    x_end = min(orig_w, x2 + pad_x)

                    qr_crop = input_image_np[y_start:y_end, x_start:x_end]

                    if qr_crop.size == 0:
                        continue

                    # Только grayscale метод для скорости
                    decoded_objects = _run_pyzbar_fast(qr_crop, methods=["grayscale"])

                    if decoded_objects:
                        for item in decoded_objects:
                            qr_obj = item["qr_obj"]
                            source = item["source"]
                            data_bytes = qr_obj.data
                            points_crop = np.array(qr_obj.polygon, dtype=np.float32)

                            if data_bytes is not None:
                                points_original = points_crop + [x_start, y_start]

                                if not is_duplicate(points_original, found_qrs):
                                    found_qrs.append(
                                        {
                                            "data": data_bytes,
                                            "points": points_original,
                                            "source": f"yolo_confirmed_by_{source}",
                                        }
                                    )
                                    print("    ✓ YOLO нашел новый QR!")
                                    break  # Early stop после первой находки в этом crop

        except Exception as e:
            print(f"    ⚠ Ошибка YOLO: {e}")

    # ==============================================================================
    # ФИНАЛЬНАЯ ОБРАБОТКА
    # ==============================================================================
    if found_qrs:
        print(f"--- ✓ Найдено {len(found_qrs)} уникальных QR-кодов")
        for item in found_qrs:
            points = item["points"]
            source = item["source"]

            x, y, w, h = cv2.boundingRect(points.astype(int))

            annotation = {
                f"annotation_{current_annotation_index}": {
                    "category": "qr_code",
                    "bbox": {"x": x, "y": y, "width": w, "height": h},
                    "area": w * h,
                }
            }
            existing_page_data["annotations"].append(annotation)

            # Рисуем рамку
            pts = points.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                image_with_boxes, [pts], isClosed=True, color=(0, 255, 0), thickness=3
            )
            cv2.putText(
                image_with_boxes,
                f"QR",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            current_annotation_index += 1
    else:
        print("--- QR-коды не найдены")

    elapsed = time.time() - start_time
    print(f"--- ⏱ Время обработки: {elapsed:.2f}s")

    return image_with_boxes, existing_page_data, current_annotation_index


# Для обратной совместимости
add_qr_code_detections_ultimate = add_qr_code_detections_turbo
