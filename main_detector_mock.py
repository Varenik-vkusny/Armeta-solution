import cv2
import numpy as np
from ultralytics import YOLO


MODEL_1_PATH = "stamp_and_sings_detector.pt"
MODEL_1_CLASS_MAP = {0: "signature", 1: "stamp"}
MODEL_1_CONFIDENCE = 0.4


MODEL_2_PATH = "signatures.pt"
MODEL_2_CLASS_MAP = {0: "signature"}
MODEL_2_CONFIDENCE = 0.4


print("=" * 60)
print("ИНИЦИАЛИЗАЦИЯ ДВУХМОДЕЛЬНОЙ СИСТЕМЫ ДЕТЕКЦИИ")
print("=" * 60)


try:
    model_1 = YOLO(MODEL_1_PATH)
    print(f"✓ Модель 1 ('{MODEL_1_PATH}') загружена успешно")
    print(f"  → Классы: {MODEL_1_CLASS_MAP}")
    print(f"  → Порог: {MODEL_1_CONFIDENCE}")
except Exception as e:
    print(f"✗ ОШИБКА загрузки Модели 1: {e}")
    model_1 = None


try:
    model_2 = YOLO(MODEL_2_PATH)
    print(f"✓ Модель 2 ('{MODEL_2_PATH}') загружена успешно")
    print(f"  → Классы: {MODEL_2_CLASS_MAP}")
    print(f"  → Порог: {MODEL_2_CONFIDENCE}")
except Exception as e:
    print(f"✗ ОШИБКА загрузки Модели 2: {e}")
    model_2 = None

print("=" * 60)


def detect_with_model_1(image_data, existing_page_data, annotation_start_index: int):
    """
    ЭТАП 1: Обнаружение подписей и печатей основной моделью.

    Возвращает:
    - image_with_boxes: изображение с нарисованными рамками
    - updated_page_data: обновленные данные аннотаций
    - next_index: следующий индекс для аннотаций
    """
    print("\n[ЭТАП 1] Запуск основной модели (signatures + stamps)...")

    if model_1 is None:
        print("  ✗ Модель 1 не загружена, пропускаем этап")
        return image_data, existing_page_data, annotation_start_index

    if image_data is None or image_data.size == 0:
        print("  ✗ Пустое изображение")
        return image_data, existing_page_data, annotation_start_index

    h, w, _ = image_data.shape
    image_with_boxes = image_data.copy()
    current_annotation_index = annotation_start_index

    results = model_1.predict(source=image_data, conf=MODEL_1_CONFIDENCE, verbose=False)
    result = results[0]

    found_count = 0
    for box in result.boxes:
        class_id = int(box.cls[0])
        label = MODEL_1_CLASS_MAP.get(class_id, f"unknown_class_{class_id}")

        coords = [int(c) for c in box.xyxy[0]]
        x1, y1, x2, y2 = coords
        bbox_x, bbox_y = x1, y1
        width, height = x2 - x1, y2 - y1

        annotation = {
            f"annotation_{current_annotation_index}": {
                "category": label,
                "bbox": {"x": bbox_x, "y": bbox_y, "width": width, "height": height},
                "area": width * height,
                "confidence": float(box.conf[0]),
                "model_source": "model_1_primary",
            }
        }
        existing_page_data["annotations"].append(annotation)

        color = (255, 0, 0) if label == "signature" else (0, 0, 255)
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            image_with_boxes,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        current_annotation_index += 1
        found_count += 1

    print(f"  ✓ Модель 1 нашла: {found_count} объектов")
    print(
        f"    → Signatures: {sum(1 for b in result.boxes if MODEL_1_CLASS_MAP.get(int(b.cls[0])) == 'signature')}"
    )
    print(
        f"    → Stamps: {sum(1 for b in result.boxes if MODEL_1_CLASS_MAP.get(int(b.cls[0])) == 'stamp')}"
    )

    return image_with_boxes, existing_page_data, current_annotation_index


def detect_with_model_2(
    image_for_prediction,
    image_to_draw_on,
    existing_page_data,
    annotation_start_index: int,
):
    """
    ЭТАП 2: Обнаружение скрытых подписей...
    """
    print("\n[ЭТАП 2] Запуск специализированной модели (скрытые signatures)...")

    if model_2 is None:
        print("  ✗ Модель 2 не загружена, пропускаем этап")

        return image_to_draw_on, existing_page_data, annotation_start_index

    if image_for_prediction is None or image_for_prediction.size == 0:
        print("  ✗ Пустое изображение")
        return image_to_draw_on, existing_page_data, annotation_start_index

    image_with_boxes = image_to_draw_on
    h, w, _ = image_with_boxes.shape
    current_annotation_index = annotation_start_index

    results = model_2.predict(
        source=image_for_prediction, conf=MODEL_2_CONFIDENCE, verbose=False
    )
    result = results[0]

    found_count = 0
    duplicate_count = 0

    for box in result.boxes:
        class_id = int(box.cls[0])
        label = MODEL_2_CLASS_MAP.get(class_id, f"signature")

        coords = [int(c) for c in box.xyxy[0]]
        x1, y1, x2, y2 = coords
        bbox_x, bbox_y = x1, y1
        width, height = x2 - x1, y2 - y1

        is_duplicate = False
        for existing_annotation in existing_page_data["annotations"]:
            ann_data = list(existing_annotation.values())[0]
            if ann_data["category"] != "signature":
                continue

            ex = ann_data["bbox"]["x"]
            ey = ann_data["bbox"]["y"]
            ew = ann_data["bbox"]["width"]
            eh = ann_data["bbox"]["height"]

            ix = max(bbox_x, ex)
            iy = max(bbox_y, ey)
            iw = min(bbox_x + width, ex + ew) - ix
            ih = min(bbox_y + height, ey + eh) - iy

            if iw > 0 and ih > 0:
                intersection = iw * ih
                union = (width * height) + (ew * eh) - intersection
                iou = intersection / union

                if iou > 0.3:
                    is_duplicate = True
                    duplicate_count += 1
                    break

        if is_duplicate:
            continue

        annotation = {
            f"annotation_{current_annotation_index}": {
                "category": label,
                "bbox": {"x": bbox_x, "y": bbox_y, "width": width, "height": height},
                "area": width * height,
            }
        }
        existing_page_data["annotations"].append(annotation)

        color = (255, 0, 0)
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            image_with_boxes,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        current_annotation_index += 1
        found_count += 1

    print(f"  ✓ Модель 2 нашла: {found_count} новых подписей")
    if duplicate_count > 0:
        print(f"    → Пропущено дубликатов: {duplicate_count}")

    return image_with_boxes, existing_page_data, current_annotation_index


def detect_signatures_and_stamps_dual(image_data, annotation_start_index: int = 0):
    """
    🎯 ГЛАВНАЯ ФУНКЦИЯ-ОРКЕСТРАТОР

    Последовательно прогоняет изображение через две модели:
    1. Модель 1: Основная детекция (signatures + stamps)
    2. Модель 2: Дополнительная детекция скрытых подписей

    Параметры:
    - image_data: изображение (numpy array, BGR)
    - annotation_start_index: начальный индекс для аннотаций

    Возвращает:
    - image_with_boxes: финальное изображение с рамками от обеих моделей
    - page_data: полный словарь аннотаций
    - next_index: следующий свободный индекс
    """
    print("\n" + "=" * 60)
    print("🎯 ЗАПУСК ДВУХМОДЕЛЬНОГО ПАЙПЛАЙНА")
    print("=" * 60)

    if image_data is None or image_data.size == 0:
        print("✗ ОШИБКА: пустое изображение")
        return image_data, {"annotations": [], "page_size": {}}, annotation_start_index

    h, w, _ = image_data.shape

    page_data = {"annotations": [], "page_size": {"width": int(w), "height": int(h)}}

    current_index = annotation_start_index

    image_result_M1, page_data, current_index = detect_with_model_1(
        image_data, page_data, current_index
    )

    image_result_FINAL, page_data, current_index = detect_with_model_2(
        image_data,
        image_result_M1,
        page_data,
        current_index,
    )

    total_objects = len(page_data["annotations"])
    signatures = sum(
        1
        for ann in page_data["annotations"]
        if list(ann.values())[0]["category"] == "signature"
    )
    stamps = sum(
        1
        for ann in page_data["annotations"]
        if list(ann.values())[0]["category"] == "stamp"
    )

    print("\n" + "=" * 60)
    print("📊 ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 60)
    print(f"Всего найдено объектов: {total_objects}")
    print(f"  → Подписей (signatures): {signatures}")
    print(f"  → Печатей (stamps): {stamps}")
    print("=" * 60 + "\n")

    return image_result_FINAL, page_data, current_index
