# main_detector_mock.py
import cv2
import numpy as np


def detect_signatures_and_stamps_mock(image_data, annotation_start_index: int):
    """
    MOCK FUNCTION: Имитирует работу модели по обнаружению подписей и печатей.
    Теперь принимает на вход не путь к файлу, а сам объект изображения (Numpy array).

    :param image_data: Изображение для обработки в формате OpenCV (Numpy array, BGR).
    :param annotation_start_index: Начальный индекс для нумерации аннотаций.
    :return: Кортеж (
        - изображение с нарисованными рамками (numpy array),
        - словарь с данными аннотаций для этой страницы,
        - следующий свободный индекс для аннотации
    )
    """
    print(f"--- (Модель 1) Имитация поиска подписей и печатей...")
    if image_data is None:
        print("--- (Модель 1) Ошибка: на вход получено пустое изображение.")
        return None, {}, annotation_start_index

    h, w, _ = image_data.shape
    image_with_boxes = image_data.copy()

    # --- ИМИТАЦИЯ НАЙДЕННЫХ ОБЪЕКТОВ ---
    fake_detections = [
        {"box": (100, 200, 250, 80), "label": "signature"},
        {"box": (500, 600, 200, 200), "label": "stamp"},
    ]

    # --- ФОРМИРОВАНИЕ JSON ДЛЯ ЭТОЙ СТРАНИЦЫ ---
    # --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ ЗДЕСЬ! ---
    # Преобразуем w и h (которые являются numpy.int64) в стандартный int.
    page_data = {"annotations": [], "page_size": {"width": int(w), "height": int(h)}}

    current_annotation_index = annotation_start_index
    for det in fake_detections:
        (x, y, width, height) = det["box"]
        label = det["label"]

        # Здесь x, y, width, height - это уже обычные int, их преобразовывать не нужно.
        annotation = {
            f"annotation_{current_annotation_index}": {
                "category": label,
                "bbox": {"x": x, "y": y, "width": width, "height": height},
                "area": width * height,
            }
        }
        page_data["annotations"].append(annotation)

        color = (255, 0, 0) if label == "signature" else (0, 0, 255)
        cv2.rectangle(image_with_boxes, (x, y), (x + width, y + height), color, 3)
        cv2.putText(
            image_with_boxes,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
        )

        current_annotation_index += 1

    print(f"--- (Модель 1) Найдено {len(fake_detections)} объектов.")
    return image_with_boxes, page_data, current_annotation_index
