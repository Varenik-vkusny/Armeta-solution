# qr_detector.py
import cv2
from ultralytics import YOLO
from pyzbar.pyzbar import decode

# --- ГЛАВНОЕ ИЗМЕНЕНИЕ: ЗАГРУЖАЕМ НЕЙРОСЕТЬ YOLOv8 ---
# Указываем путь к файлу с весами, который вы скачали.
# Модель загрузится один раз при первом вызове.
try:
    model = YOLO("yolo11n.pt")
    print("--- Модель YOLOv8 для детекции QR-кодов успешно загружена. ---")
except Exception as e:
    print(
        f"--- [КРИТИЧЕСКАЯ ОШИБКА] Не удалось загрузить модель 'yolo11n.pt'. Убедитесь, что файл находится в папке проекта. Ошибка: {e} ---"
    )
    model = None


def add_qr_code_detections(
    input_image_np, existing_page_data, annotation_start_index: int
):
    """
    Финальная версия детектора на YOLOv8 с порогом уверенности и масштабированием.
    """
    print("--- (Модель 2) Поиск QR-кодов (YOLOv8 + постобработка)...")

    if model is None:
        return input_image_np, existing_page_data, annotation_start_index

    # --- НАСТРОЙКИ, КОТОРЫЕ МОЖНО МЕНЯТЬ ---
    # 1. Порог уверенности: отсекаем все, что ниже этого значения.
    #    Увеличивайте, чтобы убрать ложные срабатывания. Уменьшайте, если модель пропускает настоящие QR.
    CONFIDENCE_THRESHOLD = 0.5  # Начинаем с 50%

    # 2. Масштабирование: увеличиваем изображение, чтобы найти маленькие QR-коды.
    SCALE_FACTOR = 2.0

    # --- ШАГ 1: МАСШТАБИРОВАНИЕ ---
    h, w, _ = input_image_np.shape
    upscaled_image = cv2.resize(
        input_image_np,
        (int(w * SCALE_FACTOR), int(h * SCALE_FACTOR)),
        interpolation=cv2.INTER_CUBIC,
    )
    print(f"--- (Модель 2) Изображение увеличено в {SCALE_FACTOR} раза.")

    image_with_boxes = input_image_np.copy()
    current_annotation_index = annotation_start_index

    # --- ШАГ 2: ДЕТЕКЦИЯ С ПОРОГОМ УВЕРЕННОСТИ ---
    # Передаем в модель параметр `conf`, чтобы она сама отфильтровала слабые детекции
    results = model(upscaled_image, conf=CONFIDENCE_THRESHOLD, verbose=False)

    found_boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(found_boxes) > 0:
        print(
            f"--- (Модель 2) YOLOv8 нашла {len(found_boxes)} QR-кодов с уверенностью > {CONFIDENCE_THRESHOLD*100}%."
        )

        for box in found_boxes:
            # Координаты пришли с увеличенного изображения, масштабируем их обратно
            scaled_box = [int(coord / SCALE_FACTOR) for coord in box]
            x1, y1, x2, y2 = scaled_box

            # --- ШАГ 3: ДЕКОДИРОВАНИЕ ---
            # Вырезаем найденный QR-код из ОРИГИНАЛЬНОГО изображения для чистоты
            qr_crop = input_image_np[y1:y2, x1:x2]

            decoded_data = ""
            try:
                decoded_objects = decode(qr_crop)
                if decoded_objects:
                    decoded_data = decoded_objects[0].data.decode("utf-8")
                else:
                    decoded_data = "decoding_failed"
            except Exception:
                decoded_data = "decoding_error"

            # Формируем аннотацию
            w, h = x2 - x1, y2 - y1
            annotation = {
                f"annotation_{current_annotation_index}": {
                    "category": "qr_code",
                    "bbox": {"x": x1, "y": y1, "width": w, "height": h},
                    "area": w * h,
                    "decoded_data": decoded_data,
                }
            }
            existing_page_data["annotations"].append(annotation)

            # Рисуем рамку на итоговом изображении
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                image_with_boxes,
                "QR",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            current_annotation_index += 1
    else:
        print(
            f"--- (Модель 2) YOLOv8 не нашла QR-кодов с уверенностью > {CONFIDENCE_THRESHOLD*100}%."
        )

    return image_with_boxes, existing_page_data, current_annotation_index
