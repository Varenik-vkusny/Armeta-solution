# main_pipeline.py
import os
import json
import cv2
import glob
import numpy as np
from pdf2image import convert_from_path
from pathlib import Path
from stamps_and_signs_detector import detect_signatures_and_stamps_dual
from a import find_qrs


INPUT_PDF_DIRECTORY = "test"
OUTPUT_DIRECTORY = "processed_results"
POPPLER_PATH = r"C:\Users\user\Downloads\Release-25.11.0-0\poppler-25.11.0\Library\bin"


def process_single_pdf(
    pdf_path: Path, output_dir: Path, global_annotation_counter: int
):
    """
    Полный конвейер обработки ОДНОГО PDF файла, работающий с изображениями в памяти.
    Возвращает словарь с аннотациями для этого файла и новый счетчик аннотаций.
    """
    pdf_filename = pdf_path.name
    print(f"====== Начинаю обработку файла: {pdf_filename} ======")

    sanitized_stem = pdf_path.stem.strip().rstrip("-. ")

    if not sanitized_stem:
        sanitized_stem = f"unnamed_pdf_{pdf_path.name}"

    pdf_specific_output_dir = output_dir / sanitized_stem
    pdf_specific_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        images = convert_from_path(str(pdf_path), dpi=300, poppler_path=POPPLER_PATH)
    except Exception as e:
        print(f"[ОШИБКА] Не удалось конвертировать PDF '{pdf_filename}': {e}")
        return {}, global_annotation_counter

    pdf_annotations = {}
    annotation_counter = global_annotation_counter

    for i, pil_image in enumerate(images):
        page_num = i + 1
        print(f"\n--- Обработка страницы {page_num} ---")

        image_np_rgb = np.array(pil_image)
        image_np_bgr = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR)

        img_after_model1, page_data_after_model1, annotation_counter = (
            detect_signatures_and_stamps_dual(image_np_bgr, annotation_counter)
        )

        if img_after_model1 is None:
            print(
                f"Модель 1 не смогла обработать страницу {page_num} файла {pdf_filename}"
            )
            continue

        found_qr_codes = find_qrs(img_after_model1)

        # 2. Теперь мы сами обрабатываем результаты и добавляем их в наши данные.
        #    Это дает нам больше контроля.
        final_page_data = page_data_after_model1  # Начинаем с данных, полученных после детекции печатей
        final_image = (
            img_after_model1  # и с изображения, на котором уже нарисованы их рамки
        )

        if found_qr_codes:
            print(
                f"--- Найдено {len(found_qr_codes)} QR-кодов. Добавляю аннотации и рамки... ---"
            )
            for qr in found_qr_codes:
                # Извлекаем координаты из формата [x1, y1, x2, y2]
                x1, y1, x2, y2 = qr["points"]
                w, h = x2 - x1, y2 - y1

                # Получаем расшифрованное значение
                value = qr["value"]

                # Создаем аннотацию в нужном нам формате
                annotation = {
                    f"annotation_{annotation_counter}": {
                        "category": "qr_code",
                        "bbox": {
                            "x": int(x1),
                            "y": int(y1),
                            "width": int(w),
                            "height": int(h),
                        },
                        "area": int(w * h),
                    }
                }
                # Добавляем аннотацию в общий список для этой страницы
                final_page_data["annotations"].append(annotation)

                # Рисуем рамку и текст на итоговом изображении
                cv2.rectangle(final_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(
                    final_image,
                    "QR",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

                # Увеличиваем счетчик для следующей аннотации
                annotation_counter += 1
            else:
                print("--- QR-коды не найдены на этой странице. ---")

        pdf_annotations[f"page_{page_num}"] = final_page_data

        output_image_path = pdf_specific_output_dir / f"page_{page_num}_processed.png"

        try:

            is_success, im_buf_arr = cv2.imencode(".png", final_image)

            if is_success:

                with open(output_image_path, "wb") as f:
                    f.write(im_buf_arr.tobytes())
                print(
                    f"Результат для страницы {page_num} сохранен в {output_image_path}"
                )
            else:
                print(
                    f"[ОШИБКА] Не удалось закодировать изображение для страницы {page_num}."
                )

        except Exception as e:
            print(
                f"[КРИТИЧЕСКАЯ ОШИБКА] Не удалось сохранить файл {output_image_path}. Ошибка: {e}"
            )

    return {pdf_filename: pdf_annotations}, annotation_counter


def process_directory(input_dir: str, output_dir: str):
    """
    Обрабатывает все PDF файлы в указанной входной директории.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    pdf_files = list(input_path.glob("*.pdf"))

    if not pdf_files:
        print(f"В папке '{input_dir}' не найдено PDF файлов для обработки.")
        return

    print(
        f"Найдено {len(pdf_files)} PDF файлов для обработки: {[p.name for p in pdf_files]}"
    )

    all_results = {}
    annotation_counter = 1

    for pdf_path in pdf_files:
        result_for_one_pdf, annotation_counter = process_single_pdf(
            pdf_path, output_path, annotation_counter
        )
        if result_for_one_pdf:
            all_results.update(result_for_one_pdf)

    if all_results:
        output_json_path = output_path / "annotations.json"

        try:
            print("\n====== Пытаюсь сохранить итоговый JSON... ======")
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)

            print(
                f"\n====== ВСЯ ОБРАБОТКА ЗАВЕРШЕНА. Итоговый JSON сохранен в {output_json_path} ======"
            )

        except TypeError as e:

            print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!! КРИТИЧЕСКАЯ ОШИБКА: Не удалось сохранить JSON !!!!!!")
            print(f"!!!!!! Точная ошибка: {e}")
            print(
                "!!!!!! Причина: В данных остались несовместимые с JSON типы (например, numpy.int32)."
            )
            print(
                "!!!!!! Убедитесь, что вы сохранили ИЗМЕНЕНИЯ во всех файлах (qr_detector.py и main_detector_mock.py)."
            )
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            debug_file_path = output_path / "debug_data_dump.txt"
            with open(debug_file_path, "w", encoding="utf-8") as f:
                f.write(str(all_results))
            print(
                f"\nСодержимое данных (для отладки) было сохранено в: {debug_file_path}"
            )

    else:
        print(
            "\n====== ОБРАБОТКА ЗАВЕРШЕНА. Не было найдено данных для сохранения в JSON. ======"
        )


if __name__ == "__main__":
    Path(INPUT_PDF_DIRECTORY).mkdir(exist_ok=True)
    print(f"Проверьте, что ваши PDF файлы находятся в папке: '{INPUT_PDF_DIRECTORY}'")
    print(f"Результаты будут сохранены в папку: '{OUTPUT_DIRECTORY}'")

    process_directory(INPUT_PDF_DIRECTORY, OUTPUT_DIRECTORY)
