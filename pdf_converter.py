import os
from pdf2image import convert_from_path
from pathlib import Path


def batch_convert_pdfs_to_pngs(source_folder, output_base_folder):
    """
    Находит все PDF-файлы в исходной папке, конвертирует каждый из них
    в серию PNG-изображений и сохраняет в отдельную подпапку.

    :param source_folder: Папка, где лежат исходные PDF-файлы.
    :param output_base_folder: Главная папка для сохранения всех результатов.
    """
    # Проверяем, существует ли папка для исходных файлов
    if not os.path.isdir(source_folder):
        print(f"Ошибка: Исходная папка '{source_folder}' не найдена.")
        return

    # Создаем базовую папку для вывода, если ее нет
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)
        print(f"Создана папка для результатов: '{output_base_folder}'")

    # Проходим по всем файлам в исходной папке
    for filename in os.listdir(source_folder):
        # Проверяем, что файл имеет расширение .pdf
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(source_folder, filename)
            print(f"\n--- Начинаю обработку файла: {filename} ---")

            # Создаем имя для подпапки на основе имени PDF-файла (без расширения)
            pdf_name_without_ext = os.path.splitext(filename)[0]
            specific_output_folder = os.path.join(
                output_base_folder, pdf_name_without_ext
            )

            # Создаем эту подпапку, если она еще не существует
            if not os.path.exists(specific_output_folder):
                os.makedirs(specific_output_folder)

            try:
                # Конвертируем PDF в список изображений
                # Для Windows может потребоваться указать путь к poppler:
                # images = convert_from_path(pdf_path, poppler_path=r"C:\путь\к\poppler\bin")
                poppler_path = r"C:\Users\user\Downloads\Release-25.11.0-0\poppler-25.11.0\Library\bin"
                images = convert_from_path(pdf_path, poppler_path=poppler_path)

                # Сохраняем каждую страницу как отдельный PNG-файл
                for i, image in enumerate(images):
                    output_filename = os.path.join(
                        specific_output_folder, f"page_{i + 1}.png"
                    )
                    image.save(output_filename, "PNG")
                    print(f"  Страница {i + 1} сохранена -> {output_filename}")

                print(
                    f"--- Файл '{filename}' успешно сконвертирован. Сохранено {len(images)} страниц. ---"
                )

            except Exception as e:
                print(f"Не удалось обработать файл '{filename}'. Ошибка: {e}")
                print(
                    "Убедитесь, что poppler установлен и прописан в системной переменной PATH."
                )

    print("\nВсе PDF-файлы обработаны.")


# --- Пример использования ---
if __name__ == "__main__":

    current_file_path = Path(__file__).resolve()

    # Получить путь к папке, в которой находится текущий файл
    current_folder = current_file_path.parent

    # Укажите путь к папке с вашими PDF-файлами
    folder_with_pdfs = current_folder / "pdfs"

    # Укажите папку, куда будут сохраняться результаты
    # Внутри нее будут созданы подпапки для каждого PDF
    main_output_folder = "output_png_images"

    batch_convert_pdfs_to_pngs(folder_with_pdfs, main_output_folder)
