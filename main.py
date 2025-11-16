import os
import shutil
import tempfile
import logging
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from contextlib import asynccontextmanager
from orkestrator_or import process_directory
from starlette.concurrency import run_in_threadpool

logging.basicConfig(level=logging.INFO)


def cleanup_files(*paths: str):
    """Удаляет временные папки и файлы."""
    for path in paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
                print(f"Удалена временная папка: {path}")
            else:
                os.remove(path)
                print(f"Удален временный файл: {path}")


@asynccontextmanager
async def lifespan(app: FastAPI):

    logging.info("Приложение запускается...")
    yield
    logging.info("Приложение останавливается...")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешаем все источники
    allow_credentials=True,
    allow_methods=["*"],  # Разрешаем все методы (GET, POST и т.д.)
    allow_headers=["*"],  # Разрешаем все заголовки
)

# --- ДОБАВЛЕНО: МОНТИРОВАНИЕ СТАТИЧЕСКИХ ФАЙЛОВ ---
# FastAPI будет автоматически отдавать файлы из папки 'static' по пути '/static'
# Например, ваш файл 'static/images/v5_46.png' будет доступен по URL '/static/images/v5_46.png'
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Читаем ваш HTML-файл и возвращаем его как ответ
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.post(
    "/process_batch_zip/",
    summary="Обработать PDF-батч и вернуть ZIP с диска",
    response_class=FileResponse,
)
async def process_batch_zip_endpoint(
    background_tasks: BackgroundTasks,  # <--- ЭТО ВАЖНО ДЛЯ ОЧИСТКИ
    files: List[UploadFile] = File(..., description="Список PDF файлов"),
):
    # 1. Создаем временные папки
    # tempfile.TemporaryDirectory гарантирует удаление при завершении, но
    # в FastAPI лучше контролировать самому с помощью shutil.rmtree
    input_temp_dir = tempfile.mkdtemp()
    output_temp_dir = tempfile.mkdtemp()

    zip_filename = f"batch_results_{os.path.basename(input_temp_dir)}.zip"
    zip_path = os.path.join(tempfile.gettempdir(), zip_filename)

    try:
        # 2. Сохраняем загруженные файлы в INPUT-папку
        for file in files:
            if not file.filename.lower().endswith(".pdf"):
                raise HTTPException(
                    status_code=400, detail=f"Файл {file.filename} не является PDF."
                )

            file_path = os.path.join(input_temp_dir, file.filename)

            # Читаем чанки и записываем на диск
            with open(file_path, "wb") as buffer:
                while content := await file.read(1024 * 1024):  # Читаем по 1МБ
                    buffer.write(content)

        await run_in_threadpool(process_directory, input_temp_dir, output_temp_dir)

        # ===== ИЗМЕНЕНИЕ №3: СОЗДАНИЕ АРХИВА ТОЖЕ ЗАПУСКАЕМ В ПОТОКЕ =====
        await run_in_threadpool(
            shutil.make_archive,
            base_name=os.path.splitext(zip_path)[0],
            format="zip",
            root_dir=output_temp_dir,
        )

    except Exception as e:
        # В случае любой ошибки нужно почистить папки и выдать 500
        cleanup_files(input_temp_dir, output_temp_dir, zip_path)
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

    # 5. Возвращаем ZIP и планируем очистку

    # Регистрируем функцию очистки, которая сработает ПОСЛЕ отправки ответа клиенту
    # Очищаем: входную папку, выходную папку и сам созданный ZIP-файл
    background_tasks.add_task(cleanup_files, input_temp_dir, output_temp_dir, zip_path)

    return FileResponse(
        zip_path, media_type="application/zip", filename="processing_results.zip"
    )
