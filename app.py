import time
import psutil
import numpy as np
import streamlit as st
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
import onnxruntime as ort
from numpy.linalg import norm
import os
import zipfile
import gdown
import huggingface_hub
from pathlib import Path

# ======================
# Вспомогательные функции
# ======================

def download_model(source, model_id, model_dir):
    """Скачивает модель с GDrive или HF."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if any(model_dir.glob("*")):
        print(f"📂 Модель уже есть в {model_dir}")
        return model_dir

    if source == "gdrive":
        zip_path = f"{model_dir}.zip"
        print(f"📥 Скачиваю модель с Google Drive: {model_id}")
        gdown.download(f"https://drive.google.com/uc?id={model_id}", str(zip_path), quiet=False)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(model_dir)
        os.remove(zip_path)
    elif source == "hf":
        print(f"📥 Скачиваю модель с Hugging Face: {model_id}")
        huggingface_hub.snapshot_download(
            repo_id=model_id,
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )
    else:
        raise ValueError(f"❌ Неизвестный источник: {source}")
    return model_dir


def find_quantized_file(model_dir):
    """Ищет квантованный ONNX-файл в папке модели."""
    model_dir = Path(model_dir)
    quant_files = list(model_dir.rglob("model_quantized.onnx"))
    if quant_files:
        print(f"✅ Найден квантованный файл: {quant_files[0]}")
        return str(quant_files[0])
    print("⚠️ Квантованный файл не найден, загружаем обычную модель")
    return None


@st.cache_resource
def load_model(model_path, model_type="sentence-transformers", quantized=False):
    """Загрузка модели (оригинал или квантованная)."""
    if model_type == "sentence-transformers":
        if quantized:
            quant_file = find_quantized_file(model_path)
            if quant_file:
                return SentenceTransformer(model_path, backend="onnx", model_kwargs={"file_name": Path(quant_file).name})
        return SentenceTransformer(model_path)
    elif model_type == "transformers":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        return model, tokenizer
    elif model_type == "onnx":
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        return ort.InferenceSession(model_path, sess_options=so, providers=providers)


def measure_resources(func, *args, **kwargs):
    """Измерение времени, RAM и CPU."""
    process = psutil.Process()
    start_mem = process.memory_info().rss / 1024**2
    start_cpu = psutil.cpu_percent(interval=None)

    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()

    end_mem = process.memory_info().rss / 1024**2
    end_cpu = psutil.cpu_percent(interval=None)

    return {
        "result": result,
        "time": end_time - start_time,
        "ram_used": end_mem - start_mem,
        "cpu": end_cpu
    }


def cosine_similarity(vec1, vec2):
    """Косинусная схожесть."""
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))


# ======================
# Интерфейс Streamlit
# ======================

st.title("🔍 Тестер квантизованных моделей")

# Ввод текста
input_text = st.text_area("Введите текст:", "Это тестовое предложение.")

# Выбор источника модели
source_choice = st.radio("Выберите источник модели:", ["gdrive", "hf"])

# ID моделей
original_id = st.text_input("ID/Repo оригинальной модели:", "deepvk/USER-BGE-M3")
quantized_id = st.text_input("ID/Repo квантованной модели:", "1lkrvCPIE1wvffIuCSHGtbEz3Epjx5R36")

# Кнопка запуска
if st.button("🔎 Запустить тест"):
    st.write("⏳ Скачиваю и загружаю модели...")

    # Скачиваем модели
    orig_dir = download_model(source_choice, original_id, "original_model")
    quant_dir = download_model(source_choice, quantized_id, "quantized_model")

    # Загружаем модели
    original_model = load_model(str(orig_dir), model_type="sentence-transformers")
    quantized_model = load_model(str(quant_dir), model_type="sentence-transformers", quantized=True)

    # Кодирование
    st.write("⚡ Измеряю производительность оригинальной модели...")
    orig = measure_resources(original_model.encode, [input_text], normalize_embeddings=True)

    st.write("⚡ Измеряю производительность квантованной модели...")
    quant = measure_resources(quantized_model.encode, [input_text], normalize_embeddings=True)

    # Качество
    similarity = cosine_similarity(orig["result"][0], quant["result"][0])

    # Вывод результатов
    st.subheader("📊 Результаты")
    st.write(f"**Время (оригинал):** {orig['time']:.4f} сек")
    st.write(f"**Время (квант):** {quant['time']:.4f} сек")
    st.write(f"**RAM (оригинал):** {orig['ram_used']:.2f} MB")
    st.write(f"**RAM (квант):** {quant['ram_used']:.2f} MB")
    st.write(f"**CPU нагрузка:** {quant['cpu']}%")
    st.write(f"**Косинусная схожесть:** {similarity:.4f}")

    # Графики
    st.bar_chart({
        "Время (сек)": [orig["time"], quant["time"]],
        "RAM (MB)": [orig["ram_used"], quant["ram_used"]]
    })
