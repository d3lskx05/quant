import time
import psutil
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
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

@st.cache_resource
def download_model(source, model_id, model_dir):
    """Скачивает модель с GDrive или HF (с кэшированием)."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if any(model_dir.glob("*")):
        return str(model_dir)

    if source == "gdrive":
        zip_path = f"{model_dir}.zip"
        gdown.download(f"https://drive.google.com/uc?id={model_id}", str(zip_path), quiet=False)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(model_dir)
        os.remove(zip_path)
    elif source == "hf":
        huggingface_hub.snapshot_download(
            repo_id=model_id,
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )
    else:
        raise ValueError(f"❌ Неизвестный источник: {source}")
    return str(model_dir)


def find_quantized_file(model_dir):
    """Ищет квантованный ONNX-файл в папке модели."""
    model_dir = Path(model_dir)
    quant_files = list(model_dir.rglob("model_quantized.onnx"))
    return str(quant_files[0]) if quant_files else None


@st.cache_resource
def load_model(model_path, quantized=False):
    """Загрузка модели: SentenceTransformers или ONNX напрямую."""
    if quantized:
        quant_file = find_quantized_file(model_path)
        if quant_file:
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            return ort.InferenceSession(quant_file, sess_options=so, providers=["CPUExecutionProvider"])
        else:
            st.warning("⚠️ Квантованный файл не найден, загружаем обычную модель")
    return SentenceTransformer(model_path)


def encode_onnx(session, tokenizer, text):
    """Кодирование текста через чистый ONNX Runtime."""
    import torch
    from transformers import AutoTokenizer

    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
    ort_inputs = {k: v for k, v in inputs.items()}
    ort_outs = session.run(None, ort_inputs)
    return ort_outs[0]


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

# Выбор источников
orig_source = st.radio("Источник оригинальной модели:", ["gdrive", "hf"], index=1, key="orig")
quant_source = st.radio("Источник квантованной модели:", ["gdrive", "hf"], index=0, key="quant")

# ID моделей
original_id = st.text_input("ID/Repo оригинальной модели:", "deepvk/USER-BGE-M3")
quantized_id = st.text_input("ID/Repo квантованной модели:", "1lkrvCPIE1wvffIuCSHGtbEz3Epjx5R36")

# Кнопка запуска
if st.button("🔎 Запустить тест"):
    st.write("⏳ Скачиваю и загружаю модели...")

    orig_dir = download_model(orig_source, original_id, "original_model")
    quant_dir = download_model(quant_source, quantized_id, "quantized_model")

    original_model = load_model(orig_dir, quantized=False)
    quantized_model = load_model(quant_dir, quantized=True)

    st.write("⚡ Измеряю производительность оригинальной модели...")
    orig = measure_resources(original_model.encode, [input_text], normalize_embeddings=True)

    st.write("⚡ Измеряю производительность квантованной модели...")
    if isinstance(quantized_model, ort.InferenceSession):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(orig_dir)
        quant = measure_resources(encode_onnx, quantized_model, tokenizer, [input_text])
    else:
        quant = measure_resources(quantized_model.encode, [input_text], normalize_embeddings=True)

    similarity = cosine_similarity(orig["result"][0], quant["result"][0])

    st.subheader("📊 Результаты")
    st.write(f"**Время (оригинал):** {orig['time']:.4f} сек")
    st.write(f"**Время (квант):** {quant['time']:.4f} сек")
    st.write(f"**RAM (оригинал):** {orig['ram_used']:.2f} MB")
    st.write(f"**RAM (квант):** {quant['ram_used']:.2f} MB")
    st.write(f"**CPU нагрузка:** {quant['cpu']}%")
    st.write(f"**Косинусная схожесть:** {similarity:.4f}")

    st.bar_chart({
        "Время (сек)": [orig["time"], quant["time"]],
        "RAM (MB)": [orig["ram_used"], quant["ram_used"]]
    })
