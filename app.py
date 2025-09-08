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

# ======================
# Вспомогательные функции
# ======================

@st.cache_resource
def load_model(model_path, model_type="sentence-transformers", quantized=False):
    """Загрузка модели (оригинал или квантованная)."""
    if model_type == "sentence-transformers":
        if quantized:
            quant_file = os.path.join(model_path, "model_quantized.onnx")
            if os.path.exists(quant_file):
                return SentenceTransformer(model_path, backend="onnx", model_kwargs={"file_name": "model_quantized.onnx"})
            else:
                return SentenceTransformer(model_path)
        else:
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

    # Приводим результат к вектору
    result = np.array(result)
    if result.ndim > 1:
        result = result.mean(axis=0)

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

# Пути к моделям
original_model_path = st.text_input("Путь к оригинальной модели или HF repo_id:", "deepvk/USER-BGE-M3")
quantized_model_path = st.text_input("Путь к квантованной модели:", "onnx-user-bge-m3")

# Кнопка запуска
if st.button("🔎 Запустить тест"):
    st.write("⏳ Загружаю модели...")

    original_model = load_model(original_model_path, model_type="sentence-transformers")
    quantized_model = load_model(quantized_model_path, model_type="sentence-transformers", quantized=True)

    # Кодирование
    st.write("⚡ Измеряю производительность оригинальной модели...")
    orig = measure_resources(original_model.encode, [input_text], normalize_embeddings=True)

    st.write("⚡ Измеряю производительность квантованной модели...")
    quant = measure_resources(quantized_model.encode, [input_text], normalize_embeddings=True)

    # Качество
    similarity = cosine_similarity(orig["result"], quant["result"])

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
