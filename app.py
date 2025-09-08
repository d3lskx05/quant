import streamlit as st
from pathlib import Path
import gdown
import os
import psutil
import time
import numpy as np
import torch
import onnxruntime as ort
from transformers import AutoTokenizer

# ========================
# 🔹 Конфигурация страницы
# ========================
st.set_page_config(page_title="USER-BGE-M3 ONNX Test", layout="wide")
st.title("🔍 Тестирование квантизованной модели USER-BGE-M3 (int8)")

MODEL_URL = "https://drive.google.com/uc?id=1lkrvCPIE1wvffIuCSHGtbEz3Epjx5R36"
MODEL_DIR = Path("onnx-user-bge-m3")
MODEL_FILE = MODEL_DIR / "model_quantized.onnx"


# ========================
# 📥 Загрузка модели
# ========================
@st.cache_resource
def load_model():
    MODEL_DIR.mkdir(exist_ok=True)

    if not MODEL_FILE.exists():
        st.write("📥 Скачиваю модель с Google Drive...")
        gdown.download(MODEL_URL, str(MODEL_FILE), quiet=False)

    # Загружаем ONNX модель
    session = ort.InferenceSession(str(MODEL_FILE), providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained("deepvk/USER-BGE-M3")  # оригинальный токенайзер
    return session, tokenizer


session, tokenizer = load_model()
st.success("✅ Модель готова к использованию!")


# ========================
# 📝 Интерфейс
# ========================
texts = st.text_area(
    "Введите текст(ы) для кодирования (по одному на строку):",
    "Это тестовое предложение.\nПример использования модели."
).split("\n")

if st.button("🔍 Запустить инференс"):
    process = psutil.Process(os.getpid())

    # Метрики до
    cpu_before = psutil.cpu_percent(interval=1)
    mem_before = process.memory_info().rss / (1024 ** 2)

    start = time.perf_counter()

    # Токенизация
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="np")

    # Подготовка входов для ONNX
    ort_inputs = {k: v for k, v in inputs.items()}

    # Прогон через модель
    ort_outputs = session.run(None, ort_inputs)
    embeddings = ort_outputs[0].mean(axis=1)

    # Нормализация
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    end = time.perf_counter()

    # Метрики после
    cpu_after = psutil.cpu_percent(interval=1)
    mem_after = process.memory_info().rss / (1024 ** 2)

    # ========================
    # 📊 Результаты
    # ========================
    st.subheader("Результаты инференса")
    st.write(f"⏱️ Время выполнения: **{end - start:.3f} сек**")
    st.write(f"🔢 Размер эмбеддингов: **{embeddings.shape}**")
    st.write("📄 Первые 10 значений первого эмбеддинга:")
    st.json(embeddings[0][:10].tolist())

    st.subheader("📈 Ресурсы")
    col1, col2, col3 = st.columns(3)
    col1.metric("CPU до (%)", f"{cpu_before:.1f}")
    col2.metric("CPU после (%)", f"{cpu_after:.1f}")
    col3.metric("Память (MB)", f"{mem_after:.1f}")

    st.success("Тестирование завершено!")
