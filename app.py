import streamlit as st
from pathlib import Path
import gdown
import os
import psutil
import time
import torch
import numpy as np
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction

# ========================
# 🔹 Конфигурация страницы
# ========================
st.set_page_config(
    page_title="USER-BGE-M3 Quantized ONNX Test",
    layout="wide"
)
st.title("🔍 Тестирование квантизованной модели USER-BGE-M3 (int8)")

MODEL_URL = "https://drive.google.com/uc?id=1lkrvCPIE1wvffIuCSHGtbEz3Epjx5R36"
MODEL_DIR = Path("onnx-user-bge-m3")
MODEL_FILE = MODEL_DIR / "model_quantized.onnx"


# ========================
# 📥 Функция загрузки модели
# ========================
@st.cache_resource
def load_model_and_tokenizer():
    MODEL_DIR.mkdir(exist_ok=True)

    if not MODEL_FILE.exists():
        st.write("📥 Скачиваю модель с Google Drive...")
        gdown.download(MODEL_URL, str(MODEL_FILE), quiet=False)

    st.write("⚙️ Загружаю модель...")
    model = ORTModelForFeatureExtraction.from_pretrained(MODEL_DIR, file_name=MODEL_FILE.name)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    return model, tokenizer


# ========================
# 🚀 Загрузка модели
# ========================
model, tokenizer = load_model_and_tokenizer()
st.success("✅ Модель готова к использованию!")


# ========================
# 📝 Интерфейс ввода текста
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

    start_time = time.perf_counter()

    # Токенизация
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Прогон через модель
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    # Нормализация
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    end_time = time.perf_counter()

    # Метрики после
    cpu_after = psutil.cpu_percent(interval=1)
    mem_after = process.memory_info().rss / (1024 ** 2)

    # ========================
    # 📊 Результаты
    # ========================
    st.subheader("Результаты инференса")
    st.write(f"⏱️ Время выполнения: **{end_time - start_time:.3f} сек**")
    st.write(f"🔢 Размер эмбеддингов: **{embeddings.shape}**")
    st.write("📄 Первые 10 значений первого эмбеддинга:")
    st.json(embeddings[0][:10].tolist())

    st.subheader("📈 Ресурсы")
    col1, col2, col3 = st.columns(3)
    col1.metric("CPU до (%)", f"{cpu_before:.1f}")
    col2.metric("CPU после (%)", f"{cpu_after:.1f}")
    col3.metric("Память (MB)", f"{mem_after:.1f}")

    st.success("Тестирование завершено!")
