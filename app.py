import time
import psutil
import streamlit as st
import numpy as np
import gdown
import zipfile
from pathlib import Path
from sentence_transformers import SentenceTransformer

# =============================
# 🔗 Ссылка на модель в GDrive
# =============================
MODEL_ID = "1lkrvCPIE1wvffIuCSHGtbEz3Epjx5R36"  # <-- сюда вставь ID
MODEL_ZIP_URL = f"https://drive.google.com/uc?id={MODEL_ID}"
MODEL_DIR = Path("onnx-user-bge-m3")

# =============================
# 📥 Функция загрузки модели
# =============================
@st.cache_resource
def download_and_load_model():
    if not MODEL_DIR.exists():
        st.write("📥 Скачиваю модель с Google Drive...")
        gdown.download(MODEL_ZIP_URL, "model.zip", quiet=False)
        with zipfile.ZipFile("model.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
        st.write("✅ Модель распакована!")

    st.write("🚀 Загружаю модель...")
    model = SentenceTransformer(
        str(MODEL_DIR),
        backend="onnx",
        model_kwargs={"file_name": "model_quantized.onnx"}
    )
    return model

model = download_and_load_model()
st.success("Модель готова к использованию!")

# =============================
# 📝 Интерфейс
# =============================
st.title("Тест квантизованной модели USER-BGE-M3 (INT8)")
user_input = st.text_area("Введите текст для кодирования:", "Это тестовое предложение.")

if st.button("Получить эмбеддинг"):
    if user_input.strip():
        process = psutil.Process()
        start_mem = process.memory_info().rss / 1024 / 1024
        start_time = time.time()

        embedding = model.encode([user_input], normalize_embeddings=True)

        elapsed_time = time.time() - start_time
        end_mem = process.memory_info().rss / 1024 / 1024

        st.write(f"⏱ Время инференса: **{elapsed_time:.4f} сек**")
        st.write(f"💾 Использование памяти: **{end_mem - start_mem:.2f} MB**")
        st.write(f"📐 Размер эмбеддинга: {embedding.shape}")
        st.json({"Первые 10 значений эмбеддинга": embedding[0][:10].tolist()})
    else:
        st.warning("Введите текст!")

cpu_usage = psutil.cpu_percent(interval=0.5)
mem_usage = psutil.virtual_memory().percent

st.metric("CPU загрузка", f"{cpu_usage}%")
st.metric("RAM загрузка", f"{mem_usage}%")
