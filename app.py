import time
import psutil
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer

# =============================
# 🚀 Настройки
# =============================
MODEL_PATH = "onnx-user-bge-m3"  # папка с моделью

# =============================
# 📥 Загрузка модели
# =============================
@st.cache_resource
def load_model():
    return SentenceTransformer(
        MODEL_PATH,
        backend="onnx",
        model_kwargs={"file_name": "model_quantized.onnx"}
    )

model = load_model()
st.success("Модель загружена!")

# =============================
# 📝 Интерфейс
# =============================
st.title("Тест квантизованной модели USER-BGE-M3 (INT8)")
user_input = st.text_area("Введите текст для кодирования:", "Это тестовое предложение.")

if st.button("Получить эмбеддинг"):
    if user_input.strip():
        # =============================
        # 📊 Замер ресурсов
        # =============================
        process = psutil.Process()
        start_mem = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()

        # Инференс
        embedding = model.encode([user_input], normalize_embeddings=True)
        
        elapsed_time = time.time() - start_time
        end_mem = process.memory_info().rss / 1024 / 1024  # MB

        # =============================
        # 📈 Вывод результатов
        # =============================
        st.write(f"⏱ Время инференса: **{elapsed_time:.4f} сек**")
        st.write(f"💾 Использование памяти: **{end_mem - start_mem:.2f} MB**")
        st.write(f"📐 Размер эмбеддинга: {embedding.shape}")

        st.json({"Первые 10 значений эмбеддинга": embedding[0][:10].tolist()})
    else:
        st.warning("Введите текст для кодирования!")

# =============================
# 🖥️ Доп. метрики
# =============================
cpu_usage = psutil.cpu_percent(interval=0.5)
mem_usage = psutil.virtual_memory().percent

st.metric("CPU загрузка", f"{cpu_usage}%")
st.metric("RAM загрузка", f"{mem_usage}%")
