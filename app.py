import streamlit as st
from pathlib import Path
import gdown
import os
import psutil
import time
from sentence_transformers import SentenceTransformer

# ========================
# 🔹 Конфигурация страницы
# ========================
st.set_page_config(
    page_title="USER-BGE-M3 Quantized Test",
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
def load_model():
    MODEL_DIR.mkdir(exist_ok=True)

    # Скачиваем модель, если её нет
    if not MODEL_FILE.exists():
        st.write("📥 Скачиваю модель с Google Drive...")
        gdown.download(MODEL_URL, str(MODEL_FILE), quiet=False)

    # Загружаем SentenceTransformer
    st.write("⚙️ Загружаю модель...")
    model = SentenceTransformer(
        str(MODEL_DIR),
        backend="onnx",
        model_kwargs={"file_name": MODEL_FILE.name}
    )
    return model


# ========================
# 🚀 Загрузка модели
# ========================
model = load_model()
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
    embeddings = model.encode(texts, normalize_embeddings=True)
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

