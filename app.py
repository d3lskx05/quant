# app.py
import os
import time
import traceback
import numpy as np
import psutil
import streamlit as st
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from quant_model import QuantModel  # <-- твой класс из quant_model.py

st.set_page_config(page_title="Quantized model tester", layout="wide")


# -----------------------
# Encoding helpers
# -----------------------
def to_vector(embs):
    """Превращает батч эмбеддингов в один вектор (усреднение)."""
    arr = np.array(embs)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr[0]
    return arr.mean(axis=0)


def cosine_similarity(vec1, vec2):
    return float(np.dot(vec1, vec2) / ((norm(vec1) * norm(vec2)) + 1e-12))


# -----------------------
# UI
# -----------------------
st.title("🔍 Сравнение оригинальной и квантованной моделей")

st.markdown("Оригинал грузим через `SentenceTransformer`, квантованную через `QuantModel` (onnxruntime).")

col1, col2 = st.columns(2)
with col1:
    st.header("Оригинальная модель")
    orig_id = st.text_input("HF repo ID или локальный путь", "deepvk/USER-BGE-M3")

with col2:
    st.header("Квантованная модель")
    quant_source = st.selectbox("Источник", ["gdrive", "hf", "local"], index=1)
    quant_id = st.text_input("ID/Repo/Path", "1lkrvCPIE1wvffIuCSHGtbEz3Epjx5R36")
    quant_dir = st.text_input("Папка для кванта (локальная)", "onnx-user-bge-m3")
    tokenizer_name = st.text_input("Tokenizer name (опционально)", "")

st.markdown("---")
input_text = st.text_area("Тексты для теста (по одной строке)", "Это тестовое предложение.\nПример второй строки.")
texts = [t.strip() for t in input_text.split("\n") if t.strip()]

batch_size = st.slider("Количество повторов для throughput-теста", 1, 128, 8)
run_button = st.button("🚀 Запустить тест")


# -----------------------
# Logic
# -----------------------
if run_button:
    try:
        proc = psutil.Process()

        # --- Загружаем оригинал через SentenceTransformer
        with st.spinner("Загрузка оригинальной модели..."):
            orig_model = SentenceTransformer(orig_id)
            st.success(f"✅ Оригинал загружен из {orig_id}")

        # --- Загружаем квантованную модель через QuantModel
        with st.spinner("Загрузка квантованной модели..."):
            quant_model = QuantModel(
                model_id=quant_id,
                source=quant_source,
                model_dir=quant_dir,
                tokenizer_name=tokenizer_name if tokenizer_name else None
            )
            st.success(f"✅ Квантованная модель загружена ({quant_model.model_path})")

        # Подготовим тексты
        texts_for_run = (texts * batch_size)[:max(len(texts), 1)]

        # --- Прогон оригинала
        t0 = time.perf_counter()
        orig_embs = orig_model.encode(texts_for_run, normalize_embeddings=True)
        t1 = time.perf_counter()
        orig_time = t1 - t0
        mem_after_orig = proc.memory_info().rss / 1024 ** 2

        # --- Прогон кванта
        t0 = time.perf_counter()
        quant_embs = quant_model.encode(texts_for_run, normalize=True)
        t1 = time.perf_counter()
        quant_time = t1 - t0
        mem_after_quant = proc.memory_info().rss / 1024 ** 2

        # --- Косинусная схожесть
        v_orig = to_vector(orig_embs)
        v_quant = to_vector(quant_embs)

        if v_orig.shape != v_quant.shape:
            st.warning(f"Размерности различаются: {v_orig.shape} vs {v_quant.shape}, усечем до min.")
            m = min(v_orig.size, v_quant.size)
            v_orig = v_orig[:m]
            v_quant = v_quant[:m]

        cos = cosine_similarity(v_orig, v_quant)

        # --- Результаты
        st.subheader("📊 Результаты")
        st.metric("Latency Original (s)", f"{orig_time:.4f}")
        st.metric("Latency Quant (s)", f"{quant_time:.4f}")
        st.metric("Cosine Similarity", f"{cos:.4f}")
        st.write(f"Memory after original: {mem_after_orig:.1f} MB")
        st.write(f"Memory after quant: {mem_after_quant:.1f} MB")

        st.bar_chart({
            "Latency (s)": [orig_time, quant_time],
            "Memory (MB)": [mem_after_orig, mem_after_quant]
        })

    except Exception as e:
        st.error(f"Ошибка: {e}")
        st.text(traceback.format_exc())
