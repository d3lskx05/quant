import os
import zipfile
import time
import traceback
from pathlib import Path
from functools import lru_cache
from typing import Optional

import gdown
import huggingface_hub
import numpy as np
import psutil
import pandas as pd
import streamlit as st
import onnxruntime as ort
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

st.set_page_config(page_title="Quantized model tester", layout="wide")

# ============================================================
# 🔥 QuantModel
# ============================================================
class QuantModel:
    """Универсальный загрузчик квантизированных ONNX моделей."""
    def __init__(self, model_id: str, source: str = "gdrive",
                 model_dir: str = "onnx_model", tokenizer_name: Optional[str] = None,
                 force_download: bool = False):
        self.model_id = model_id
        self.source = source
        self.model_dir = Path(model_dir)
        self.tokenizer_name = tokenizer_name
        self.force_download = force_download
        self.model_path = None

        self._ensure_model()
        self.session = self._load_session()
        self.tokenizer = self._load_tokenizer()

    def _ensure_model(self):
        os.makedirs(self.model_dir, exist_ok=True)
        need_download = self.force_download or not any(self.model_dir.glob("*.onnx"))

        if need_download:
            if self.source == "gdrive":
                zip_path = f"{self.model_dir}.zip"
                gdown.download(f"https://drive.google.com/uc?id={self.model_id}", zip_path, quiet=False)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(self.model_dir)
                os.remove(zip_path)
            elif self.source == "hf":
                huggingface_hub.snapshot_download(
                    repo_id=self.model_id,
                    local_dir=self.model_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True
                )

        onnx_files = list(self.model_dir.rglob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"❌ Нет .onnx модели в {self.model_dir}")
        self.model_path = onnx_files[0]

    def _load_session(self):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        try:
            if ort.get_device() == "GPU":
                providers.insert(0, "CUDAExecutionProvider")
        except Exception:
            pass
        return ort.InferenceSession(str(self.model_path), sess_options=so, providers=providers)

    def _load_tokenizer(self):
        if self.tokenizer_name:
            return AutoTokenizer.from_pretrained(self.tokenizer_name)
        try:
            return AutoTokenizer.from_pretrained(str(self.model_dir))
        except Exception:
            return AutoTokenizer.from_pretrained("deepvk/USER-BGE-M3")

    @lru_cache(maxsize=1024)
    def _encode_cached(self, text: str, normalize: bool = True):
        inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors="np")
        ort_inputs = {k: v for k, v in inputs.items()}
        embeddings = self.session.run(None, ort_inputs)[0]
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
            embeddings = embeddings / norms
        return embeddings[0]

    def encode(self, texts, normalize=True):
        if isinstance(texts, str):
            texts = [texts]
        return np.array([self._encode_cached(t, normalize) for t in texts])


# ============================================================
# 🔧 Helpers
# ============================================================
def to_vector(embs):
    arr = np.array(embs)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr[0]
    return arr.mean(axis=0)


def cosine_similarity(vec1, vec2):
    return float(np.dot(vec1, vec2) / ((norm(vec1) * norm(vec2)) + 1e-12))


# ============================================================
# 🎛️ UI
# ============================================================
st.title("🔍 Тестирование моделей: Оригинал vs Квант")

# Кнопка сброса сессии
if st.button("♻️ Сбросить сессию"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

mode = st.radio("Выберите режим:", ["Оригинальная модель", "Квантованная модель", "Сравнение обеих"])

input_text = st.text_area("Тексты для теста (по одной строке)", "Это тестовое предложение.\nПример второй строки.")
texts = [t.strip() for t in input_text.split("\n") if t.strip()]

batch_size = st.slider("Количество повторов для throughput-теста", 1, 128, 8)
force_download = st.checkbox("♻️ Перекачать модель заново", False)

# Параметры для моделей
orig_id = st.text_input("HF repo ID для оригинала", "deepvk/USER-BGE-M3")
quant_source = st.selectbox("Источник квантованной", ["gdrive", "hf", "local"], index=1)
quant_id = st.text_input("ID/Repo/Path квантованной", "1lkrvCPIE1wvffIuCSHGtbEz3Epjx5R36")
quant_dir = st.text_input("Папка для кванта", "onnx-user-bge-m3")
tokenizer_name = st.text_input("Tokenizer name", "")

run_button = st.button("🚀 Запустить тест")

# ============================================================
# 🚀 Запуск теста
# ============================================================
if run_button:
    try:
        proc = psutil.Process()
        texts_for_run = (texts * batch_size)[:max(len(texts), 1)]
        results = []

        def run_original():
            with st.spinner("Загрузка оригинальной модели..."):
                model = SentenceTransformer(orig_id)
            t0 = time.perf_counter()
            embs = model.encode(texts_for_run, normalize_embeddings=True)
            t1 = time.perf_counter()
            mem = proc.memory_info().rss / 1024 ** 2
            return embs, t1 - t0, mem

        def run_quant():
            with st.spinner("Загрузка квантованной модели..."):
                model = QuantModel(
                    model_id=quant_id,
                    source=quant_source,
                    model_dir=quant_dir,
                    tokenizer_name=tokenizer_name if tokenizer_name else None,
                    force_download=force_download
                )
            t0 = time.perf_counter()
            embs = model.encode(texts_for_run, normalize=True)
            t1 = time.perf_counter()
            mem = proc.memory_info().rss / 1024 ** 2
            return embs, t1 - t0, mem

        if mode == "Оригинальная модель":
            e, t, m = run_original()
            results.append({"Model": "Original", "Batch": batch_size, "Latency (s)": t, "Memory (MB)": m})

        elif mode == "Квантованная модель":
            e, t, m = run_quant()
            results.append({"Model": "Quantized", "Batch": batch_size, "Latency (s)": t, "Memory (MB)": m})

        else:  # Сравнение обеих
            orig_e, orig_t, _ = run_original()
            quant_e, quant_t, _ = run_quant()
            cos = cosine_similarity(to_vector(orig_e), to_vector(quant_e))
            results.append({"Model": "Original", "Batch": batch_size, "Latency (s)": orig_t})
            results.append({"Model": "Quantized", "Batch": batch_size, "Latency (s)": quant_t, "Cosine Sim": cos})

        df = pd.DataFrame(results)
        st.subheader("📊 Результаты")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Скачать результаты (CSV)",
            data=csv,
            file_name="metrics.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Ошибка: {e}")
        st.text(traceback.format_exc())
