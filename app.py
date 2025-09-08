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
import streamlit as st
import onnxruntime as ort
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

st.set_page_config(page_title="Quantized model tester", layout="wide")

# ============================================================
# 🔥 QuantModel (встроен сюда, с кэшированием и force_download)
# ============================================================
class QuantModel:
    """
    Универсальный загрузчик квантизированных ONNX моделей.
    Источники: Google Drive (gdrive), Hugging Face Hub (hf), локальная (local).
    Поддержка кэширования и флага force_download.
    """

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
        """Скачивание и распаковка модели, с учётом кэширования."""
        os.makedirs(self.model_dir, exist_ok=True)

        need_download = self.force_download or not any(self.model_dir.glob("*.onnx"))

        if need_download:
            if self.source == "gdrive":
                zip_path = f"{self.model_dir}.zip"
                print(f"📥 Скачиваю модель с Google Drive: {self.model_id}")
                gdown.download(f"https://drive.google.com/uc?id={self.model_id}", zip_path, quiet=False)
                print(f"📦 Распаковка {zip_path}...")
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(self.model_dir)
                os.remove(zip_path)

            elif self.source == "hf":
                print(f"📥 Скачиваю модель с Hugging Face: {self.model_id}")
                huggingface_hub.snapshot_download(
                    repo_id=self.model_id,
                    local_dir=self.model_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True
                )

            elif self.source == "local":
                print(f"📂 Использую локальную модель: {self.model_dir}")
            else:
                raise ValueError(f"❌ Неизвестный источник: {self.source}")
        else:
            print(f"✅ Использую закэшированную модель из {self.model_dir}")

        onnx_files = list(self.model_dir.rglob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"❌ В {self.model_dir} не найден .onnx файл!")
        self.model_path = onnx_files[0]
        print(f"✅ Найден ONNX файл: {self.model_path}")

    def _load_session(self):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        try:
            if ort.get_device() == "GPU":
                providers.insert(0, "CUDAExecutionProvider")
        except Exception:
            pass
        print(f"🚀 Загружаю модель на провайдерах: {providers}")
        return ort.InferenceSession(str(self.model_path), sess_options=so, providers=providers)

    def _load_tokenizer(self):
        if self.tokenizer_name:
            return AutoTokenizer.from_pretrained(self.tokenizer_name)
        try:
            return AutoTokenizer.from_pretrained(str(self.model_dir))
        except Exception:
            print("⚠️ Токенайзер не найден в папке, используем deepvk/USER-BGE-M3")
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
    """Превращает батч эмбеддингов в один вектор (усреднение)."""
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
st.title("🔍 Сравнение оригинальной и квантованной моделей")

st.markdown(
    """
    - Оригинал грузим через **SentenceTransformer**.
    - Квантованную модель грузим через **QuantModel** (onnxruntime, память экономится).
    """
)

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
    force_download = st.checkbox("♻️ Перекачать модель заново", False)

st.markdown("---")
input_text = st.text_area("Тексты для теста (по одной строке)", "Это тестовое предложение.\nПример второй строки.")
texts = [t.strip() for t in input_text.split("\n") if t.strip()]

batch_size = st.slider("Количество повторов для throughput-теста", 1, 128, 8)
run_button = st.button("🚀 Запустить тест")

# ============================================================
# 🚀 Основная логика
# ============================================================
if run_button:
    try:
        proc = psutil.Process()

        # --- Загружаем оригинал через SentenceTransformer
        with st.spinner("Загрузка оригинальной модели..."):
            orig_model = SentenceTransformer(orig_id)
            st.success(f"✅ Оригинал загружен из {orig_id}")

        # --- Загружаем квант через QuantModel
        with st.spinner("Загрузка квантованной модели..."):
            quant_model = QuantModel(
                model_id=quant_id,
                source=quant_source,
                model_dir=quant_dir,
                tokenizer_name=tokenizer_name if tokenizer_name else None,
                force_download=force_download
            )
            st.success(f"✅ Квантованная модель загружена ({quant_model.model_path})")

        texts_for_run = (texts * batch_size)[:max(len(texts), 1)]

        # Оригинал
        t0 = time.perf_counter()
        orig_embs = orig_model.encode(texts_for_run, normalize_embeddings=True)
        t1 = time.perf_counter()
        orig_time = t1 - t0
        mem_after_orig = proc.memory_info().rss / 1024 ** 2

        # Квант
        t0 = time.perf_counter()
        quant_embs = quant_model.encode(texts_for_run, normalize=True)
        t1 = time.perf_counter()
        quant_time = t1 - t0
        mem_after_quant = proc.memory_info().rss / 1024 ** 2

        v_orig = to_vector(orig_embs)
        v_quant = to_vector(quant_embs)
        if v_orig.shape != v_quant.shape:
            st.warning(f"Размерности различаются: {v_orig.shape} vs {v_quant.shape}, усечем до min.")
            m = min(v_orig.size, v_quant.size)
            v_orig = v_orig[:m]
            v_quant = v_quant[:m]

        cos = cosine_similarity(v_orig, v_quant)

        # Вывод
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
