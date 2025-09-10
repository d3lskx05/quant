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
# Универсальный загрузчик квантизированных ONNX моделей.
# ============================================================
class QuantModel:
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
        need_download = self.force_download or not any(self.model_dir.rglob("*.onnx"))
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
            elif self.source == "local":
                pass
            else:
                raise ValueError(f"❌ Неизвестный источник: {self.source}")
        onnx_files = list(self.model_dir.rglob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"❌ Нет .onnx модели в {self.model_dir}")
        self.model_path = onnx_files[0]
        for f in onnx_files:
            if "quant" in f.name.lower():
                self.model_path = f
                break

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
            tok = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
            print(f"🔑 Tokenizer загружен из repo/id: {self.tokenizer_name}")
            return tok
        try:
            tok = AutoTokenizer.from_pretrained(str(self.model_dir), use_fast=True)
            print(f"🔑 Tokenizer найден локально в {self.model_dir}")
            return tok
        except Exception:
            tok = AutoTokenizer.from_pretrained("deepvk/USER-BGE-M3", use_fast=True)
            print("⚠️ Tokenizer не найден локально, использован deepvk/USER-BGE-M3")
            return tok

    @lru_cache(maxsize=1024)
    def _encode_cached(self, text: str, normalize: bool = True):
        inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors="np")
        ort_inputs = {k: v for k, v in inputs.items()}
        outputs = self.session.run(None, ort_inputs)
        embeddings = outputs[0]
        if embeddings.ndim == 3:
            mask = ort_inputs["attention_mask"].astype(np.float32)  # (batch, seq)
            embeddings = (embeddings * mask[..., None]).sum(1) \
                         / np.clip(mask.sum(1, keepdims=True), 1e-6, None)
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
            embeddings = embeddings / norms
        return embeddings[0]

    def encode(self, texts, normalize=True):
        if isinstance(texts, str):
            texts = [texts]
        return np.array([self._encode_cached(t, normalize) for t in texts])

# ============================================================
# 🔧 Вспомогательные функции
# ============================================================
def cosine_batch(A, B):
    A = np.asarray(A)
    B = np.asarray(B)
    if A.shape != B.shape:
        m = min(A.shape[-1], B.shape[-1])
        A = A[..., :m]
        B = B[..., :m]
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return (A * B).sum(axis=1)

# ============================================================
# 🎛️ UI
# ============================================================
st.title("🔍 Тестирование моделей: Оригинал vs Квант")
mode = st.radio("Выберите режим:", ["Оригинальная модель", "Квантованная модель", "Сравнение обеих"])

if st.button("♻️ Сбросить сессию"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

input_text = st.text_area("Тексты для теста (по одной строке)", 
                          "Это тестовое предложение.\nПример второй строки.")
texts = [t.strip() for t in input_text.split("\n") if t.strip()]
batch_size = st.slider("Количество повторов для throughput-теста", 1, 128, 8)
force_download = st.checkbox("♻️ Перекачать квант-модель заново", False)

metrics_df = None

if mode == "Оригинальная модель":
    model_id = st.text_input("HF repo ID", "deepvk/USER-BGE-M3")

elif mode == "Квантованная модель":
    col1, col2 = st.columns(2)
    with col1:
        quant_source = st.selectbox("Источник", ["gdrive", "hf", "local"], index=1)
        quant_id = st.text_input("ID/Repo/Path", "1ym0Lb_1C0p0QSIEMOmFIFaGGtCk7JNO5")
    with col2:
        quant_dir = st.text_input("Папка для кванта", "onnx-user-bge-m3")
        tokenizer_name = st.text_input("Tokenizer name", "")

else:  # Сравнение обеих
    st.markdown("В этом режиме измеряем **качество (cosine similarity)** и **скорость**.")
    col1, col2 = st.columns(2)
    with col1:
        model_id = st.text_input("HF repo ID (оригинал)", "deepvk/USER-BGE-M3", key="orig_repo_cmp")
    with col2:
        quant_source = st.selectbox("Источник кванта", ["gdrive", "hf", "local"], 
                                    index=1, key="quant_src_cmp")
        quant_id = st.text_input("ID/Repo/Path (квант)", 
                                 "1ym0Lb_1C0p0QSIEMOmFIFaGGtCk7JNO5", key="quant_id_cmp")
    col3, col4 = st.columns(2)
    with col3:
        quant_dir = st.text_input("Папка для кванта", 
                                  "onnx-user-bge-m3-quantized-dyn", key="quant_dir_cmp")
    with col4:
        tokenizer_name = st.text_input("Tokenizer name", "", key="tok_cmp")

run_button = st.button("🚀 Запустить тест")

# ============================================================
# 🚀 Запуск теста
# ============================================================
if run_button:
    try:
        texts_for_run = texts * batch_size

        if mode == "Оригинальная модель":
            proc = psutil.Process()
            with st.spinner("Загрузка оригинальной модели..."):
                model = SentenceTransformer(model_id)
            t0 = time.perf_counter()
            embs = model.encode(texts_for_run, normalize_embeddings=True)
            t1 = time.perf_counter()
            latency = t1 - t0
            memory = proc.memory_info().rss / 1024 ** 2

            metrics_df = pd.DataFrame([{
                "Mode": "Original",
                "Batch Size": batch_size,
                "Latency (s)": latency,
                "Throughput (texts/s)": len(texts_for_run) / max(latency, 1e-12),
                "Memory (MB)": memory
            }])

            st.subheader("📊 Результаты")
            st.dataframe(metrics_df)

        elif mode == "Квантованная модель":
            proc = psutil.Process()
            with st.spinner("Загрузка квантованной модели..."):
                model = QuantModel(
                    model_id=quant_id,
                    source=quant_source,
                    model_dir=quant_dir,
                    tokenizer_name=tokenizer_name if tokenizer_name else None,
                    force_download=force_download
                )
            st.write(f"🔑 Используемый токенизатор: `{model.tokenizer.name_or_path}`")

            t0 = time.perf_counter()
            embs = model.encode(texts_for_run, normalize=True)
            t1 = time.perf_counter()
            latency = t1 - t0
            memory = proc.memory_info().rss / 1024 ** 2

            metrics_df = pd.DataFrame([{
                "Mode": "Quantized",
                "Batch Size": batch_size,
                "Latency (s)": latency,
                "Throughput (texts/s)": len(texts_for_run) / max(latency, 1e-12),
                "Memory (MB)": memory
            }])

            st.subheader("📊 Результаты")
            st.dataframe(metrics_df)

        else:  # Сравнение обеих
            with st.spinner("Загрузка оригинальной модели..."):
                orig = SentenceTransformer(model_id)
            t0 = time.perf_counter()
            orig_embs = orig.encode(texts_for_run, normalize_embeddings=True)
            t1 = time.perf_counter()
            orig_latency = t1 - t0

            with st.spinner("Загрузка квантованной модели..."):
                quant = QuantModel(
                    model_id=quant_id,
                    source=quant_source,
                    model_dir=quant_dir,
                    tokenizer_name=tokenizer_name if tokenizer_name else None,
                    force_download=force_download
                )
            st.write(f"🔑 Используемый токенизатор: `{quant.tokenizer.name_or_path}`")

            t0 = time.perf_counter()
            quant_embs = quant.encode(texts_for_run, normalize=True)
            t1 = time.perf_counter()
            quant_latency = t1 - t0

            O = np.asarray(orig_embs)
            Q = np.asarray(quant_embs)
            if O.ndim == 3:
                O = O.mean(axis=1)
            if Q.ndim == 3:
                Q = Q.mean(axis=1)
            if O.shape[1] != Q.shape[1]:
                m = min(O.shape[1], Q.shape[1])
                O = O[:, :m]
                Q = Q[:, :m]

            per_text_cos = cosine_batch(O, Q)
            avg_cos = float(per_text_cos.mean())
            med_cos = float(np.median(per_text_cos))

            metrics_df = pd.DataFrame([
                {
                    "Mode": "Original",
                    "Batch Size": batch_size,
                    "Latency (s)": orig_latency,
                    "Throughput (texts/s)": len(texts_for_run) / max(orig_latency, 1e-12),
                },
                {
                    "Mode": "Quantized",
                    "Batch Size": batch_size,
                    "Latency (s)": quant_latency,
                    "Throughput (texts/s)": len(texts_for_run) / max(quant_latency, 1e-12),
                },
            ])

            st.subheader("📊 Метрики скорости")
            st.dataframe(metrics_df)

            st.subheader("🎯 Качество")
            st.write(f"Средняя cosine similarity (по {len(per_text_cos)} текстам): **{avg_cos:.4f}**")
            st.write(f"Медиана cosine similarity: **{med_cos:.4f}**")

        if metrics_df is not None:
            csv = metrics_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Скачать результаты (CSV)",
                data=csv,
                file_name="metrics.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Ошибка: {e}")
        st.text(traceback.format_exc())
