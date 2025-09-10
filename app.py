# app.py
import os
import zipfile
import time
import traceback
from pathlib import Path
from typing import Optional, List

import gdown
import numpy as np
import pandas as pd
import streamlit as st
import onnxruntime as ort
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizerFast, AutoTokenizer

# ============================
# Streamlit page config
# ============================
st.set_page_config(page_title="QUANT vs FP32 comparer", layout="wide")
st.title("🔍 Сравнение: оригинал (FP32) vs квант (ONNX INT8) — USER-BGE-M3")

with st.sidebar:
    st.header("Настройки")
    gdrive_id = st.text_input("Google Drive file ID (zip с ONNX)", "")
    quant_dir = st.text_input("Локальная папка для распаковки кванта", "onnx-quant")
    orig_model_id = st.text_input("Оригинальная модель (HF)", "deepvk/USER-bge-m3")
    cpu_only = st.checkbox("Принудительно CPU (для оригинала)", True)
    warmup_runs = st.number_input("Warmup runs", min_value=0, max_value=10, value=2)
    bench_batch_size = st.number_input("Batch size для теста скорости", min_value=1, max_value=256, value=32)
    calib_texts_count = st.number_input("Количество текстов для оценки cosine (samples)", min_value=1, max_value=5000, value=500)
    run_button = st.button("🚀 Запустить сравнение")

# ============================
# Helpers
# ============================
def download_and_extract_gdrive(gdrive_id: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir.with_suffix(".zip")
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    gdown.download(url, str(zip_path), quiet=False)
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(out_dir))
    zip_path.unlink(missing_ok=True)

def find_onnx_file(model_dir: Path) -> Optional[Path]:
    onnxs = list(model_dir.rglob("*.onnx"))
    if not onnxs:
        return None
    for f in onnxs:
        if "quant" in f.name.lower():
            return f
    return onnxs[0]

# ============================
# ONNX encoder
# ============================
class OnnxEncoder:
    def __init__(self, onnx_path: Path, tokenizer):
        self.onnx_path = str(onnx_path)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(self.onnx_path, sess_options=so, providers=providers)
        self.tokenizer = tokenizer

    @staticmethod
    def _mean_pooling(embs: np.ndarray, attention_mask: np.ndarray):
        if embs.ndim == 3:
            mask = attention_mask.astype(np.float32)[..., None]
            summed = (embs * mask).sum(axis=1)
            counts = mask.sum(axis=1)
            counts = np.clip(counts, 1e-9, None)
            return summed / counts
        elif embs.ndim == 2:
            return embs
        return embs.mean(axis=1)

    def encode_batch(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="np")

        # ✅ оставить только те входы, что ждёт ONNX
        model_inputs = [inp.name for inp in self.sess.get_inputs()]
        ort_inputs = {k: v for k, v in inputs.items() if k in model_inputs}

        outputs = self.sess.run(None, ort_inputs)
        emb = outputs[0]
        pooled = self._mean_pooling(emb, ort_inputs.get("attention_mask", np.ones((len(texts), 1))))

        if normalize:
            norms = np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-12
            pooled = pooled / norms
        return pooled

# ============================
# Utils
# ============================
def cosine_batch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return (A * B).sum(axis=1)

def benchmark_encoder(encoder_func, texts: List[str], batch_size: int = 32) -> float:
    start = time.perf_counter()
    for i in range(0, len(texts), batch_size):
        _ = encoder_func(texts[i:i + batch_size])
    return time.perf_counter() - start

# ============================
# Main
# ============================
if run_button:
    try:
        model_dir = Path(quant_dir)
        if gdrive_id.strip():
            download_and_extract_gdrive(gdrive_id.strip(), model_dir)

        onnx_file = find_onnx_file(model_dir)
        if onnx_file is None:
            st.error("ONNX файл не найден.")
            st.stop()

        # load tokenizer
        try:
            tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model_dir))
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(orig_model_id, use_fast=True)

        onnx_encoder = OnnxEncoder(onnx_file, tokenizer)
        orig = SentenceTransformer(orig_model_id, device="cpu" if cpu_only else None)

        input_texts = st.text_area("Введите тестовые тексты:", value="Пример текста\nКак дела?\nМодель тест")
        user_texts = [t.strip() for t in input_texts.splitlines() if t.strip()]
        if not user_texts:
            user_texts = [f"Тестовая строка {i}" for i in range(50)]
        eval_texts = (user_texts * ((calib_texts_count + len(user_texts) - 1) // len(user_texts)))[:calib_texts_count]

        # warmup
        for _ in range(int(warmup_runs)):
            _ = orig.encode(user_texts[:8], normalize_embeddings=True)
            _ = onnx_encoder.encode_batch(user_texts[:8])

        # benchmark
        bench_texts = user_texts * 10
        orig_time = benchmark_encoder(lambda x: orig.encode(x, normalize_embeddings=True, batch_size=int(bench_batch_size)), bench_texts)
        onnx_time = benchmark_encoder(lambda x: onnx_encoder.encode_batch(x), bench_texts)

        # cosine similarity
        emb_orig = orig.encode(eval_texts, normalize_embeddings=True, batch_size=int(bench_batch_size))
        emb_onx = onnx_encoder.encode_batch(eval_texts)
        per_cos = cosine_batch(np.asarray(emb_orig), np.asarray(emb_onx))
        avg_cos, med_cos = float(np.mean(per_cos)), float(np.median(per_cos))

        # results
        st.subheader("📊 Результаты")
        df = pd.DataFrame({
            "Metric": ["Avg cosine", "Median cosine", "Orig time (s)", "Quant time (s)"],
            "Value": [f"{avg_cos:.6f}", f"{med_cos:.6f}", f"{orig_time:.4f}", f"{onnx_time:.4f}"]
        })
        st.table(df)
        st.success("Готово ✅")

    except Exception as e:
        st.error(f"Ошибка: {e}")
        st.text(traceback.format_exc())
