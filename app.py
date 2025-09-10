# app.py
import os
import zipfile
import time
import traceback
from pathlib import Path
from functools import lru_cache
from typing import Optional, List

import gdown
import numpy as np
import pandas as pd
import psutil
import streamlit as st
import onnxruntime as ort
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizerFast, AutoTokenizer

# ============================
# Streamlit page config
# ============================
st.set_page_config(page_title="QUANT vs FP32 comparer", layout="wide")
st.title("🔍 Сравнение: оригинал (FP32) vs квант (ONNX INT8) — USER-BGE-M3")
st.markdown(
    "Загрузите ZIP квант-модели (Google Drive ID), укажите папку распаковки и запустите тест. "
    "Приложение измерит скорость инференса (CPU) и качество (cosine similarity) между моделями."
)

# ============================
# UI inputs
# ============================
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

st.write("**Инструкции:** загрузите ZIP (содержит model_quantized.onnx и файлы токенизатора), "
         "введите его Google Drive ID сверху и нажмите `Запустить сравнение`.")

# ============================
# Helpers
# ============================
def download_and_extract_gdrive(gdrive_id: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir.with_suffix(".zip")
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    st.info(f"Скачивание {url} -> {zip_path.name} ...")
    gdown.download(url, str(zip_path), quiet=False)
    st.info(f"Распаковка {zip_path.name} → {out_dir} ...")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(out_dir))
    zip_path.unlink(missing_ok=True)

def list_files_recursive(d: Path) -> List[str]:
    res = []
    if not d.exists():
        return res
    for p in d.rglob("*"):
        if p.is_file():
            res.append(str(p.relative_to(d)))
    return res

def flatten_if_nested(model_dir: Path):
    entries = list(model_dir.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        for p in inner.iterdir():
            target = model_dir / p.name
            if not target.exists():
                p.rename(target)
        try:
            inner.rmdir()
        except Exception:
            pass

def ensure_tokenizer_filenames(model_dir: Path):
    mappings = {
        "tokenizer": "tokenizer.json",
        "config": "config.json",
        "tokenizer_config": "tokenizer_config.json",
        "special_tokens_map": "special_tokens_map.json",
    }
    for src, dst in mappings.items():
        src_path = model_dir / src
        dst_path = model_dir / dst
        if src_path.exists() and not dst_path.exists():
            try:
                src_path.rename(dst_path)
            except Exception:
                pass

def find_onnx_file(model_dir: Path) -> Optional[Path]:
    onnxs = list(model_dir.rglob("*.onnx"))
    if not onnxs:
        return None
    for f in onnxs:
        if "quant" in f.name.lower():
            return f
    return onnxs[0]

# ============================
# ONNX encoder class
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
        else:
            return embs.mean(axis=1)

    def encode_batch(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="np")
        # ✅ фильтруем только те входы, которые реально принимает ONNX
        input_names = {inp.name for inp in self.sess.get_inputs()}
        ort_inputs = {k: v for k, v in inputs.items() if k in input_names}

        outputs = self.sess.run(None, ort_inputs)
        emb = outputs[0]
        pooled = self._mean_pooling(
            emb, ort_inputs.get("attention_mask", np.ones((len(texts), 1)))
        )

        if normalize:
            norms = np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-12
            pooled = pooled / norms
        return pooled

# ============================
# Utilities
# ============================
def cosine_batch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A)
    B = np.asarray(B)
    if A.shape[1] != B.shape[1]:
        m = min(A.shape[1], B.shape[1])
        A = A[:, :m]
        B = B[:, :m]
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return (A * B).sum(axis=1)

def benchmark_encoder(encoder_func, texts: List[str], batch_size: int = 32) -> float:
    start = time.perf_counter()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        _ = encoder_func(batch)
    end = time.perf_counter()
    return end - start

# ============================
# Main runner
# ============================
if run_button:
    try:
        st.info("Запуск. Подготовка директорий и (при необходимости) скачивание архива...")
        model_dir = Path(quant_dir)

        if gdrive_id.strip():
            download_and_extract_gdrive(gdrive_id.strip(), model_dir)

        flatten_if_nested(model_dir)
        ensure_tokenizer_filenames(model_dir)

        st.subheader("📂 Содержимое папки квант-модели")
        files = list_files_recursive(model_dir)
        if not files:
            st.error(f"Папка {model_dir} пуста или не существует. Проверьте путь/архив.")
            st.stop()
        st.write(files)

        onnx_file = find_onnx_file(model_dir)
        if onnx_file is None:
            st.error("ONNX файл не найден в указанной папке.")
            st.stop()
        st.success(f"Найден ONNX: {onnx_file.name}")

        tokenizer = None
        used_local_tokenizer = False
        try:
            tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model_dir))
            used_local_tokenizer = True
            st.success(f"Локальный токенизатор загружен из {model_dir}")
        except Exception as e_local:
            st.warning(f"Не удалось загрузить локальный токенизатор: {e_local}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
                used_local_tokenizer = True
                st.success("AutoTokenizer смог загрузить токенизатор локально.")
            except Exception as e2:
                st.warning(f"AutoTokenizer локально не сработал: {e2}")
                st.info("Будем пытаться взять токенизатор из HF hub (deepvk/USER-bge-m3).")
                tokenizer = AutoTokenizer.from_pretrained(orig_model_id, use_fast=True)
                used_local_tokenizer = False
                st.success(f"Токенизатор загружен из HF: {orig_model_id}")

        st.write(f"🔑 Используем локальный токенизатор? {used_local_tokenizer}")

        onnx_encoder = OnnxEncoder(onnx_file, tokenizer)

        device_arg = "cpu" if cpu_only else None
        st.info("Загружаем оригинальную модель на CPU...")
        if device_arg is not None:
            orig = SentenceTransformer(orig_model_id, device="cpu")
        else:
            orig = SentenceTransformer(orig_model_id)
        st.success("Оригинальная модель загружена.")

        st.subheader("🧪 Ввод тестовых фраз")
        input_texts = st.text_area(
            "Введите тестовые тексты (по одной строке). Оставьте пустым, чтобы генерировать автоматически.",
            value="Тестовая строка для замера скорости.\nПример использования модели.\nКак дела?"
        )
        user_texts = [t.strip() for t in input_texts.splitlines() if t.strip()]
        if not user_texts:
            user_texts = [f"Тестовая строка номер {i}" for i in range(50)]
        if len(user_texts) < calib_texts_count:
            times = (calib_texts_count + len(user_texts) - 1) // len(user_texts)
            eval_texts = (user_texts * times)[:calib_texts_count]
        else:
            eval_texts = user_texts[:calib_texts_count]

        st.write(f"Используем {len(eval_texts)} текстов для оценки cosine similarity и {len(user_texts)} уникальных введённых фраз.")

        st.info(f"Прогрев моделей ({warmup_runs} прогонов)...")
        for _ in range(int(warmup_runs)):
            _ = orig.encode(user_texts[:min(8, len(user_texts))], normalize_embeddings=True)
            _ = onnx_encoder.encode_batch(user_texts[:min(8, len(user_texts))], normalize=True)

        st.info("Измеряем скорость инференса (полный прогон)...")
        bench_texts = user_texts * 10
        t0 = time.perf_counter()
        _ = orig.encode(bench_texts, normalize_embeddings=True, batch_size=int(bench_batch_size))
        t1 = time.perf_counter()
        orig_time = t1 - t0
        t0 = time.perf_counter()
        _ = onnx_encoder.encode_batch(bench_texts, normalize=True)
        t1 = time.perf_counter()
        onnx_time = t1 - t0

        st.info("Генерируем эмбеддинги для оценки качества...")
        emb_orig = orig.encode(eval_texts, normalize_embeddings=True, batch_size=int(bench_batch_size))
        emb_onx = onnx_encoder.encode_batch(eval_texts, normalize=True)

        per_cos = cosine_batch(np.asarray(emb_orig), np.asarray(emb_onx))
        avg_cos = float(np.mean(per_cos))
        med_cos = float(np.median(per_cos))

        def get_dir_size_mb(path: Path) -> float:
            total = 0
            for p in path.rglob("*"):
                if p.is_file():
                    total += p.stat().st_size
            return total / (1024 * 1024)

        quant_size = get_dir_size_mb(model_dir)

        st.subheader("📊 Результаты сравнения")
        metrics = {
            "Metric": ["Avg cosine (orig vs quant)", "Median cosine", "Orig time (s)", "Quant time (s)", "Quant model size (MB)"],
            "Value": [f"{avg_cos:.6f}", f"{med_cos:.6f}", f"{orig_time:.4f}", f"{onnx_time:.4f}", f"{quant_size:.1f}"]
        }
        df = pd.DataFrame(metrics)
        st.table(df)

        col1, col2, col3 = st.columns(3)
        col1.metric("Среднее cosine", f"{avg_cos:.4f}")
        col2.metric("Время оригинала (s)", f"{orig_time:.3f}")
        col3.metric("Время кванта (s)", f"{onnx_time:.3f}")

        st.write("🧾 Доп. информация")
        st.write(f"Используем локальный токенизатор? {used_local_tokenizer}")
        st.write(f"ONNX файл: {onnx_file}")
        st.write("Список файлов в quant_dir:")
        st.json(files)

        st.success("Готово ✅")

    except Exception as e:
        st.error(f"Ошибка: {e}")
        st.text(traceback.format_exc())
