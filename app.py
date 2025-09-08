# app.py
import os
import zipfile
import time
import traceback
from pathlib import Path
from typing import Optional, List

import gdown
import numpy as np
import onnxruntime as ort
import psutil
import streamlit as st
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

st.set_page_config(page_title="Quantized model tester", layout="wide")


# ======================
# Вспомогательные функции
# ======================
def _extract_gdrive_id(text: str) -> str:
    """Принимает GDrive ID или ссылку — возвращает ID."""
    if not text:
        return text
    if "drive.google.com" in text:
        if "id=" in text:
            return text.split("id=")[1].split("&")[0]
        parts = text.split("/")
        if "d" in parts:
            try:
                idx = parts.index("d")
                return parts[idx + 1]
            except Exception:
                pass
        for p in reversed(parts):
            if p:
                return p
    return text


def download_from_gdrive(gid: str, target_dir: str) -> str:
    """Скачать архив с Google Drive и разархивировать."""
    os.makedirs(target_dir, exist_ok=True)
    zip_path = f"{target_dir}.zip"
    gdown.download(f"https://drive.google.com/uc?id={gid}", zip_path, quiet=False)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)
    os.remove(zip_path)
    return target_dir


def download_from_hf(repo_id: str, target_dir: str, token: Optional[str] = None) -> str:
    """Скачать модель с HuggingFace Hub."""
    snapshot_download(repo_id=repo_id, local_dir=target_dir,
                      local_dir_use_symlinks=False, token=token)
    return target_dir


def download_model(source: str, identifier: str, target_dir: str, hf_token: Optional[str] = None) -> str:
    """
    Скачивает модель в target_dir и возвращает путь.
    source: 'gdrive' | 'hf' | 'local'
    """
    if os.path.exists(target_dir) and os.listdir(target_dir):
        return target_dir

    if source == "local":
        if not os.path.exists(identifier):
            raise FileNotFoundError(f"Local path not found: {identifier}")
        os.makedirs(target_dir, exist_ok=True)
        # копируем файлы
        import shutil
        for item in os.listdir(identifier):
            s = os.path.join(identifier, item)
            d = os.path.join(target_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        return target_dir

    if source == "gdrive":
        gid = _extract_gdrive_id(identifier)
        return download_from_gdrive(gid, target_dir)

    if source == "hf":
        return download_from_hf(identifier, target_dir, hf_token)

    raise ValueError(f"Unknown source: {source}")


# ======================
# QuantModel
# ======================
class QuantModel:
    """ONNX-модель для экономии памяти и быстрой инференции."""
    def __init__(self, model_dir: str, tokenizer_name: Optional[str] = None):
        self.model_dir = model_dir
        self.onnx_path = self._find_onnx_file()
        self.session = self._create_session()
        self.tokenizer = self._load_tokenizer(tokenizer_name)

    def _find_onnx_file(self) -> str:
        for name in ["model_quantized.onnx", "model.onnx"]:
            path = Path(self.model_dir) / name
            if path.exists():
                return str(path)
        # искать рекурсивно
        matches = list(Path(self.model_dir).rglob("*.onnx"))
        if matches:
            return str(matches[0])
        raise FileNotFoundError(f"ONNX файл не найден в {self.model_dir}")

    def _create_session(self):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(self.onnx_path, sess_options=so, providers=["CPUExecutionProvider"])

    def _load_tokenizer(self, tokenizer_name: Optional[str]):
        candidates = [tokenizer_name, self.model_dir, "deepvk/USER-BGE-M3"]
        for c in candidates:
            if not c:
                continue
            try:
                return AutoTokenizer.from_pretrained(c, use_fast=True)
            except Exception:
                continue
        raise RuntimeError("Не удалось загрузить токенайзер.")

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="np")
        ort_inputs = {k: v for k, v in inputs.items()}
        outputs = self.session.run(None, ort_inputs)
        emb = outputs[0]
        if emb.ndim == 3:
            emb = emb.mean(axis=1)
        if normalize:
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10
            emb = emb / norms
        return emb


# ======================
# UI
# ======================
st.title("🔍 Сравнение оригинальной и квантованной модели (ONNX + QuantModel)")

col1, col2 = st.columns(2)
with col1:
    st.header("Оригинальная модель")
    orig_source = st.selectbox("Источник (original)", ["hf", "gdrive", "local"], index=0)
    orig_id = st.text_input("ID/Repo/Path (original)", "deepvk/USER-BGE-M3")
    orig_token = st.text_input("HF token (original, optional)", "", type="password")

with col2:
    st.header("Квантованная модель")
    quant_source = st.selectbox("Источник (quant)", ["hf", "gdrive", "local"], index=1)
    quant_id = st.text_input("ID/Repo/Path (quant)", "1lkrvCPIE1wvffIuCSHGtbEz3Epjx5R36")
    quant_token = st.text_input("HF token (quant, optional)", "", type="password")

st.markdown("---")
input_text = st.text_area("Тексты для теста", "Это тестовое предложение.\nПример второй строки.")
texts = [t.strip() for t in input_text.split("\n") if t.strip()]
batch_size = st.slider("Количество повторов для throughput-теста", 1, 128, 8)
run_button = st.button("Запустить тест")


if run_button:
    try:
        st.info("Скачиваем модели...")
        orig_dir = f"cache_orig_{orig_id.replace('/', '_')}"
        quant_dir = f"cache_quant_{quant_id.replace('/', '_')}"
        orig_path = download_model(orig_source, orig_id, orig_dir, orig_token or None)
        quant_path = download_model(quant_source, quant_id, quant_dir, quant_token or None)

        st.info("Грузим оригинальную модель...")
        orig_model = SentenceTransformer(orig_path)

        st.info("Грузим квантованную модель...")
        quant_model = QuantModel(quant_path)

        texts_for_run = (texts * batch_size)[:max(len(texts), 1)]
        proc = psutil.Process()

        # original
        t0 = time.perf_counter()
        orig_emb = orig_model.encode(texts_for_run, convert_to_numpy=True, show_progress_bar=False)
        t1 = time.perf_counter()
        orig_time = t1 - t0
        mem_after_orig = proc.memory_info().rss / 1024 ** 2

        # quant
        t0 = time.perf_counter()
        quant_emb = quant_model.encode(texts_for_run, normalize=True)
        t1 = time.perf_counter()
        quant_time = t1 - t0
        mem_after_quant = proc.memory_info().rss / 1024 ** 2

        v_orig = np.mean(orig_emb, axis=0)
        v_quant = np.mean(quant_emb, axis=0)
        min_dim = min(v_orig.size, v_quant.size)
        cos_sim = float(np.dot(v_orig[:min_dim], v_quant[:min_dim]) /
                        ((np.linalg.norm(v_orig[:min_dim]) * np.linalg.norm(v_quant[:min_dim])) + 1e-12))

        st.subheader("📊 Результаты")
        st.metric("Latency original (s)", f"{orig_time:.4f}")
        st.metric("Latency quant (s)", f"{quant_time:.4f}")
        st.metric("Throughput original (texts/s)", f"{len(texts_for_run) / max(orig_time, 1e-12):.1f}")
        st.metric("Throughput quant (texts/s)", f"{len(texts_for_run) / max(quant_time, 1e-12):.1f}")
        st.write(f"Cosine similarity: **{cos_sim:.4f}**")
        st.write(f"Memory after original: {mem_after_orig:.1f} MB")
        st.write(f"Memory after quant: {mem_after_quant:.1f} MB")

    except Exception as e:
        st.error(f"Ошибка: {e}")
        st.text(traceback.format_exc())
