# app.py
import os
import zipfile
import traceback
from pathlib import Path
import time

import gdown
import huggingface_hub
import numpy as np
import psutil
import streamlit as st
import onnxruntime as ort
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from numpy.linalg import norm
from typing import Optional

st.set_page_config(page_title="Quantized model tester", layout="wide")


# -----------------------
# Helpers: download & prep
# -----------------------
def _extract_gdrive_id(text: str) -> str:
    """Принимает GDrive id или ссылку — возвращает id."""
    if not text:
        return text
    if "drive.google.com" in text:
        # try to parse id parameter or last path part
        if "id=" in text:
            return text.split("id=")[1].split("&")[0]
        parts = text.split("/")
        # look for file id typically after /d/
        if "d" in parts:
            try:
                idx = parts.index("d")
                return parts[idx + 1]
            except Exception:
                pass
        # fallback last non-empty part
        for p in reversed(parts):
            if p:
                return p
    return text


@st.cache_resource
def download_model(source: str, identifier: str, target_dir: str, hf_token: Optional[str] = None) -> str:
    """
    Скачивает модель в target_dir и возвращает путь к папке.
    source: 'gdrive' | 'hf' | 'local'
    identifier: gdrive id or hf repo id or local path
    """
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    # if already has files - assume downloaded
    if any(target.iterdir()):
        return str(target)

    if source == "local":
        # identifier is a path
        src = Path(identifier)
        if not src.exists():
            raise FileNotFoundError(f"Local path not found: {identifier}")
        # copy or symlink? we'll copy files (light)
        for p in src.iterdir():
            dst = target / p.name
            if p.is_dir():
                # copytree if not exists
                if not dst.exists():
                    import shutil
                    shutil.copytree(p, dst)
            else:
                if not dst.exists():
                    import shutil
                    shutil.copy2(p, dst)
        return str(target)

    if source == "gdrive":
        gid = _extract_gdrive_id(identifier)
        zip_path = Path(f"{target_dir}.zip")
        # download
        gdown.download(f"https://drive.google.com/uc?id={gid}", str(zip_path), quiet=False)
        # unzip
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target)
        zip_path.unlink()
        return str(target)

    if source == "hf":
        # snapshot_download
        huggingface_hub.snapshot_download(repo_id=identifier, local_dir=str(target), local_dir_use_symlinks=False, token=hf_token)
        return str(target)

    raise ValueError("Unknown source: " + str(source))


def find_onnx_file(model_dir: str, prefer_quantized: bool = True) -> Optional[str]:
    p = Path(model_dir)
    if not p.exists():
        return None
    if prefer_quantized:
        q = list(p.rglob("model_quantized.onnx"))
        if q:
            return str(q[0])
    # find any onnx
    all_onnx = list(p.rglob("*.onnx"))
    return str(all_onnx[0]) if all_onnx else None


@st.cache_resource
def prepare_model(model_dir: str,
                  mode_choice: str = "auto",   # 'auto'|'sentence-transformers'|'onnx'
                  use_quantized: bool = False,
                  tokenizer_name: Optional[str] = None,
                  hf_token: Optional[str] = None):
    """
    Возвращает dict с подготовленной моделью:
      {'type': 'sentence_transformer'|'onnx', 'model': ..., 'tokenizer': ..., 'onnx_path': ...}
    mode_choice:
      - auto: попытаемся сначала загрузить SentenceTransformer, иначе ONNX
      - sentence-transformers: попытка загрузить ST (если не проходит -> ошибка)
      - onnx: загружаем ONNX runtime (требует токенайзер)
    """
    model_dir = str(model_dir)
    result = {"type": None, "model": None, "tokenizer": None, "onnx_path": None, "model_dir": model_dir}

    # find ONNX if present
    onnx_path = find_onnx_file(model_dir, prefer_quantized=use_quantized)
    if use_quantized and not onnx_path:
        # try to find any onnx if quantized requested
        onnx_path = find_onnx_file(model_dir, prefer_quantized=False)

    # Helper to load tokenizer
    def load_tokenizer_try(loc: str):
        # try tokenizer_name, then model_dir, then fallback to common
        candidates = []
        if tokenizer_name:
            candidates.append(tokenizer_name)
        candidates.append(loc)
        # fallback
        candidates.append("deepvk/USER-BGE-M3")
        for c in candidates:
            try:
                tok = AutoTokenizer.from_pretrained(c, use_fast=True, trust_remote_code=False)
                return tok
            except Exception:
                continue
        return None

    # Mode-specific logic
    if mode_choice == "sentence-transformers":
        # Try to load as SentenceTransformer: if quantized and onnx_path exists, try ST with onnx backend first
        if use_quantized and onnx_path:
            try:
                st_model = SentenceTransformer(model_dir, backend="onnx", model_kwargs={"file_name": Path(onnx_path).name})
                result.update({"type": "sentence_transformer", "model": st_model, "onnx_path": onnx_path})
                return result
            except Exception as e:
                # fallback to trying plain ST (if folder is a ST package)
                pass
        # Try to load plain SentenceTransformer
        try:
            st_model = SentenceTransformer(model_dir)
            result.update({"type": "sentence_transformer", "model": st_model, "onnx_path": onnx_path})
            return result
        except Exception as e:
            raise RuntimeError(f"SentenceTransformer load failed for {model_dir}: {e}")

    if mode_choice == "onnx":
        if not onnx_path:
            raise FileNotFoundError(f"No ONNX file found in {model_dir}")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
        # tokenizer needed: try to load from model_dir or tokenizer_name
        tokenizer = load_tokenizer_try(model_dir)
        result.update({"type": "onnx", "model": session, "tokenizer": tokenizer, "onnx_path": onnx_path})
        return result

    # mode_choice == 'auto'
    # Prefer ST if folder looks like ST package; else if onnx exists and ST fails, load onnx
    # Try SentenceTransformer first (works for regular ST repos and also ST wrappers)
    try:
        if use_quantized and onnx_path:
            # try ST with onnx backend first
            try:
                st_model = SentenceTransformer(model_dir, backend="onnx", model_kwargs={"file_name": Path(onnx_path).name})
                result.update({"type": "sentence_transformer", "model": st_model, "onnx_path": onnx_path})
                return result
            except Exception:
                pass
        # plain ST
        st_model = SentenceTransformer(model_dir)
        result.update({"type": "sentence_transformer", "model": st_model, "onnx_path": onnx_path})
        return result
    except Exception:
        # fallback to ONNX if available
        if onnx_path:
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
            tokenizer = load_tokenizer_try(model_dir)
            result.update({"type": "onnx", "model": session, "tokenizer": tokenizer, "onnx_path": onnx_path})
            return result
        raise RuntimeError(f"Can't load model from {model_dir} in auto mode (no ST and no ONNX).")



# -----------------------
# Encoding wrapper
# -----------------------
def _normalize_to_vector(arr):
    a = np.array(arr)
    if a.ndim == 0:
        return a
    if a.ndim == 1:
        return a
    # if batch x dim -> average over batch
    if a.ndim == 2:
        # if shape (batch, dim) and batch==1 -> return flattened
        if a.shape[0] == 1:
            return a[0]
        return a.mean(axis=0)
    # if token-level (batch, seq, dim) -> mean over seq then mean over batch
    if a.ndim == 3:
        a = a.mean(axis=1)
        return _normalize_to_vector(a)
    # else flatten and return
    return a.ravel()


def encode_with_prepared(prep: dict, texts, normalize: bool = True):
    """
    Возвращает numpy array shape (n, dim) for list of texts or shape (dim,) for single text.
    prep: структура из prepare_model
    """
    if isinstance(texts, str):
        texts = [texts]
    if prep["type"] == "sentence_transformer":
        emb = prep["model"].encode(texts, convert_to_numpy=True, show_progress_bar=False)
        emb = np.array(emb)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        if normalize:
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10
            emb = emb / norms
        return emb
    elif prep["type"] == "onnx":
        session: ort.InferenceSession = prep["model"]
        tokenizer = prep["tokenizer"]
        if tokenizer is None:
            # try to use tokenizer from original model_dir (maybe stored nearby)
            raise RuntimeError("Tokenizer not found for ONNX model — please provide tokenizer in the model folder or set tokenizer_name.")
        # prepare batch
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="np")
        ort_inputs = {k: v for k, v in inputs.items()}
        outs = session.run(None, ort_inputs)
        # pick first output
        emb = np.array(outs[0])
        # convert shapes like (batch, seq, dim) -> mean over seq
        if emb.ndim == 3:
            emb = emb.mean(axis=1)
        if emb.ndim == 2:
            # normalize if asked
            if normalize:
                norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10
                emb = emb / norms
            return emb
        # fallback: flatten
        emb = emb.reshape(len(texts), -1)
        if normalize:
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10
            emb = emb / norms
        return emb
    else:
        raise RuntimeError("Unknown prepared model type: " + str(prep["type"]))


# -----------------------
# UI
# -----------------------
st.title("🔍 Тестер (оригинал vs квант) — гибкая загрузка (HF / GDrive / Local)")

st.markdown(
    """
    В этом интерфейсе для каждой модели можно указать:
    - источник (hf / gdrive / local),
    - идентификатор (repo_id / gdrive id / local path),
    - режим загрузки: **auto**, **sentence-transformers**, **onnx**.
    """
)

col1, col2 = st.columns(2)

with col1:
    st.header("Оригинальная модель")
    orig_source = st.selectbox("Источник (original)", ["hf", "gdrive", "local"], index=0)
    orig_id = st.text_input("ID/Repo или путь (original)", "deepvk/USER-BGE-M3")
    orig_mode = st.selectbox("Режим загрузки (original)", ["auto", "sentence-transformers", "onnx"], index=0)
    orig_token = st.text_input("HF token (original, optional)", value="", type="password")

with col2:
    st.header("Квантованная модель")
    quant_source = st.selectbox("Источник (quant)", ["gdrive", "hf", "local"], index=1)
    quant_id = st.text_input("ID/Repo или путь (quant)", "1lkrvCPIE1wvffIuCSHGtbEz3Epjx5R36")
    quant_mode = st.selectbox("Режим загрузки (quant)", ["auto", "sentence-transformers", "onnx"], index=2)
    quant_token = st.text_input("HF token (quant, optional)", value="", type="password")

st.markdown("---")
input_text = st.text_area("Тексты для теста (по одной в строке)", "Это тестовое предложение.\nПример второй строки.")
texts = [t.strip() for t in input_text.split("\n") if t.strip()]
batch_size = st.slider("Количество строк для throughput-теста (повторить набор)", 1, 128, 8)

run_button = st.button("Запустить тест")

# Results area
if run_button:
    st.info("Начинаем: скачиваем/подготавливаем модели и прогоняем замеры...")
    try:
        # prepare dirs unique for inputs to avoid collision
        orig_dir = f"cached_orig_{(orig_id.replace('/','_'))}"
        quant_dir = f"cached_quant_{(quant_id.replace('/','_'))}"

        # Download (cached)
        with st.spinner("Скачивание оригинальной модели..."):
            orig_path = download_model(orig_source, orig_id, orig_dir, hf_token=orig_token or None)
            st.success(f"Оригинал: {orig_path}")

        with st.spinner("Скачивание квантованной модели..."):
            quant_path = download_model(quant_source, quant_id, quant_dir, hf_token=quant_token or None)
            st.success(f"Квант: {quant_path}")

        # Prepare (cached)
        with st.spinner("Подготовка оригинальной модели..."):
            orig_prep = prepare_model(orig_path, mode_choice=orig_mode, use_quantized=False,
                                      tokenizer_name=None, hf_token=orig_token or None)
            st.success(f"Оригинал загружен как: {orig_prep['type']}")

        with st.spinner("Подготовка квантованной модели..."):
            quant_prep = prepare_model(quant_path, mode_choice=quant_mode, use_quantized=True,
                                       tokenizer_name=None, hf_token=quant_token or None)
            st.success(f"Квант загружен как: {quant_prep['type']} (onnx: {quant_prep.get('onnx_path')})")

        # Warm-up + timing
        # make repeated array for throughput measurement
        texts_for_run = (texts * batch_size)[: max(len(texts), 1)]

        # measure orig
        st.write("Измеряю оригинал...")
        proc = psutil.Process()
        t0 = time.perf_counter()
        orig_embs = encode_with_prepared(orig_prep, texts_for_run, normalize=True)
        t1 = time.perf_counter()
        orig_time = t1 - t0
        mem_after_orig = proc.memory_info().rss / 1024 ** 2

        # measure quant
        st.write("Измеряю квант...")
        t0 = time.perf_counter()
        quant_embs = encode_with_prepared(quant_prep, texts_for_run, normalize=True)
        t1 = time.perf_counter()
        quant_time = t1 - t0
        mem_after_quant = proc.memory_info().rss / 1024 ** 2

        # Prepare single-vector comparison: average embedding per set -> vector
        def to_vector(embs):
            emb = np.array(embs)
            if emb.ndim == 1:
                return emb
            # average across batch
            return emb.mean(axis=0)

        v_orig = to_vector(orig_embs)
        v_quant = to_vector(quant_embs)

        # if shapes mismatch, try project or warn
        if v_orig.shape != v_quant.shape:
            st.warning(f"Размерности эмбеддингов не совпадают: original {v_orig.shape} vs quant {v_quant.shape}. Конвертирую к общему виду по усреднению/обрезке.")
            # simple fallback: truncate to min dim
            m = min(v_orig.size, v_quant.size)
            v_orig = v_orig[:m]
            v_quant = v_quant[:m]

        # cosine
        cos = float(np.dot(v_orig, v_quant) / ((norm(v_orig) * norm(v_quant)) + 1e-12))

        # throughput (texts/sec)
        orig_throughput = len(texts_for_run) / max(orig_time, 1e-12)
        quant_throughput = len(texts_for_run) / max(quant_time, 1e-12)

        # show results
        st.subheader("Результаты")
        st.metric("Latency original (s)", f"{orig_time:.4f}")
        st.metric("Latency quant (s)", f"{quant_time:.4f}")
        st.metric("Throughput original (texts/s)", f"{orig_throughput:.1f}")
        st.metric("Throughput quant (texts/s)", f"{quant_throughput:.1f}")
        st.write(f"Cosine similarity (orig vs quant): **{cos:.4f}**")
        st.write(f"Memory after original run: {mem_after_orig:.1f} MB")
        st.write(f"Memory after quant run: {mem_after_quant:.1f} MB")
        st.write("Оригинальную модель подготовлено как:", orig_prep["type"])
        st.write("Квантовую модель подготовлено как:", quant_prep["type"])
        if quant_prep.get("onnx_path"):
            st.write("Квантованный onnx:", quant_prep.get("onnx_path"))

        st.bar_chart({
            "Latency (s)": [orig_time, quant_time],
            "Throughput (texts/s)": [orig_throughput, quant_throughput]
        })

    except Exception as e:
        st.error("Ошибка: " + str(e))
        st.text(traceback.format_exc())
