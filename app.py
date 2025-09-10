# app.py
import os
import zipfile
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import streamlit as st
import onnxruntime as ort
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizerFast

# ============================
# Page config
# ============================
st.set_page_config(page_title="FP32 vs ONNX(INT8) comparer", layout="wide")
st.title("🔍 FP32 (HF)  vs ONNX(INT8) comparer — USER-BGE-M3 style")

st.markdown(
    "Загрузите/укажите ONNX-квант (локально или через распакованный ZIP), "
    "укажите HF id оригинала (или оставьте по умолчанию), затем нажмите `Run`. "
    "Программа рассчитана на CPU."
)

# ============================
# Sidebar controls
# ============================
with st.sidebar:
    st.header("Settings")
    orig_model_id = st.text_input("Original HF model id", value="deepvk/USER-BGE-M3")
    # allow local path or folder name where onnx & tokenizer stored (after unzip)
    quant_local_dir = st.text_input("Quant ONNX local folder (or leave empty)", value="onnx-quant")
    gdrive_zip_id = st.text_input("Optional: Google Drive zip file id (zip will be downloaded->extracted into folder above)", value="")
    quant_onnx_name = st.text_input("ONNX filename (leave empty to auto-find)", value="")  # e.g. model_quantized.onnx
    cpu_only = st.checkbox("Force CPU for HF model", value=True)
    warmup_runs = st.number_input("Warmup runs", min_value=0, max_value=5, value=2)
    bench_batch_size = st.number_input("Batch size for benchmark", min_value=1, max_value=256, value=32)
    eval_samples = st.number_input("Number of samples for cosine evaluation", min_value=1, max_value=2000, value=500)
    run_button = st.button("🚀 Run comparison")

# ============================
# Utilities: unzip, find files
# ============================
def unzip_to_dir(zip_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(out_dir))

def find_onnx_file(model_dir: Path, prefer_name: Optional[str] = None) -> Optional[Path]:
    if not model_dir.exists():
        return None
    if prefer_name:
        p = model_dir / prefer_name
        if p.exists():
            return p
    onnxs = list(model_dir.rglob("*.onnx"))
    if not onnxs:
        return None
    # prefer file with 'quant' in name
    for f in onnxs:
        if "quant" in f.name.lower():
            return f
    return onnxs[0]

# ============================
# Caching heavy resources
# ============================
@st.cache_resource
def load_tokenizer_prefer_local(local_dir: Optional[str], hf_id: str):
    """
    Попытаемся загрузить токенизатор: сначала из local_dir (если файлы есть),
    иначе из HF (hf_id).
    """
    # Try local
    if local_dir:
        p = Path(local_dir)
        if p.exists():
            try:
                t = PreTrainedTokenizerFast.from_pretrained(str(p))
                return t, True
            except Exception:
                try:
                    t = AutoTokenizer.from_pretrained(str(p), use_fast=True)
                    return t, True
                except Exception:
                    pass
    # fallback to HF
    t = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    return t, False

@st.cache_resource
def load_pytorch_model(hf_id: str, force_cpu: bool = True):
    """
    Загружаем AutoModel (PyTorch). Загружаем на CPU, не используем device_map="auto" чтобы не требовать accelerate.
    """
    # Force CPU
    model = AutoModel.from_pretrained(hf_id, torch_dtype=torch.float32, low_cpu_mem_usage=False)
    if force_cpu:
        # ensure on cpu
        model.to("cpu")
    model.eval()
    return model

@st.cache_resource
def load_onnx_session(onnx_path: str):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    return sess

# ============================
# Encoding helpers
# ============================
def pytorch_encode_batch(model, tokenizer, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
    """Return numpy array (n, dim). Uses mean pooling with attention_mask."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cpu")
        attention_mask = inputs["attention_mask"].to("cpu")
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            # last_hidden_state usually is index .last_hidden_state
            token_embs = out.last_hidden_state  # (batch, seq, dim)
            mask = attention_mask.unsqueeze(-1).to(torch.float32)  # (batch, seq, 1)
            summed = (token_embs * mask).sum(dim=1)                 # (batch, dim)
            counts = mask.sum(dim=1)                               # (batch, 1)
            counts = torch.clamp(counts, min=1e-9)
            sent_emb = summed / counts                             # (batch, dim)
            if normalize:
                sent_emb = torch.nn.functional.normalize(sent_emb, p=2, dim=1)
            all_embs.append(sent_emb.cpu().numpy())
    return np.vstack(all_embs)

def onnx_encode_batch(sess: ort.InferenceSession, tokenizer, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
    """Encode with ONNX session. Tokenizer return_tensors='np'"""
    all_embs = []
    # get model input names to filter out token_type_ids etc.
    model_input_names = {inp.name for inp in sess.get_inputs()}
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="np")
        # filter only allowed inputs
        ort_inputs = {k: v for k, v in inputs.items() if k in model_input_names}
        # Ensure attention_mask present for pooling fallback
        if "attention_mask" not in ort_inputs:
            # create ones
            seq_len = inputs.get("input_ids").shape[1]
            ort_inputs["attention_mask"] = np.ones((len(batch), seq_len), dtype=np.int64)
        outputs = sess.run(None, ort_inputs)
        emb = outputs[0]  # assume first output is token embeddings or pooled embeddings
        # pooling
        if emb.ndim == 3:
            att = ort_inputs.get("attention_mask")
            mask = att.astype(np.float32)[..., None]  # (batch, seq, 1)
            summed = (emb * mask).sum(axis=1)
            counts = mask.sum(axis=1)
            counts = np.clip(counts, 1e-9, None)
            sent_emb = summed / counts
        elif emb.ndim == 2:
            sent_emb = emb
        else:
            sent_emb = emb.mean(axis=1)
        if normalize:
            norms = np.linalg.norm(sent_emb, axis=1, keepdims=True) + 1e-12
            sent_emb = sent_emb / norms
        all_embs.append(sent_emb)
    return np.vstack(all_embs)

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

# ============================
# Main run
# ============================
if run_button:
    st.info("Start: preparing resources...")
    # prepare local quant folder
    qdir = Path(quant_local_dir)
    if gdrive_zip_id.strip():
        # download zip via gdown only if present
        try:
            import gdown
            zpath = qdir.with_suffix(".zip")
            st.info("Downloading zip from Google Drive...")
            gdown.download(f"https://drive.google.com/uc?id={gdrive_zip_id}", str(zpath), quiet=False)
            st.info("Unzipping...")
            unzip_to_dir(zpath, qdir)
            zpath.unlink(missing_ok=True)
        except Exception as e:
            st.error(f"Failed to download/unzip GDrive zip: {e}")
            st.stop()

    # find ONNX
    onnx_file = find_onnx_file(qdir, prefer_name=(quant_onnx_name or None))
    if onnx_file is None:
        st.warning("ONNX not found in local quant dir — you must supply a folder containing .onnx or use a valid zip.")
        # but still try to continue if user only wants to test original
    else:
        st.success(f"Found ONNX: {onnx_file}")

    # load tokenizer (prefer local tokenizers from quant folder)
    try:
        tokenizer, used_local = load_tokenizer_prefer_local(str(qdir) if qdir.exists() else None, orig_model_id)
        st.write(f"Tokenizer loaded. used_local={used_local}")
    except Exception as e:
        st.error(f"Failed to load tokenizer: {e}")
        st.stop()

    # load original HF model (PyTorch) — handle missing weights gracefully
    try:
        st.info("Loading original PyTorch model (may take time)...")
        model_orig = load_pytorch_model(orig_model_id, force_cpu=cpu_only)
        st.success("Original PyTorch model loaded.")
    except Exception as e:
        st.error("Failed to load original HF model. Make sure hf repo contains weights (pytorch_model.bin or model.safetensors).")
        st.text(str(e))
        st.stop()

    # load onnx session if present
    sess = None
    if onnx_file:
        try:
            st.info("Loading ONNX session...")
            sess = load_onnx_session(str(onnx_file))
            st.success("ONNX session loaded.")
            # log ONNX input names for debugging
            onnx_inputs = [inp.name for inp in sess.get_inputs()]
            st.write("ONNX expects inputs:", onnx_inputs)
        except Exception as e:
            st.error(f"Failed to create ONNX session: {e}")
            st.stop()

    # prepare texts
    st.subheader("Input texts")
    user_input = st.text_area("Enter test lines (one per line). Leave empty to auto-generate.", value="Тестовая строка для замера скорости.\nПример использования модели.\nКак дела?")
    user_lines = [l.strip() for l in user_input.splitlines() if l.strip()]
    if not user_lines:
        user_lines = [f"Тестовая строка {i}" for i in range(50)]
    # build eval set of requested size
    if len(user_lines) < eval_samples:
        times = (eval_samples + len(user_lines) - 1) // len(user_lines)
        eval_texts = (user_lines * times)[:eval_samples]
    else:
        eval_texts = user_lines[:eval_samples]

    st.write(f"Using {len(eval_texts)} samples for cosine evaluation and {len(user_lines)} unique lines.")

    # warmup
    st.info("Warmup runs...")
    for _ in range(int(warmup_runs)):
        _ = pytorch_encode_batch(model_orig, tokenizer, user_lines[:min(8, len(user_lines))], batch_size=int(bench_batch_size))
        if sess:
            _ = onnx_encode_batch(sess, tokenizer, user_lines[:min(8, len(user_lines))], batch_size=int(bench_batch_size))

    # benchmark speed (use longer list)
    st.info("Benchmarking inference time...")
    bench_texts = user_lines * 10
    t0 = time.perf_counter()
    _ = pytorch_encode_batch(model_orig, tokenizer, bench_texts, batch_size=int(bench_batch_size))
    t1 = time.perf_counter()
    orig_time = t1 - t0

    onnx_time = None
    if sess:
        t0 = time.perf_counter()
        _ = onnx_encode_batch(sess, tokenizer, bench_texts, batch_size=int(bench_batch_size))
        t1 = time.perf_counter()
        onnx_time = t1 - t0

    # compute evaluation embeddings
    st.info("Generating embeddings for evaluation...")
    emb_orig = pytorch_encode_batch(model_orig, tokenizer, eval_texts, batch_size=int(bench_batch_size))
    emb_onnx = None
    if sess:
        emb_onnx = onnx_encode_batch(sess, tokenizer, eval_texts, batch_size=int(bench_batch_size))

    # compute cosine similarity
    if emb_onnx is None:
        st.warning("ONNX embeddings not available (no ONNX file). Only original model metrics are shown.")
    else:
        per_cos = cosine_batch(emb_orig, emb_onnx)
        avg_cos = float(np.mean(per_cos))
        med_cos = float(np.median(per_cos))

    # file sizes
    def dir_size_mb(path: Path) -> float:
        if not path.exists():
            return 0.0
        total = 0
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        return total / (1024 * 1024)

    quant_size = dir_size_mb(qdir) if qdir.exists() else 0.0

    # results table
    st.subheader("Results")
    rows = []
    rows.append(("Orig time (s)", f"{orig_time:.4f}"))
    if onnx_time is not None:
        rows.append(("Quant time (s)", f"{onnx_time:.4f}"))
    if emb_onnx is not None:
        rows.append(("Avg cosine (orig vs quant)", f"{avg_cos:.6f}"))
        rows.append(("Median cosine", f"{med_cos:.6f}"))
    rows.append(("Quant model size (MB)", f"{quant_size:.1f}"))

    df = pd.DataFrame(rows, columns=["Metric", "Value"])
    st.table(df)

    # big metrics
    c1, c2, c3 = st.columns(3)
    if emb_onnx is not None:
        c1.metric("Avg cosine", f"{avg_cos:.4f}")
    else:
        c1.metric("Avg cosine", "n/a")
    c2.metric("Orig time (s)", f"{orig_time:.3f}")
    c3.metric("Quant time (s)", f"{onnx_time:.3f}" if onnx_time is not None else "n/a")

    st.success("Done ✅")
