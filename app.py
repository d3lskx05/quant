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
            embeddings = (embeddings * mask[..., None]).*
