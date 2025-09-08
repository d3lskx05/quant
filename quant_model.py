# quant_model.py
import os
import zipfile
import gdown
import numpy as np
import onnxruntime as ort
from pathlib import Path
from functools import lru_cache
from transformers import AutoTokenizer
import huggingface_hub


class QuantModel:
    """
    Универсальный загрузчик квантизированных ONNX моделей.
    - Источники: Google Drive (gdrive), Hugging Face Hub (hf), локальная (local)
    - Автопоиск .onnx файла
    - Кэширование модели и эмбеддингов
    """

    def __init__(self, model_id: str, source: str = "gdrive",
                 model_dir: str = "onnx_model", tokenizer_name: str = None):
        self.model_id = model_id
        self.source = source
        self.model_dir = Path(model_dir)
        self.tokenizer_name = tokenizer_name
        self.model_path = None

        self._ensure_model()
        self.session = self._load_session()
        self.tokenizer = self._load_tokenizer()

    # ========================
    # 📥 Загрузка модели
    # ========================
    def _ensure_model(self):
        """Скачивание и распаковка модели, если она не локальная."""
        os.makedirs(self.model_dir, exist_ok=True)

        if not any(self.model_dir.glob("*.onnx")):
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
                    local_dir_use_symlinks=False
                )

            elif self.source == "local":
                print(f"📂 Использую локальную модель: {self.model_dir}")
            else:
                raise ValueError(f"❌ Неизвестный источник: {self.source}")

        # Ищем первый .onnx
        onnx_files = list(self.model_dir.rglob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"❌ В {self.model_dir} не найден .onnx файл!")
        self.model_path = onnx_files[0]
        print(f"✅ Найден ONNX файл: {self.model_path}")

    # ========================
    # 🚀 Загрузка ONNX Session
    # ========================
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

    # ========================
    # 📝 Загрузка токенайзера
    # ========================
    def _load_tokenizer(self):
        if self.tokenizer_name:
            return AutoTokenizer.from_pretrained(self.tokenizer_name)
        try:
            return AutoTokenizer.from_pretrained(str(self.model_dir))
        except Exception:
            print("⚠️ Токенайзер не найден в папке, используем deepvk/USER-BGE-M3")
            return AutoTokenizer.from_pretrained("deepvk/USER-BGE-M3")

    # ========================
    # 🔥 Кодирование
    # ========================
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


# ========================
# 🔗 Глобальный доступ
# ========================
@lru_cache(maxsize=1)
def get_model():
    model_id = os.getenv("MODEL_ID", "1lkrvCPIE1wvffIuCSHGtbEz3Epjx5R36")
    source = os.getenv("MODEL_SOURCE", "gdrive")
    model_dir = os.getenv("MODEL_DIR", "onnx-user-bge-m3")
    tokenizer = os.getenv("TOKENIZER_NAME", None)
    return QuantModel(model_id, source, model_dir, tokenizer)
