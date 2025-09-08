import os, psutil
@st.cache
def load_model():
    save_dir = Path('model')
    save_dir.mkdir(exist_ok=True)
    model_path = save_dir / "user_bge_m3.onnx"
    if not model_path.exists():
        url = "https://drive.google.com/uc?id=1lkrvCPIE1wvffIuCSHGtbEz3Epjx5R36"
        import gdown
        gdown.download(url, str(model_path), quiet=False)
    # После скачивания возвращаем загруженную модель
    return SentenceTransformer(str(save_dir), backend="onnx", 
                               model_kwargs={"file_name": "model_quantized.onnx"})

proc = psutil.Process()
cpu = psutil.cpu_percent(interval=1)  # процент CPU
mem = proc.memory_info().rss / (1024**3)  # занято RAM в ГБ
st.metric("Загрузка CPU", f"{cpu:.1f}%")
st.metric("Используемая память", f"{mem:.2f} GB")
