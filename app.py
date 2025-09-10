import streamlit as st
import time
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache(allow_output_mutation=True)
def load_models(orig_id, quant_id):
    tokenizer = AutoTokenizer.from_pretrained(orig_id)
    model_orig = AutoModel.from_pretrained(orig_id, device_map="auto")
    model_q    = AutoModel.from_pretrained(quant_id, device_map="auto")
    return tokenizer, model_orig, model_q

st.title("Сравнение оригинальной и квантованной модели")

orig_id = "user/model-orig"
quant_id = "user/model-8bit"
tokenizer, model_orig, model_q = load_models(orig_id, quant_id)

text = st.text_area("Введите текст для анализа", "Hello, world!")
if st.button("Сравнить"):
    inputs = tokenizer(text, return_tensors="pt")
    # Оригинальная модель
    start = time.perf_counter()
    emb_orig = model_orig(**inputs).last_hidden_state.mean(dim=1).detach().numpy()
    orig_time = time.perf_counter() - start
    # Квантованная модель
    start = time.perf_counter()
    emb_q = model_q(**inputs).last_hidden_state.mean(dim=1).detach().numpy()
    quant_time = time.perf_counter() - start
    # Косинусная близость
    cos = cosine_similarity(emb_orig, emb_q)[0][0]
    st.write(f"Cosine similarity: **{cos:.4f}**")
    st.write(f"Time (orig): {orig_time:.3f}s, Time (quant): {quant_time:.3f}s")
