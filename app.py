import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import gdown

# Fungsi untuk normalisasi teks
def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return ' '.join(text.split())

# Cache model dan tokenizer hanya sekali di Streamlit
@st.cache_resource
def load_model_tokenizer():
    # Download model dari Google Drive
    url = "https://drive.google.com/uc?id=11ohiewVNh2I7nLP12PypjzOnuU6boNOa"
    output = "best_model.pt"
    gdown.download(url, output, quiet=True)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        "flax-community/indonesian-roberta-base",
        num_labels=1
    )
    model.load_state_dict(torch.load(output, map_location="cpu"))
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("flax-community/indonesian-roberta-base")
    return model, tokenizer

# Inisialisasi model dan tokenizer
model, tokenizer = load_model_tokenizer()

# Fungsi prediksi
def predict_toxicity(text: str) -> float:
    normalized = normalize_text(text)
    inputs = tokenizer(
        normalized,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
    score = torch.sigmoid(outputs.logits.squeeze()).item()
    return score

# UI Streamlit
st.set_page_config(page_title="Deteksi Toxic Comments", layout="centered")
st.title("🔍 Deteksi Toxic Comments Bahasa Indonesia")

user_input = st.text_area("💬 Masukkan teks:", height=150)

if st.button("Periksa"):
    if not user_input.strip():
        st.warning("Silakan masukkan teks terlebih dahulu!")
    else:
        prob = predict_toxicity(user_input)
        label = "Toxic 🚫" if prob > 0.5 else "Non-Toxic ✅"
        st.subheader("Hasil Deteksi:")
        st.write(f"**{label}** (Skor: {prob:.4f})")
        st.progress(round(prob if label == "Toxic 🚫" else 1 - prob, 2))
