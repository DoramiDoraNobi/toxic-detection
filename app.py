import streamlit as st
import torch
import re
import gdown
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Inisialisasi awal
model = None
tokenizer = None

def init_model():
    global model, tokenizer
    try:
        # Download model
        url = "https://drive.google.com/uc?id=11ohiewVNh2I7nLP12PypjzOnuU6boNOa"
        output = "best_model.pt"
        gdown.download(url, output, quiet=True)
        
        # Load model dengan config yang benar
        model = AutoModelForSequenceClassification.from_pretrained(
            "flax-community/indonesian-roberta-base",
            num_labels=1,
            ignore_mismatched_sizes=True
        )
        model.load_state_dict(torch.load(output, map_location='cpu'))
        
        # Inisialisasi tokenizer
        tokenizer = AutoTokenizer.from_pretrained("flax-community/indonesian-roberta-base")
        
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        st.stop()

@st.cache_resource
def load_resources():
    init_model()
    return model, tokenizer

# Panggil inisialisasi
load_resources()

def predict_toxicity(text):
    try:
        # Normalisasi teks
        text = re.sub(r'[^\w\s]', '', text.lower()).strip()
        
        # Tokenisasi
        inputs = tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        # Prediksi
        with torch.no_grad():
            outputs = model(**inputs)
            
        return torch.sigmoid(outputs.logits.squeeze()).item()
        
    except Exception as e:
        st.error(f"Error prediksi: {str(e)}")
        return 0.5

# UI
st.title('🔍 Deteksi Komentar Toxic')
user_input = st.text_area("Masukkan teks:")
if st.button("Periksa"):
    if user_input:
        prob = predict_toxicity(user_input)
        st.write(f"**{'🚫 Toxic' if prob > 0.5 else '✅ Aman'}** (Skor: {prob:.4f}")
    else:
        st.warning("Masukkan teks terlebih dahulu!")
