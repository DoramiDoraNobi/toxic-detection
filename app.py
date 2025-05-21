import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import gdown

# Inisialisasi model dan tokenizer di luar cache untuk akses global
model = None
tokenizer = None

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return ' '.join(text.split())

@st.cache_resource
def load_model():
    global model, tokenizer  # Gunakan global variable
    
    # Download model
    url = "https://drive.google.com/uc?id=11ohiewVNh2I7nLP12PypjzOnuU6boNOa"
    output = "best_model.pt"
    gdown.download(url, output, quiet=False)
    
    # Load model dan tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        "flax-community/indonesian-roberta-base", 
        num_labels=1
    )
    model.load_state_dict(torch.load(output, map_location='cpu'))
    
    tokenizer = AutoTokenizer.from_pretrained("flax-community/indonesian-roberta-base")
    
    return model, tokenizer

# Panggil fungsi load_model() saat aplikasi dimulai
load_model()

def predict_toxicity(text):
    normalized_text = normalize_text(text)
    
    # Pastikan tokenizer sudah terinisialisasi
    if tokenizer is None:
        st.error("Tokenizer belum terinisialisasi!")
        return 0.5
    
    inputs = tokenizer(
        normalized_text,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    return torch.sigmoid(outputs.logits.squeeze()).item()

# UI
st.title('🔍 Deteksi Toxic Comments Bahasa Indonesia')
user_input = st.text_area("Masukkan teks yang ingin diperiksa:", "")
predict_btn = st.button("Periksa")

if predict_btn:
    if user_input:
        probability = predict_toxicity(user_input)
        prediction = "Toxic 🚫" if probability > 0.5 else "Non-Toxic ✅"
        
        st.subheader("Hasil Deteksi:")
        st.write(f"**{prediction}** (Skor: {probability:.4f})")
        st.progress(round(probability if prediction == "Toxic 🚫" else 1 - probability, 2))
    else:
        st.warning("Silakan masukkan teks terlebih dahulu!")
