import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

def normalize_text(text):
    """
    Fungsi normalisasi teks tanpa dependensi NLTK
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Hapus tanda baca
    text = re.sub(r'\d+', '', text)      # Hapus angka
    text = ' '.join(text.split())        # Hapus spasi berlebih
    return text

@st.cache_resource
def load_model():
    # Load model dan tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        "flax-community/indonesian-roberta-base", 
        num_labels=1
    )
    model.load_state_dict(torch.load('best_model.pt', map_location=torch.device('cpu')))
    tokenizer = AutoTokenizer.from_pretrained("flax-community/indonesian-roberta-base")
    return model, tokenizer

model, tokenizer = load_model()

def predict_toxicity(text):
    # Normalisasi teks
    normalized_text = normalize_text(text)
    
    # Tokenisasi
    inputs = tokenizer(
        normalized_text,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Prediksi
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Hitung probabilitas
    prob = torch.sigmoid(outputs.logits.squeeze()).item()
    return prob

# UI Streamlit
st.title('🔍 Deteksi Toxic Comments Bahasa Indonesia')
st.write("Aplikasi ini mendeteksi komentar toxic menggunakan model RoBERTa")

user_input = st.text_area("Masukkan teks yang ingin diperiksa:", "")
predict_btn = st.button("Periksa")

if predict_btn and user_input:
    probability = predict_toxicity(user_input)
    threshold = 0.5
    prediction = "Toxic 🚫" if probability > threshold else "Non-Toxic ✅"
    
    st.subheader("Hasil Deteksi:")
    st.write(f"Prediksi: **{prediction}**")
    st.write(f"Confidence Score: {probability:.4f}")
    
    # Visualisasi progress bar
    progress_bar = st.progress(0)
    progress_value = probability if prediction == "Toxic 🚫" else (1 - probability)
    progress_bar.progress(round(progress_value, 2))

elif predict_btn and not user_input:
    st.warning("Silakan masukkan teks terlebih dahulu!")