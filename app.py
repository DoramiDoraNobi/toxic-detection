import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import gdown

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Hapus tanda baca
    text = re.sub(r'\d+', '', text)      # Hapus angka
    text = ' '.join(text.split())        # Hapus spasi berlebih
    return text

@st.cache_resource
def load_model():
    # Ganti "1FILE_ID" dengan ID file Google Drive Anda
    url = "https://drive.google.com/uc?id=11ohiewVNh2I7nLP12PypjzOnuU6boNOa"
    output = "best_model.pt"
    
    # Download model
    gdown.download(url, output, quiet=False)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        "flax-community/indonesian-roberta-base", 
        num_labels=1
    )
    model.load_state_dict(torch.load(output, map_location=torch.device('cpu')))
    tokenizer = AutoTokenizer.from_pretrained("flax-community/indonesian-roberta-base")
    
    return model, tokenizer

def predict_toxicity(text):
    normalized_text = normalize_text(text)
    
    inputs = tokenizer(
        normalized_text,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    prob = torch.sigmoid(outputs.logits.squeeze()).item()
    return prob

# UI Streamlit
st.title('🔍 Deteksi Toxic Comments Bahasa Indonesia')
st.write("Aplikasi deteksi komentar toxic menggunakan model RoBERTa")

user_input = st.text_area("Masukkan teks yang ingin diperiksa:", "")
predict_btn = st.button("Periksa")

if predict_btn and user_input:
    probability = predict_toxicity(user_input)
    threshold = 0.5
    prediction = "Toxic 🚫" if probability > threshold else "Non-Toxic ✅"
    
    st.subheader("Hasil Deteksi:")
    st.write(f"Prediksi: **{prediction}**")
    st.write(f"Skor Keyakinan: {probability:.4f}")
    
    # Visualisasi
    progress_value = probability if prediction == "Toxic 🚫" else (1 - probability)
    st.progress(round(progress_value, 2))

elif predict_btn and not user_input:
    st.warning("Silakan masukkan teks terlebih dahulu!")