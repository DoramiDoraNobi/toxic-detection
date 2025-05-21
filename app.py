import streamlit as st
import torch
import re
import gdown
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Cache resources dengan benar
@st.cache_resource
def load_model():
    # Download model
    url = "https://drive.google.com/uc?id=11ohiewVNh2I7nLP12PypjzOnuU6boNOa"
    output = "best_model.pt"
    gdown.download(url, output, quiet=True)
    
    # Load model dengan config
    model = AutoModelForSequenceClassification.from_pretrained(
        "flax-community/indonesian-roberta-base",
        num_labels=1,
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(output, map_location='cpu'))
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("flax-community/indonesian-roberta-base")
    
    return model, tokenizer

# Inisialisasi awal
try:
    model, tokenizer = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {str(e)}")
    st.stop()

def predict_toxicity(text):
    try:
        # Preprocessing
        text = re.sub(r'[^\w\s]', '', text.lower()).strip()
        if not text:
            return 0.5
            
        # Tokenisasi
        inputs = tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        # Inference
        with torch.inference_mode():
            outputs = model(**inputs)
            
        return torch.sigmoid(outputs.logits.squeeze()).item()
        
    except Exception as e:
        st.error(f"Error prediksi: {str(e)}")
        return 0.5

# UI
st.title('🔍 Deteksi Konten Toxic')
user_input = st.text_area("Masukkan teks:")
if st.button("Periksa"):
    if user_input:
        prob = predict_toxicity(user_input)
        st.write(f"**Hasil:** {'🚫 Toxic' if prob > 0.5 else '✅ Aman'}")
        st.write(f"Skor Keyakinan: {prob:.4f}")
    else:
        st.warning("Silakan masukkan teks terlebih dahulu!")
