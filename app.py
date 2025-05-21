import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import gdown

# Inisialisasi awal
@st.cache_resource
def load_components():
    # Download model
    url = "https://drive.google.com/uc?id=11ohiewVNh2I7nLP12PypjzOnuU6boNOa"
    output = "best_model.pt"
    gdown.download(url, output, quiet=False)
    
    # Load model dengan ignore_mismatched_sizes
    model = AutoModelForSequenceClassification.from_pretrained(
        "flax-community/indonesian-roberta-base",
        num_labels=1,
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(output, map_location='cpu'))
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("flax-community/indonesian-roberta-base")
    
    return model, tokenizer

# Panggil fungsi load_components
try:
    model, tokenizer = load_components()
except Exception as e:
    st.error(f"Gagal memuat model: {str(e)}")
    st.stop()

def predict_toxicity(text):
    try:
        normalized_text = re.sub(r'[^\w\s]', '', text.lower()).strip()
        inputs = tokenizer(
            normalized_text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**inputs)
        return torch.sigmoid(outputs.logits.squeeze()).item()
    except Exception as e:
        st.error(f"Error dalam prediksi: {str(e)}")
        return 0.5

# UI
st.title('🔍 Deteksi Toxic Comments')
user_input = st.text_area("Masukkan teks:")
if st.button("Periksa") and user_input:
    prob = predict_toxicity(user_input)
    st.write(f"**{'🚫 Toxic' if prob > 0.5 else '✅ Aman'}** (Skor: {prob:.4f})")
elif not user_input:
    st.warning("Masukkan teks terlebih dahulu!")
