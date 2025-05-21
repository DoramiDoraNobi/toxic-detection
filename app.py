import streamlit as st
import torch
import re
import gdown
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@st.cache_resource
def load_model():
    # Download model dengan format khusus
    url = "https://drive.google.com/uc?id=11ohiewVNh2I7nLP12PypjzOnuU6boNOa&confirm=t&format=download"
    output = "best_model.pt"
    
    # Force download untuk menghindari cache
    gdown.download(url, output, quiet=True, fuzzy=True)
    
    # Load model dengan config khusus
    model = AutoModelForSequenceClassification.from_pretrained(
        "flax-community/indonesian-roberta-base",
        num_labels=1,
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(output, map_location='cpu'))
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("flax-community/indonesian-roberta-base")
    
    return model, tokenizer

try:
    model, tokenizer = load_model()
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.stop()

def predict_toxicity(text):
    try:
        # Preprocessing
        text = re.sub(r'[^\w\s]', '', text.lower()).strip()
        if not text:
            return 0.5
            
        inputs = tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.inference_mode():
            outputs = model(**inputs)
            
        return torch.sigmoid(outputs.logits.squeeze()).item()
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return 0.5

# UI
st.title('🔍 Deteksi Konten Toxic')
user_input = st.text_area("Masukkan teks:")
if st.button("Periksa") and user_input:
    prob = predict_toxicity(user_input)
    st.write(f"**Hasil:** {'🚫 Toxic' if prob > 0.5 else '✅ Aman'}")
    st.write(f"Skor Keyakinan: {prob:.4f}")
elif not user_input:
    st.warning("Silakan masukkan teks terlebih dahulu!")
