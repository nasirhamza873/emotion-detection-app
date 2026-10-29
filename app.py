import streamlit as st
import tensorflow as tf
import zipfile
import os
import numpy as np
from PIL import Image

st.title("😊 Emotion Recognition App")
st.write("Apna model successfully load ho raha hai, please wait...")

# 1. Zip file ko unzip karne ka code
@st.cache_resource
def load_my_model():
    if not os.path.exists('best_model.keras') and os.path.exists('best_model.zip'):
        with zipfile.ZipFile('best_model.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
    
    # Model load karein
    model = tf.keras.models.load_model('best_model.keras')
    return model

# Model load kar rahe hain
try:
    model = load_my_model()
    st.success("Model successfully load ho gaya hai!")
except Exception as e:
    st.error(f"Model load hone mein error aaya: {e}")

# 2. User se image upload karwane ka UI
uploaded_file = st.file_uploader("Ek tasveer (Image) upload karein (Emotion check karne ke liye):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L') # Grayscale (FER-2013 ke liye)
    image_resized = image.resize((48, 48)) # Model ke mutabiq size
    st.image(image, caption='Aap ki upload ki hui tasveer', width=150)
    
    # Prediction logic yahan add kar sakte hain
    st.write("Tasveer mil gayi hai! Model is par prediction jald karega.")
