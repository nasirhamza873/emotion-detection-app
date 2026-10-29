import streamlit as st
import tensorflow as tf
import zipfile
import os
from PIL import Image

st.title("😊 Emotion Recognition App")
st.write("Loading model, please wait...")

# 1. Zip file extraction and model loading
@st.cache_resource
def load_my_model():
    if not os.path.exists('best_model.keras') and os.path.exists('best_model.zip'):
        with zipfile.ZipFile('best_model.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
    
    # Load model
    model = tf.keras.models.load_model('best_model.keras')
    return model

# Loading model
try:
    model = load_my_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# 2. User Image Uploader UI
uploaded_file = st.file_uploader("Upload an image to check emotion:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L') # Grayscale for FER-2013 dataset
    image_resized = image.resize((48, 48)) # Resizing according to model requirement
    st.image(image, caption='Uploaded Image', width=150)
    
    st.write("Image received successfully! Prediction will be displayed here.")
