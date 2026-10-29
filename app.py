import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

st.title("Emotion Detection App")
st.write("Upload an image of a face to detect the emotion!")

# Load Model Structure & Weights
@st.cache_resource
def load_model():
    with open('model_structure.json', 'r') as f:
        model_json = f.read()
    model = tf.keras.models.model_from_json(model_json)
    model.load_weights('best_model.keras')
    return model

try:
    model = load_model()
    class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
except Exception as e:
    st.error(f"Error loading model: {e}")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L') # Convert to grayscale
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Preprocess the image
    img = image.resize((48, 48))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    
    # Predict
    predictions = model.predict(img_array)
    max_index = np.argmax(predictions[0])
    emotion = class_labels[max_index]
    
    st.success(f"Predicted Emotion: **{emotion}**")
