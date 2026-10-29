import streamlit as st
import tensorflow as tf
import os
import numpy as np
from PIL import Image

st.title("😊 Emotion Recognition App (Live & Upload)")

@st.cache_resource
def load_my_model():
    # Direct naya model load karna bina zip ke
    model = tf.keras.models.load_model('best_model.keras')
    return model

try:
    model = load_my_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# FER-2013 standard alphabetical order
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

option = st.radio("Choose Input Method:", ("Upload Image", "Live Camera"))

uploaded_file = None
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload a close-up face image:", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("Take a picture using your camera")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L') 
    st.image(image, caption='Selected Image', width=150)
    
    # Preprocessing: Resizing and Normalizing (rescale=1/255 jo training mein kiya tha)
    image_resized = image.resize((48, 48))
    img_array = np.array(image_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Shape: (1, 48, 48)
    img_array = np.expand_dims(img_array, axis=-1) # Shape: (1, 48, 48, 1)
    
    if st.button('Predict Emotion'):
        with st.spinner('Analyzing emotion...'):
            prediction = model.predict(img_array)
            
            max_index = int(np.argmax(prediction))
            predicted_emotion = emotion_labels[max_index]
            confidence = float(np.max(prediction)) * 100
            
            st.success(f"Predicted Emotion: **{predicted_emotion}** ({confidence:.2f}%)")
            
            st.write("Confidence breakdown for all emotions:")
            for i, emo in enumerate(emotion_labels):
                st.write(f"- {emo}: {float(prediction[0][i])*100:.2f}%")
