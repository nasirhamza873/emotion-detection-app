import streamlit as st
import tensorflow as tf
import zipfile
import os
import numpy as np
from PIL import Image

st.title("😊 Emotion Recognition App")
st.write("Loading model, please wait...")

@st.cache_resource
def load_my_model():
    if not os.path.exists('best_model.keras') and os.path.exists('best_model.zip'):
        with zipfile.ZipFile('best_model.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
    
    model = tf.keras.models.load_model('best_model.keras')
    return model

try:
    model = load_my_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

uploaded_file = st.file_uploader("Upload a close-up face image to check emotion:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L') 
    st.image(image, caption='Uploaded Image', width=150)
    
    # Preprocessing: Resizing and Normalizing just like training
    image_resized = image.resize((48, 48))
    img_array = np.array(image_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Shape: (1, 48, 48)
    img_array = np.expand_dims(img_array, axis=-1) # Shape: (1, 48, 48, 1)
    
    if st.button('Predict Emotion'):
        with st.spinner('Analyzing emotion...'):
            prediction = model.predict(img_array)
            
            # Print raw prediction array to see what model is outputting
            st.write("Raw Model Output:", prediction)
            
            max_index = int(np.argmax(prediction))
            predicted_emotion = emotion_labels[max_index]
            confidence = float(np.max(prediction)) * 100
            
            st.success(f"Predicted Emotion: **{predicted_emotion}** ({confidence:.2f}%)")
            
            st.write("Confidence breakdown for all emotions:")
            for i, emo in enumerate(emotion_labels):
                st.write(f"- {emo}: {float(prediction[0][i])*100:.2f}%")
