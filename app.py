import zipfile
import os
import tensorflow as tf

# 1. Agar zip file maujood hai aur .keras file nahi hai, to usay unzip karein
if os.path.exists('best_model.zip') and not os.path.exists('best_model.keras'):
    with zipfile.ZipFile('best_model.zip', 'r') as zip_ref:
        zip_ref.extractall('.')
    print("Model extracted successfully!")

# 2. Ab pehle ki tarah model ko load karlein
model = tf.keras.models.load_model('best_model.keras')
