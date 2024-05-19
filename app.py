import os

import keras
from keras.models import load_model
import streamlit as st 
import tensorflow as tf
import numpy as np
import cv2 as cv

Number_classification=[0,1,2,3,4,5,6,7,8,9]

st.header('Number Classification CNN Model')
model = load_model('Number_classification.h5')

def classify_images(image_path):
    file_bytes = np.asarray(bytearray(image_path.read()), dtype=np.uint8)
    # Decode the bytes to an image in grayscale
    img = cv.imdecode(file_bytes, cv.IMREAD_GRAYSCALE)
    predictions_single = model.predict(np.array([img]))
    predicted_label = Number_classification[np.argmax(predictions_single)]
    return predicted_label

uploaded_file = st.file_uploader('Upload an Image')
if uploaded_file is not None:
    with open(os.path.join('upload', uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, width = 200)

    st.markdown('predict output: '+str(classify_images(uploaded_file)))
