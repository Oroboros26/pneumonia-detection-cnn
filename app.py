import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Define image dimensions (must match the training dimensions)
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Load the trained model
@st.cache_resource # Cache the model loading for performance
def load_my_model():
    model_path = 'pneumonia_detection_model_transfer.h5'
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please ensure it's in the same directory as this script.")
        return None
    model = tf.keras.models.load_model(model_path)
    return model

model = load_my_model()

st.title('Pneumonia Detection from Chest X-ray Images')
st.write('Upload a chest X-ray image to get a prediction (Pneumonia/Normal).')

if model is None:
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded X-ray Image', use_column_width=True)
    st.write("\n")

    # Preprocess the image
    img_array = np.array(image.resize((IMG_WIDTH, IMG_HEIGHT))) # Resize
    img_array = img_array / 255.0 # Rescale to [0, 1]
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)
    confidence = prediction[0][0]

    st.subheader('Prediction:')

    if confidence > 0.5:
        st.markdown(f"<h3 style='color:red;'>Pneumonia Detected! (Confidence: {confidence:.2f})</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color:green;'>Normal (Confidence: {1 - confidence:.2f})</h3>", unsafe_allow_html=True)

    st.write("\n")
    st.write('**Note**: This is a model prediction and should not be used as a substitute for professional medical advice.')