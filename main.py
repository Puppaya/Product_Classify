import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("model2/keras_model.h5", compile=False)

# Load the labels with explicit encoding
with open("model2/labels.txt", "r", encoding="utf-8") as f:
    class_names = f.readlines()

# Define a function to preprocess the image
def preprocess_image(image: Image.Image) -> np.ndarray:
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return np.expand_dims(normalized_image_array, axis=0)

# Streamlit UI
st.markdown("<h1 style='text-align: center; font-family: Phetsarath OT;'>ແບບຈຳລອງການຈຳແນກສິນຄ້າແຟຊັ່ນ</h1>",
            unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; font-family: Times New Roman;'>Model For Classify Fsdhion Products</h1>",
            unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image.', width=224)

    # Preprocess the image
    data = preprocess_image(image)

    # Predict
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Display prediction and confidence score
    st.markdown(f"<h2>Class: {class_name[2:].strip()}</h2>", unsafe_allow_html=True)
    st.markdown(f"<h3>Confidence Score: {confidence_score:.2f}</h3>", unsafe_allow_html=True)
