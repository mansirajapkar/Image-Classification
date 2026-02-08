import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Flower Classification App",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Flower Image Classification")
st.write("Upload a flower image and the model will predict its category.")

#Download Model From Google Drive
MODEL_PATH="flower_model.keras"
file_id="1nzhaqBbM6kydWvcUclpPcgjdvza_NCjp"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading Trained Model....Please Wait"):
        url=f"https://drive.google.com/uc?id={file_id}"
        gdown.download(
            url,MODEL_PATH,quiet=True
        )

# ----------------------------
# Load Model
# ----------------------------

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# Class names (same order as training)
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# ----------------------------
# Image Preprocessing Function
# ----------------------------
def preprocess_image(image):
    img_height = 180
    img_width = 180

    image = image.resize((img_width, img_height))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # Create batch
    return img_array

# ----------------------------
# Upload Form
# ----------------------------
with st.form("prediction_form"):
    uploaded_file = st.file_uploader(
        "📤 Upload Flower Image",
        type=["jpg", "jpeg", "png"]
    )
    submit = st.form_submit_button("🔍 Predict", disabled=uploaded_file is None)

# ----------------------------
# Prediction
# ----------------------------
if submit:
    if uploaded_file is None:
        st.warning("⚠️ Please upload an image first.")
    else:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        processed_image = preprocess_image(image)

        predictions = model.predict(processed_image)
        score = tf.nn.softmax(predictions[0])

        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)

        st.success(f"🌼 **Predicted Flower:** {predicted_class}")
        st.info(f"📊 **Confidence:** {confidence:.2f}%")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Mini Project | CNN Image Classification | BCA")
