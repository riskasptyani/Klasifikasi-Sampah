import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image

# Page config
st.set_page_config(page_title="Klasifikasi Sampah", page_icon="🗑️", layout="centered")

# Custom CSS
st.markdown(
    """
    <style>
        body {
            background-color: white !important;
            color: #000000;
        }
        .stApp {
            background-color: white;
        }
        h1 {
            color: #333 !important;
            font-weight: bold;
            text-align: center;
        }
        .description {
            text-align: justify;
            font-size: 1.05em;
            padding-left: 20px;
        }
        /* ⬇️ Warna nama file dan ukuran */
        span[data-testid="uploaded-file-name"],
        span[data-testid="uploaded-file-size"] {
            color: black !important;
        }
    </style>
    """,

    unsafe_allow_html=True
)

# Judul
st.markdown("<h1>🗑️ Klasifikasi Gambar Sampah</h1>", unsafe_allow_html=True)

# Gambar kiri - Deskripsi kanan
col1, col2 = st.columns([1, 2])

with col1:
    st.image("HALO.jpg", width=240)

with col2:
    st.markdown(
    """
    <div class="description" style="color:#333; text-align: justify; font-size: 1.05em; padding-left: 20px;">
    Sistem Klasifikasi Sampah ini adalah sebuah aplikasi berbasis web yang dirancang khusus untuk membantu para pengumpul dalam mengenali dan membedakan lima jenis sampah utama, yaitu kardus, kertas, logam, kaca, dan plastik. 
    Aplikasi ini menggunakan model Convolutional Neural Network (CNN) untuk memprediksi jenis sampah berdasarkan gambar yang diunggah. 
    Model telah dilatih menggunakan TensorFlow dan Keras agar mampu memberikan hasil klasifikasi yang akurat dan cepat.
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown("---")

# ⬇️ Load model hanya sekali
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("klasifikasi_sampah.h5")

model = load_model()

st.markdown("<p style='color: black; font-weight: 500;'>📸 Upload gambar sampah</p>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(label="", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diunggah", use_column_width=True)

    with st.spinner("🔍 Mengklasifikasikan..."):
        img_resized = img.resize((model.input_shape[1], model.input_shape[2]))
        img_array = keras_image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction[0])
        class_names = ["kaca", "kardus", "kertas", "logam", "plastik"]
        predicted_label = class_names[predicted_index]

    st.markdown(
    f"""
    <div style='background-color: #dfffe0; padding: 15px; border-radius: 10px;'>
        <span style='color: black; font-weight: bold; font-size: 1.1em;'>Hasil Prediksi:</span>
        <span style='color: black; font-weight: 700; font-size: 1.1em;'> {predicted_label.upper()} </span>
    </div>
    """,
    unsafe_allow_html=True
)
