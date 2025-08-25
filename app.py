import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import json
import matplotlib
import matplotlib.pyplot as plt

# Use Agg backend for Streamlit (safe for server apps)
matplotlib.use("Agg")

# ---------------------------
# 1. Paths
# ---------------------------
MODEL_PATH = r"brain_tumor_mobilenetv2_final.h5"
CLASSES_PATH = r"class_indices.json"

st.title("🧠 Brain Tumor Classification App")

# ---------------------------
# 2. Load Model & Classes
# ---------------------------
@st.cache_resource
def load_brain_model():
    model = load_model(MODEL_PATH)
    with open(CLASSES_PATH, "r") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class

model, idx_to_class = load_brain_model()
st.success("✅ Model loaded successfully!")

# ---------------------------
# 3. File Uploader
# ---------------------------
uploaded_file = st.file_uploader("📤 Upload an MRI Image", type=["jpg", "jpeg", "png"])

# ---------------------------
# 4. Prediction Function
# ---------------------------
def predict_single_image(img, target_size=(224, 224)):
    # Resize and preprocess
    img = img.resize(target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize

    # Prediction
    predictions = model.predict(img_array)
    pred_idx = np.argmax(predictions[0])
    pred_class = idx_to_class[pred_idx]
    confidence = predictions[0][pred_idx] * 100
    return pred_class, confidence, predictions[0]

# ---------------------------
# 5. Handle Uploaded Image
# ---------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼 Uploaded MRI Image", use_column_width=True)

    pred_class, confidence, probs = predict_single_image(image)

    st.subheader("🔍 Prediction Result")
    st.write(f"👉 **Predicted Tumor Type:** {pred_class}")
    st.metric(label="📊 Confidence", value=f"{confidence:.2f}%")

    # Probability bar chart
    st.subheader("📈 Probability Distribution")
    class_labels = [idx_to_class[i] for i in range(len(idx_to_class))]
    fig, ax = plt.subplots()
    ax.bar(class_labels, probs * 100, color="skyblue")
    ax.set_ylabel("Confidence (%)")
    ax.set_xlabel("Classes")
    ax.set_title("Prediction Probabilities")
    st.pyplot(fig)
