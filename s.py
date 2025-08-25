import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import json
import matplotlib
import matplotlib.pyplot as plt

# Use Agg backend for matplotlib
matplotlib.use("Agg")

# ======================================================
# 1. Classification Model (Keras)
# ======================================================
CLASS_MODEL_PATH = "brain_tumor_mobilenetv2_final.h5"
CLASSES_PATH = "class_indices.json"

@st.cache_resource
def load_classification_model():
    model = load_model(CLASS_MODEL_PATH)
    with open(CLASSES_PATH, "r") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class

clf_model, idx_to_class = load_classification_model()

# ======================================================
# 2. Segmentation Model (PyTorch UNet)
# ======================================================
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.enc1 = CBR(in_channels, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.ConvTranspose2d(512, 256, 2, stride=2)

        self.dec3 = CBR(512, 256)
        self.dec2 = CBR(384, 128)
        self.dec1 = CBR(192, 64)
        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d3 = self.up(e4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = nn.functional.interpolate(d3, scale_factor=2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = nn.functional.interpolate(d2, scale_factor=2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = torch.sigmoid(self.out(d1))
        return out

# Load UNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seg_model = UNet().to(device)
seg_model.load_state_dict(torch.load("unet_brain_tumor_final.pth", map_location=device))
seg_model.eval()

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# ======================================================
# 3. Streamlit UI
# ======================================================
st.title("🧠 Brain Tumor Classification + Segmentation")

uploaded_file = st.file_uploader("📤 Upload an MRI Image", type=["jpg", "jpeg", "png"])

# ======================================================
# 4. Prediction Functions
# ======================================================
def predict_class(img, target_size=(224, 224)):
    img_resized = img.resize(target_size)
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = clf_model.predict(img_array)
    pred_idx = np.argmax(predictions[0])
    pred_class = idx_to_class[pred_idx]
    confidence = predictions[0][pred_idx] * 100
    return pred_class, confidence, predictions[0]

def segment_tumor(image):
    input_tensor = img_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = seg_model(input_tensor)
        pred_mask = (output.squeeze().cpu().numpy() > 0.5).astype("uint8")
    return pred_mask

# ======================================================
# 5. Process Uploaded Image
# ======================================================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼 Uploaded MRI Image", use_column_width=True)

    # ---- Classification ----
    st.subheader("🔍 Classification Result")
    pred_class, confidence, probs = predict_class(image)
    st.write(f"👉 **Predicted Tumor Type:** {pred_class}")
    st.metric(label="📊 Confidence", value=f"{confidence:.2f}%")

    # Probability chart
    class_labels = [idx_to_class[i] for i in range(len(idx_to_class))]
    fig, ax = plt.subplots()
    ax.bar(class_labels, probs * 100, color="skyblue")
    ax.set_ylabel("Confidence (%)")
    ax.set_xlabel("Classes")
    ax.set_title("Prediction Probabilities")
    st.pyplot(fig)

    # ---- Segmentation (only if tumor detected) ----
    if pred_class.lower() != "notumor":
        st.subheader("🩺 Tumor Segmentation & Analysis")
        pred_mask = segment_tumor(image)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_mask.astype("uint8"), connectivity=8)

        if num_labels > 1:  # tumors found
            largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            tumor_mask = (labels == largest_idx).astype("uint8")

            tumor_size = stats[largest_idx, cv2.CC_STAT_AREA]

            # -----------------------------
            # Region-based Tumor Location
            # -----------------------------
            mask_h, mask_w = pred_mask.shape
            center_w = int(mask_w * 0.3)
            center_h = int(mask_h * 0.3)
            center_x0 = (mask_w - center_w) // 2
            center_y0 = (mask_h - center_h) // 2

            regions = {
                "Center": pred_mask[center_y0:center_y0+center_h, center_x0:center_x0+center_w],
                "Top Left": pred_mask[0:center_y0, 0:center_x0],
                "Top Right": pred_mask[0:center_y0, center_x0+center_w:mask_w],
                "Bottom Left": pred_mask[center_y0+center_h:mask_h, 0:center_x0],
                "Bottom Right": pred_mask[center_y0+center_h:mask_h, center_x0+center_w:mask_w]
            }

            region_counts = {k: np.sum(v) for k,v in regions.items()}
            tumor_location = max(region_counts, key=region_counts.get)

            # -----------------------------
            # Display Results
            # -----------------------------
            st.write(f"🟢 Tumor Size (pixels): **{tumor_size}**")
            st.write(f"📍 Tumor Location: **{tumor_location}**")

            # Heatmap
            img_np = np.array(image)
            heatmap = cv2.applyColorMap((tumor_mask*255).astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img_np, 0.7, heatmap, 0.3, 0)
            st.subheader("🔥 Tumor Heatmap")
            st.image(overlay, caption="Tumor Heatmap", use_column_width=True)

        else:
            st.warning("⚠️ No tumor region detected in segmentation.")
