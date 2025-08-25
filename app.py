import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------
# Define U-Net model
# -----------------------------
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

# -----------------------------
# Load Model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load("unet_brain_tumor_final.pth", map_location=device))
model.eval()

# -----------------------------
# Image Transform
# -----------------------------
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🧠 Brain Tumor Segmentation & Analysis")

uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    # Preprocess
    input_tensor = img_transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = (output.squeeze().cpu().numpy() > 0.5).astype("uint8")

    # -----------------------------
    # Process Tumor Mask
    # -----------------------------
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_mask.astype("uint8"), connectivity=8)

    if num_labels > 1:  # background + tumors
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # ignore background
        tumor_mask = (labels == largest_idx).astype("uint8")

        tumor_size = stats[largest_idx, cv2.CC_STAT_AREA]
        x, y, w, h, _ = stats[largest_idx]
        cx, cy = centroids[largest_idx]

        st.subheader("📊 Tumor Analysis")
        st.write(f"🟢 Tumor Size (pixels): **{tumor_size}**")
        st.write(f"📍 Tumor Location (Bounding Box): x={x}, y={y}, w={w}, h={h}")
        st.write(f"🎯 Tumor Centroid: ({int(cx)}, {int(cy)})")

        # -----------------------------
        # Heatmap Overlay
        # -----------------------------
        img_np = np.array(image)
        heatmap = cv2.applyColorMap((tumor_mask*255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_np, 0.7, heatmap, 0.3, 0)

        st.subheader("🔥 Heatmap Visualization")
        st.image(overlay, caption="Tumor Heatmap", use_column_width=True)

    else:
        st.warning("No tumor detected in this image.")
