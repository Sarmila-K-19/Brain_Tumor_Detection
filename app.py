from flask import Flask, render_template, request, redirect, url_for, session, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
import numpy as np
import cv2
import os
from fpdf import FPDF
import requests

# ---------------------------
# 0. Config
# ---------------------------
app = Flask(__name__)
app.secret_key = "supersecretkey"
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Together AI API Key (set in environment or paste directly)
TOGETHER_API_KEY = os.getenv("4edbc02489e711bcd44a83cf10f2d64ced8ccab7644dc2636781aa493f29c79f")  # safer: keep key in env

# ---------------------------
# 1. Load Classification Model
# ---------------------------
CLASS_MODEL_PATH = "brain_tumor_mobilenetv2_final.h5"
CLASSES_PATH = "class_indices.json"

clf_model = load_model(CLASS_MODEL_PATH)
with open(CLASSES_PATH, "r") as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}

# ---------------------------
# 2. Define UNet Segmentation
# ---------------------------
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
        if d3.size()[2:] != e3.size()[2:]:
            d3 = F.interpolate(d3, size=e3.size()[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = F.interpolate(d3, size=e2.size()[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = F.interpolate(d2, size=e1.size()[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = torch.sigmoid(self.out(d1))
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seg_model = UNet().to(device)
seg_model.load_state_dict(torch.load("unet_brain_tumor_final.pth", map_location=device))
seg_model.eval()

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# ---------------------------
# 3. Routes
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            session['uploaded_file'] = filepath
            return redirect(url_for('classification'))
    return render_template("upload.html")

@app.route("/classification")
def classification():
    filepath = session.get("uploaded_file")
    if not filepath:
        return redirect(url_for('upload'))

    image = Image.open(filepath).convert("RGB")
    img_resized = image.resize((224, 224))
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)/255.0
    predictions = clf_model.predict(img_array)
    pred_idx = np.argmax(predictions[0])
    pred_class = idx_to_class[pred_idx]
    confidence = predictions[0][pred_idx]*100
    probs = predictions[0]

    session['pred_class'] = pred_class

    return render_template("classification.html",
                           image_path=filepath,
                           pred_class=pred_class,
                           confidence=f"{confidence:.2f}",
                           class_labels=list(idx_to_class.values()),
                           probs=(probs*100).tolist())

@app.route("/segmentation")
def segmentation():
    filepath = session.get("uploaded_file")
    pred_class = session.get("pred_class")
    if not filepath:
        return redirect(url_for('upload'))

    image = Image.open(filepath).convert("RGB")
    img_np = np.array(image.resize((224, 224)))

    # No tumor detected
    if pred_class.lower() == "notumor":
        tumor_size = 0
        tumor_percentage = 0
        tumor_location = "N/A"
        session['tumor_size'] = tumor_size
        session['tumor_percentage'] = tumor_percentage
        session['tumor_location'] = tumor_location

        return render_template(
            "no_tumor.html",
            image_path=filepath,
            tumor_size=tumor_size,
            tumor_percentage=tumor_percentage,
            tumor_location=tumor_location
        )

    # ---------------------------
    # Tumor segmentation for detected tumor
    # ---------------------------
    input_tensor = img_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = seg_model(input_tensor)
        pred_mask = (output.squeeze().cpu().numpy() > 0.5).astype("uint8")

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        pred_mask.astype("uint8"), connectivity=8
    )

    tumor_location = "N/A"

    if num_labels > 1:
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        tumor_mask = (labels == largest_idx).astype("uint8")
        tumor_size = int(stats[largest_idx, cv2.CC_STAT_AREA])
        total_pixels = img_np.shape[0] * img_np.shape[1]
        tumor_percentage = float((tumor_size / total_pixels) * 100)

        x, y, w, h, area = stats[largest_idx]
        cx, cy = centroids[largest_idx]
        cx, cy = int(cx), int(cy)

        h_img, w_img = img_np.shape[:2]
        horizontal = "left" if cx < w_img / 3 else "right" if cx > 2 * w_img / 3 else "center"
        vertical = "top" if cy < h_img / 3 else "bottom" if cy > 2 * h_img / 3 else "center"

        if vertical == "center" and horizontal == "center":
            tumor_location = "center"
        elif vertical == "top" and horizontal == "left":
            tumor_location = "top left"
        elif vertical == "top" and horizontal == "right":
            tumor_location = "top right"
        elif vertical == "bottom" and horizontal == "left":
            tumor_location = "bottom left"
        elif vertical == "bottom" and horizontal == "right":
            tumor_location = "bottom right"
        elif vertical == "center":
            tumor_location = f"center {horizontal}"
        elif horizontal == "center":
            tumor_location = f"{vertical} center"

        boxed_img = img_np.copy()
        cv2.rectangle(boxed_img, (x, y), (x + w, y + h), (255, 0, 0), 3)

        mask_path = os.path.join(UPLOAD_FOLDER, "tumor_mask.png")
        cv2.imwrite(mask_path, (tumor_mask * 255).astype(np.uint8))

        heatmap = cv2.applyColorMap((tumor_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_np, 0.7, heatmap, 0.3, 0)
        heatmap_path = os.path.join(UPLOAD_FOLDER, "heatmap.png")
        cv2.imwrite(heatmap_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        bbox_path = os.path.join(UPLOAD_FOLDER, "tumor_bbox.png")
        cv2.imwrite(bbox_path, cv2.cvtColor(boxed_img, cv2.COLOR_RGB2BGR))
    else:
        tumor_size = 0
        tumor_percentage = 0
        tumor_location = "N/A"
        mask_path, heatmap_path, bbox_path = None, None, None

    session['tumor_size'] = int(tumor_size)
    session['tumor_percentage'] = float(tumor_percentage)
    session['tumor_location'] = tumor_location

    return render_template(
        "segmentation.html",
        image_path=filepath,
        mask_path=mask_path,
        heatmap_path=heatmap_path,
        bbox_path=bbox_path,
        tumor_size=tumor_size,
        tumor_percentage=f"{tumor_percentage:.2f}",
        tumor_location=tumor_location
    )


# 4. Download Hospital-Style Report
# ---------------------------
@app.route("/download_report")
def download_report():
    filepath = session.get("uploaded_file")
    pred_class = session.get("pred_class")
    tumor_size = session.get("tumor_size", 0)
    tumor_percentage = session.get("tumor_percentage", 0)
    tumor_location = session.get("tumor_location", "N/A")

    mask_path = os.path.join(UPLOAD_FOLDER, "tumor_mask.png")
    heatmap_path = os.path.join(UPLOAD_FOLDER, "heatmap.png")
    bbox_path = os.path.join(UPLOAD_FOLDER, "tumor_bbox.png")

    # Tumor info dictionary
    TUMOR_INFO_SUMMARY = {
        "glioma": "Gliomas are the most common type of primary brain tumor due to abnormal growth of glial cells.",
        "meningioma": "Meningiomas are usually benign tumors arising from meninges but can cause pressure effects.",
        "pituitary": "Pituitary tumors are mostly slow-growing and may impact hormone levels.",
        "notumor": "No tumor was detected in the scan. Brain structure appears normal."
    }
    tumor_info_summary = TUMOR_INFO_SUMMARY.get(pred_class.lower(), "Information not available.")

    # Hospital-style summary
    if pred_class.lower() == "notumor":
        summary_text = "No tumor was detected in the scan. The brain structure appears normal."
        image_paths = [filepath]
    else:
        summary_text = (
            f"The scan analysis indicates a {pred_class} tumor in the brain. "
            f"The tumor size is {tumor_size} pixels, covering approximately {tumor_percentage:.2f}% of the brain area. "
            f"It is located at the {tumor_location} region.\n\n"
            f"Tumor Information: {tumor_info_summary}\n"
            "Medical evaluation and follow-up are recommended for further management."
        )
        image_paths = [filepath, mask_path, heatmap_path, bbox_path]

    # Labels for each image
    image_labels = ["Original Image", "Tumor Mask", "Heatmap Overlay", "Boundary Box"]

    # Generate PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Brain Tumor Analysis Report", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Classification: {pred_class}", ln=True)
    pdf.cell(0, 10, f"Tumor Size: {tumor_size} pixels", ln=True)
    pdf.cell(0, 10, f"Tumor Area Percentage: {tumor_percentage:.2f}%", ln=True)
    pdf.cell(0, 10, f"Tumor Location: {tumor_location}", ln=True)
    pdf.ln(10)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Summary:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, summary_text)
    pdf.ln(5)
    
    # Print image label above each image
    for i, img_path in enumerate(image_paths):
        if img_path and os.path.exists(img_path):
            pdf.add_page()
            pdf.set_font("Arial", "B", 12)
            label = image_labels[i] if i < len(image_labels) else os.path.basename(img_path)
            pdf.cell(0, 10, label, ln=True, align="C")
            pdf.image(img_path, x=45, y=30, w=115, h=115)

    pdf_path = os.path.join(UPLOAD_FOLDER, "tumor_report.pdf")
    pdf.output(pdf_path)

    return send_file(pdf_path, as_attachment=True)


# ---------------------------
# Run App
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
