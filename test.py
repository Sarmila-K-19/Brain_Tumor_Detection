import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import image_dataset_from_directory, img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import json
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ---------------------------
# 1. Paths
# ---------------------------
test_dir = r"D:\sarmi\final\Testing"   # your testing dataset folder
model_path = r"brain_tumor_mobilenetv2_final.h5"
classes_path = r"D:\sarmi\class_indices.json"

# ---------------------------
# 2. Load Model & Class Mapping
# ---------------------------
model = load_model(model_path)
print("✅ Model loaded successfully!")

with open(classes_path, "r") as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}
print("Class mapping:", idx_to_class)

# ---------------------------
# 3. Data Generator for Test Set
# ---------------------------
img_size = 224
batch_size = 32

test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

# ---------------------------
# 4. Evaluate on Test Set
# ---------------------------
loss, acc = model.evaluate(test_data, verbose=1)
print(f"🧪 Test Accuracy: {acc*100:.2f}%")
print(f"🧪 Test Loss: {loss:.4f}")

# ---------------------------
# 5. Confusion Matrix & Classification Report
# ---------------------------
y_pred = model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_data.classes

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8,6))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(class_indices))
plt.xticks(tick_marks, list(class_indices.keys()), rotation=45)
plt.yticks(tick_marks, list(class_indices.keys()))

# Write numbers inside the boxes
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], "d"),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()

print("📊 Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=list(class_indices.keys())))

# ---------------------------
# 6. Single Image Prediction Function
# ---------------------------
def predict_single_image(img_path, target_size=(224, 224)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize
    
    predictions = model.predict(img_array)
    pred_idx = np.argmax(predictions[0])
    pred_class = idx_to_class[pred_idx]
    confidence = predictions[0][pred_idx] * 100
    
    print(f"\n🖼 Image: {os.path.basename(img_path)}")
    print(f"👉 Predicted Tumor Type: {pred_class} ({confidence:.2f}%)\n")
    return pred_class, confidence

# ---------------------------
# 7. Test with a User Input Image
# ---------------------------
test_image = input("Enter image path to test: ")  # e.g. D:/brain/finalim/Test/glioma/Tr-gl_0010.jpg
if os.path.exists(test_image):
    predict_single_image(test_image)
else:
    print("⚠️ File not found. Please check the path!")
