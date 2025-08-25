# ===============================================
# Brain Tumor Detection Training Script (TF 2.20)
# ===============================================

import os
import random
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.applications import MobileNetV2
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

# ---------------------------
# 1. Reproducibility
# ---------------------------
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# ---------------------------
# 2. Dataset Path
# ---------------------------
data_dir = r"D:\sarmi\final\Training"

# ---------------------------
# 3. Data Generators
# ---------------------------
img_size = 224
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest"
)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# ---------------------------
# 4. Save class indices
# ---------------------------
with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)
print("Class mapping saved:", train_data.class_indices)

# ---------------------------
# 5. Load MobileNetV2 Base Model
# ---------------------------
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False

# ---------------------------
# 6. Custom Classification Head
# ---------------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
output = Dense(train_data.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# ---------------------------
# 7. Resume from latest checkpoint if exists
# ---------------------------
checkpoints = glob.glob("checkpoint_epoch_*.h5")
if checkpoints:
    latest_ckpt = max(checkpoints, key=os.path.getctime)
    print("Resuming from checkpoint:", latest_ckpt)
    model = load_model(latest_ckpt)
else:
    print("No checkpoint found. Training from scratch.")

# ---------------------------
# 8. Compile Model (Warm-up)
# ---------------------------
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------------------------
# 9. Callbacks (Warm-up)
# ---------------------------
checkpoint_warmup = ModelCheckpoint(
    "checkpoint_warmup_epoch_{epoch:02d}_valacc_{val_accuracy:.2f}.h5",
    monitor="val_accuracy",
    save_best_only=False,
    verbose=1
)

early_stop_warmup = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True
)

# ---------------------------
# 10. Train Frozen Base Model (Warm-up)
# ---------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=[checkpoint_warmup, early_stop_warmup],
    verbose=1
)

# Save model after warm-up
model.save("brain_tumor_mobilenetv2_warmup.h5")
print("✅ Warm-up model saved as brain_tumor_mobilenetv2_warmup.h5")

# ---------------------------
# 11. Fine-Tuning (Unfreeze last 30 layers)
# ---------------------------
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

checkpoint_finetune = ModelCheckpoint(
    "checkpoint_finetune_epoch_{epoch:02d}_valacc_{val_accuracy:.2f}.h5",
    monitor="val_accuracy",
    save_best_only=False,
    verbose=1
)

early_stop_finetune = EarlyStopping(
    monitor="val_accuracy",
    patience=15,  # ensures at least 15 epochs
    restore_best_weights=True
)

history_fine = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    callbacks=[checkpoint_finetune, early_stop_finetune],
    verbose=1
)

# ---------------------------
# 12. Save Final Model
# ---------------------------
model.save("brain_tumor_mobilenetv2_final.h5")
print("✅ Final fine-tuned model saved as brain_tumor_mobilenetv2_final.h5")

# ---------------------------
# 13. Plot Accuracy & Loss
# ---------------------------
plt.figure(figsize=(12, 5))

# Combine metrics
train_acc = history.history["accuracy"] + history_fine.history["accuracy"]
val_acc = history.history["val_accuracy"] + history_fine.history["val_accuracy"]
train_loss = history.history["loss"] + history_fine.history["loss"]
val_loss = history.history["val_loss"] + history_fine.history["val_loss"]

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(train_acc, label="Train Accuracy")
plt.plot(val_acc, label="Val Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(train_loss, label="Train Loss")
plt.plot(val_loss, label="Val Loss")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()
