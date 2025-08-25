'''import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# -----------------------------
# 1. Dataset Class
# -----------------------------
class BrainTumorSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = os.listdir(image_dir)
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        mask_name = img_name.replace(".jpg", ".png")
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

# -----------------------------
# 2. Transforms
# -----------------------------
img_transform = transforms.Compose([
    transforms.ToTensor(),               
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

mask_transform = transforms.Compose([
    transforms.ToTensor(),               
])

# -----------------------------
# 3. Dataset & DataLoader
# -----------------------------
train_dataset = BrainTumorSegmentationDataset(
    image_dir="processed/train/images",
    mask_dir="processed/train/masks",
    transform=img_transform,
    mask_transform=mask_transform
)

val_dataset = BrainTumorSegmentationDataset(
    image_dir="processed/test/images",
    mask_dir="processed/test/masks",
    transform=img_transform,
    mask_transform=mask_transform
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# -----------------------------
# 4. U-Net Model
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
# 5. Training Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
criterion = nn.BCELoss()         
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 12
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# -----------------------------
# 6. Training Loop with Checkpoints
# -----------------------------
for epoch in range(1, num_epochs+1):
    print(f"\nEpoch {epoch}/{num_epochs}")
    print("-" * 30)

    # ---- Training ----
    model.train()
    train_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_loader.dataset)
    print(f"Training Loss: {train_loss:.4f}")

    # ---- Validation ----
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)
    val_loss /= len(val_loader.dataset)
    print(f"Validation Loss: {val_loss:.4f}")

    # ---- Save Checkpoint ----
    checkpoint_path = os.path.join(checkpoint_dir, f"unet_epoch_{epoch}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

# -----------------------------
# Save Final Model
# -----------------------------
final_model_path = "unet_brain_tumor_final.pth"
torch.save(model.state_dict(), final_model_path)
print(f"\nTraining complete. Final model saved: {final_model_path}")
'''
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# -----------------------------
# 1. Dataset Class
# -----------------------------
class BrainTumorSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = os.listdir(image_dir)
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        mask_name = img_name.replace(".jpg", ".png")
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

# -----------------------------
# 2. Transforms
# -----------------------------
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

mask_transform = transforms.Compose([
    transforms.ToTensor(),
])

# -----------------------------
# 3. Dataset & DataLoader
# -----------------------------
train_dataset = BrainTumorSegmentationDataset(
    image_dir="processed/train/images",
    mask_dir="processed/train/masks",
    transform=img_transform,
    mask_transform=mask_transform
)

val_dataset = BrainTumorSegmentationDataset(
    image_dir="processed/test/images",
    mask_dir="processed/test/masks",
    transform=img_transform,
    mask_transform=mask_transform
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# -----------------------------
# 4. U-Net Model
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
# 5. Training Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 12
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# -----------------------------
# 6. Training Loop with tqdm & Checkpoints
# -----------------------------
for epoch in range(1, num_epochs+1):
    print(f"\nEpoch {epoch}/{num_epochs}")
    print("-" * 30)

    # ---- Training ----
    model.train()
    train_loss = 0.0
    loop = tqdm(train_loader, total=len(train_loader), ncols=100, desc="Training")
    for images, masks in loop:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        loop.set_postfix(loss=loss.item())

    train_loss /= len(train_loader.dataset)
    print(f"Training Loss (epoch): {train_loss:.4f}")

    # ---- Validation ----
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        loop_val = tqdm(val_loader, total=len(val_loader), ncols=100, desc="Validation")
        for images, masks in loop_val:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)
            loop_val.set_postfix(loss=loss.item())

    val_loss /= len(val_loader.dataset)
    print(f"Validation Loss (epoch): {val_loss:.4f}")

    # ---- Save Checkpoint ----
    checkpoint_path = os.path.join(checkpoint_dir, f"unet_epoch_{epoch}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

# -----------------------------
# Save Final Model
# -----------------------------
final_model_path = "unet_brain_tumor_final.pth"
torch.save(model.state_dict(), final_model_path)
print(f"\nTraining complete. Final model saved: {final_model_path}")
