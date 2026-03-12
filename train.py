import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm

from models.miniunet import MiniUNet
from utils.losses import HybridLoss


# Dataset
class PolypSegDataset(Dataset):

    def __init__(self, image_dir, mask_dir, image_size=256):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.image_size = image_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = TF.resize(image,(self.image_size,self.image_size))
        mask = TF.resize(mask,(self.image_size,self.image_size),
                         interpolation=TF.InterpolationMode.NEAREST)

        if random.random()>0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        angle = random.uniform(-15,15)
        image = TF.rotate(image,angle)
        mask = TF.rotate(mask,angle)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        mask = (mask>0).float()

        return image,mask


# Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = torch.cuda.is_available()

model = MiniUNet().to(device)

criterion = HybridLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


# Training Loop
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    total_iou = 0
    total_dice = 0

    pbar = tqdm(loader, desc="Training", leave=False)

    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # metrics
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        inter = (preds * masks).sum()
        union = (preds + masks - preds * masks).sum()
        iou = (inter + 1e-6) / (union + 1e-6)
        dice = (2*inter + 1e-6) / (preds.sum() + masks.sum() + 1e-6)

        total_loss += loss.item()
        total_iou += iou.item()
        total_dice += dice.item()

        pbar.set_postfix(loss=loss.item(), IoU=iou.item(), Dice=dice.item())

    n = len(loader)
    return total_loss/n, total_iou/n, total_dice/n


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    total_iou = 0
    total_dice = 0

    for images, masks in tqdm(loader, desc="Validation", leave=False):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)

        # metrics
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        inter = (preds * masks).sum()
        union = (preds + masks - preds * masks).sum()
        iou = (inter + 1e-6) / (union + 1e-6)
        dice = (2*inter + 1e-6) / (preds.sum() + masks.sum() + 1e-6)

        total_loss += loss.item()
        total_iou += iou.item()
        total_dice += dice.item()

    n = len(loader)
    return total_loss/n, total_iou/n, total_dice/n


train_dataset = PolypSegDataset( image_dir="dataset/train/images", mask_dir="dataset/train/masks", image_size=256 ) 
val_dataset = PolypSegDataset( image_dir="dataset/val/images", mask_dir="dataset/val/masks", image_size=256 )

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

num_epochs = 60
best_val_dice = 0.0
save_path = "best_model.pth"


train_losses = []
val_losses = []

train_dices = []
val_dices = []

train_ious = []
val_ious = []


for epoch in range(num_epochs):

    train_loss, train_iou, train_dice = train_one_epoch(
        model, train_loader, optimizer, criterion)

    val_loss, val_iou, val_dice = validate(
        model, val_loader, criterion)

    scheduler.step()

    # save history
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_dices.append(train_dice)
    val_dices.append(val_dice)

    train_ious.append(train_iou)
    val_ious.append(val_iou)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f} | IoU: {train_iou:.4f} | Dice: {train_dice:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | Dice: {val_dice:.4f}")
    print("-"*50)

    if val_dice > best_val_dice:
        best_val_dice = val_dice

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_dice": val_dice,
        }, save_path)

        print("✅ Best model saved!")

        
# Plot Loss Curve        
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plot Dice Curve
plt.figure(figsize=(8,5))
plt.plot(train_dices, label="Train Dice")
plt.plot(val_dices, label="Validation Dice")
plt.title("Dice Score")
plt.xlabel("Epoch")
plt.ylabel("Dice")
plt.legend()
plt.show()


# Show Predictions
checkpoint = torch.load(save_path)
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

images, masks = next(iter(val_loader))
images = images.to(device)

with torch.no_grad():
    outputs = model(images)

preds = (torch.sigmoid(outputs) > 0.5).float().cpu()
images = images.cpu()
masks = masks.cpu()

num_samples = images.shape[0]

for i in range(num_samples):

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(images[i].permute(1,2,0))
    plt.title("Image")
    plt.subplot(1,3,2)
    plt.imshow(masks[i].squeeze(), cmap="gray")
    plt.title("Ground Truth")
    plt.subplot(1,3,3)
    plt.imshow(preds[i].squeeze(), cmap="gray")
    plt.title("Prediction")
    plt.show()