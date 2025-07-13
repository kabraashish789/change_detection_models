import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import BITChangeDetector
from utils import load_checkpoint, save_checkpoint, check_accuracy, save_predictions_as_imgs
from torch.utils.data import DataLoader
from dataset import CDDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# ----------------------------------------
# Hyperparameters & paths
# ----------------------------------------
LEARNING_RATE = 1.21499e-05
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 40
NUM_WORKERS = 1
IMAGE_SIZE = 512
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_TXT_FILE = "/home/foj-ak/data/train.txt"
TEST_TXT_FILE = "/home/foj-ak/data/test.txt"
CHECKPOINT_PATH = "best_model.pth.tar"

# Enable faster performance if your input shapes are constant
torch.backends.cudnn.benchmark = True

# ----------------------------------------
# Loss Functions
# ----------------------------------------

class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    """
    def __init__(self):
        super(DiceLoss, self).__init__()
    
    def forward(self, preds, targets, smooth=1e-5):
        preds = torch.sigmoid(preds)
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    """
    def __init__(self, alpha=0.827893, gamma=1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_with_logits = nn.BCEWithLogitsLoss()
    
    def forward(self, preds, targets):
        bce_loss = self.bce_with_logits(preds, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# ----------------------------------------
# Data Loading
# ----------------------------------------

def get_loaders(train_txt_file, test_txt_file, batch_size, train_transform, test_transform, num_workers, pin_memory):
    """
    Creates training and test DataLoader objects.
    """
    train_dataset = CDDataset(txt_file=train_txt_file, transform=train_transform)
    test_dataset = CDDataset(txt_file=test_txt_file, transform=test_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers,
        pin_memory=pin_memory, shuffle=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers,
        pin_memory=pin_memory, shuffle=False
    )
    
    return train_loader, test_loader

# ----------------------------------------
# Training Loop
# ----------------------------------------

def train_fn(train_loader, model, optimizer, loss_fn_dice, loss_fn_focal, scaler):
    """
    Training loop for one epoch.
    """
    model.train()
    torch.cuda.empty_cache()
    loop = tqdm(train_loader, desc="Training", leave=True)

    total_loss = 0.0
    num_batches = 0

    for images, masks in loop:
        images = images.float().to(DEVICE)
        masks = masks.float().unsqueeze(1).to(DEVICE)

        with torch.amp.autocast('cuda'):
            predictions = model(images)
            loss_dice = loss_fn_dice(predictions, masks)
            loss_focal = loss_fn_focal(predictions, masks)
            loss = loss_dice + loss_focal

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

        loop.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / num_batches
    return avg_train_loss

# ----------------------------------------
# Validation Loop
# ----------------------------------------

def test_fn(test_loader, model, loss_fn_dice, loss_fn_focal):
    """
    Validation/testing loop for one epoch.
    """
    model.eval()
    total_loss = 0
    num_batches = len(test_loader)
    
    with torch.no_grad():
        loop = tqdm(test_loader, desc="Testing", leave=True)
        for images, masks in loop:
            images = images.float().to(DEVICE)
            masks = masks.float().unsqueeze(1).to(DEVICE)
            
            with torch.amp.autocast('cuda'):
                predictions = model(images)
                loss_dice = loss_fn_dice(predictions, masks)
                loss_focal = loss_fn_focal(predictions, masks)
                loss = loss_dice + loss_focal
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / num_batches
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss

# ----------------------------------------
# Plotting Metrics
# ----------------------------------------

def plot_metrics(train_losses, val_losses, dice_scores, iou_scores, plot_folder="plots"):
    """
    Saves plots of training metrics.
    """
    os.makedirs(plot_folder, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(plot_folder, "loss_plot.png"))
    plt.close()

    # Dice plot
    plt.figure()
    plt.plot(epochs, dice_scores, label='Dice Score', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Dice Score over Epochs")
    plt.legend()
    plt.savefig(os.path.join(plot_folder, "dice_score_plot.png"))
    plt.close()

    # IoU plot
    plt.figure()
    plt.plot(epochs, iou_scores, label='IoU Score', color='purple')
    plt.xlabel("Epoch")
    plt.ylabel("IoU Score")
    plt.title("IoU Score over Epochs")
    plt.legend()
    plt.savefig(os.path.join(plot_folder, "iou_score_plot.png"))
    plt.close()

# ----------------------------------------
# Main training process
# ----------------------------------------

def main():
    # Data augmentations
    train_transform = A.Compose([
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=0, value=0),
        A.RandomBrightnessContrast(p=0.3),
        A.RandomGamma(p=0.3),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=20, p=0.2),
        A.Normalize(mean=[0.5]*6, std=[0.5]*6),
        ToTensorV2(),
    ])

    test_transforms = A.Compose([
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=0, value=0),
        A.Normalize(mean=[0.5]*6, std=[0.5]*6),
        ToTensorV2(),
    ])

    # Initialize BITChangeDetector
    model = BITChangeDetector(
        in_channels=6,
        out_channels=1,
        img_size=IMAGE_SIZE,
        patch_size=4,
        embed_dim=128,
        depths=[2, 2, 2, 2],
        num_heads=4
    ).to(DEVICE)

    # Losses
    loss_fn_dice = DiceLoss()
    loss_fn_focal = FocalLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    train_loader, test_loader = get_loaders(
        TRAIN_TXT_FILE,
        TEST_TXT_FILE,
        BATCH_SIZE,
        train_transform,
        test_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )

    # Load checkpoint if available
    if LOAD_MODEL:
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, weights_only=True)
            load_checkpoint(checkpoint, model)
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("Checkpoint loaded successfully!")
        except FileNotFoundError:
            print("No checkpoint found. Starting fresh.")

    scaler = torch.amp.GradScaler('cuda')
    best_dice_score = 0.0

    train_losses = []
    val_losses = []
    dice_scores = []
    iou_scores = []

    total_start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        epoch_start_time = time.time()

        train_loss = train_fn(train_loader, model, optimizer, loss_fn_dice, loss_fn_focal, scaler)
        val_loss = test_fn(test_loader, model, loss_fn_dice, loss_fn_focal)
        scheduler.step(val_loss)

        # Updated: check_accuracy now returns dice_score and iou_score
        dice_score, iou_score = check_accuracy(
            test_loader, model, device=DEVICE, save_folder="debug_plots/"
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        dice_scores.append(dice_score)
        iou_scores.append(iou_score)

        print(f"Epoch Time: {(time.time() - epoch_start_time):.2f} seconds")

        if dice_score > best_dice_score:
            best_dice_score = dice_score
            print(f"New best Dice Score: {best_dice_score:.4f}. Saving model checkpoint.")
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            save_checkpoint(checkpoint, filename=CHECKPOINT_PATH)
        
        save_predictions_as_imgs(test_loader, model, folder="saved_images/", device=DEVICE)

    total_time = time.time() - total_start_time
    print(f"\nTotal Training Time: {total_time/60:.2f} minutes")

    # Save training plots
    plot_metrics(train_losses, val_losses, dice_scores, iou_scores, plot_folder="plots")


if __name__ == "__main__":
    main()
