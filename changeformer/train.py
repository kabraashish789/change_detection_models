import os
import time
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import ChangeFormerV6
from utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    save_predictions_as_imgs,
)
from torch.utils.data import DataLoader
from dataset import CDDataset
import matplotlib
matplotlib.use('Agg')  # For headless environments like Docker
import matplotlib.pyplot as plt

# ------------------------------------------
# Hyperparameters
# ------------------------------------------

LEARNING_RATE = 8.07369e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 40
NUM_WORKERS = 3
IMAGE_SIZE = 256
PIN_MEMORY = True
LOAD_MODEL = True

TRAIN_TXT_FILE = "/home/foj-ak/data/train.txt"
TEST_TXT_FILE = "/home/foj-ak/data/test.txt"
CHECKPOINT_PATH = "best_model.pth.tar"

torch.backends.cudnn.benchmark = True

# ------------------------------------------
# Loss Functions
# ------------------------------------------

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, preds, targets, smooth=8.68709e-5):
        preds = torch.sigmoid(preds)
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.988362, gamma=3):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_with_logits = nn.BCEWithLogitsLoss()
    
    def forward(self, preds, targets):
        bce_loss = self.bce_with_logits(preds, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# ------------------------------------------
# DataLoader utility
# ------------------------------------------

def get_loaders(train_txt_file, test_txt_file, batch_size, train_transform, test_transform, num_workers, pin_memory):
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

# ------------------------------------------
# Training loop
# ------------------------------------------

def train_fn(train_loader, model, optimizer, loss_fn_dice, loss_fn_focal, scaler):
    model.train()
    torch.cuda.empty_cache()

    loop = tqdm(train_loader, desc="Training", leave=True)
    total_loss = 0.0
    num_batches = 0

    for img_before, img_after, masks in loop:
        img_before = img_before.float().to(DEVICE)
        img_after = img_after.float().to(DEVICE)
        masks = masks.float().unsqueeze(1).to(DEVICE)

        with torch.amp.autocast('cuda'):
            predictions = model(img_before, img_after)
            
            # Deep supervision (if model returns list of predictions)
            if isinstance(predictions, list):
                loss = sum(
                    loss_fn_dice(pred, masks) + loss_fn_focal(pred, masks)
                    for pred in predictions
                ) / len(predictions)
            else:
                loss = loss_fn_dice(predictions, masks) + loss_fn_focal(predictions, masks)

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

# ------------------------------------------
# Validation loop
# ------------------------------------------

def test_fn(test_loader, model, loss_fn_dice, loss_fn_focal):
    model.eval()
    total_loss = 0.0
    num_batches = len(test_loader)

    with torch.no_grad():
        loop = tqdm(test_loader, desc="Testing", leave=True)
        for img_before, img_after, masks in loop:
            img_before = img_before.float().to(DEVICE)
            img_after = img_after.float().to(DEVICE)
            masks = masks.float().unsqueeze(1).to(DEVICE)

            with torch.amp.autocast('cuda'):
                predictions = model(img_before, img_after)

                if isinstance(predictions, list):
                    loss = sum(
                        loss_fn_dice(pred, masks) + loss_fn_focal(pred, masks)
                        for pred in predictions
                    ) / len(predictions)
                else:
                    loss = loss_fn_dice(predictions, masks) + loss_fn_focal(predictions, masks)

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / num_batches
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

# ------------------------------------------
# Plotting utility
# ------------------------------------------

def plot_metrics(train_losses, val_losses, dice_scores, iou_scores, plot_folder="plots"):
    os.makedirs(plot_folder, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(plot_folder, "loss_plot.png"))
    plt.close()

    # Dice score plot
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
    plt.plot(epochs, iou_scores, label='IoU Score', color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("IoU Score")
    plt.title("IoU Score over Epochs")
    plt.legend()
    plt.savefig(os.path.join(plot_folder, "iou_score_plot.png"))
    plt.close()

# ------------------------------------------
# Main training pipeline
# ------------------------------------------

def main():
    # Define training transforms
    train_transform = A.Compose([
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=0, value=0),
        A.RandomBrightnessContrast(p=0.3),
        A.RandomGamma(p=0.3),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=20, p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ], additional_targets={"image_t1": "image"})

    # Define test transforms
    test_transforms = A.Compose([
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=0, value=0),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ], additional_targets={"image_t1": "image"})

    # Instantiate model
    model = ChangeFormerV6(in_channels=3, num_classes=1).to(DEVICE)

    loss_fn_dice = DiceLoss()
    loss_fn_focal = FocalLoss()

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    train_loader, test_loader = get_loaders(
        TRAIN_TXT_FILE, TEST_TXT_FILE, BATCH_SIZE,
        train_transform, test_transforms, NUM_WORKERS, PIN_MEMORY
    )

    if LOAD_MODEL:
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
            load_checkpoint(checkpoint, model)
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("Checkpoint loaded successfully!")
        except FileNotFoundError:
            print("No checkpoint found. Starting fresh training.")

    scaler = torch.amp.GradScaler('cuda')

    best_dice_score = 0.0
    train_losses = []
    val_losses = []
    dice_scores = []
    iou_scores = []

    total_start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        epoch_start_time = time.time()

        train_loss = train_fn(train_loader, model, optimizer, loss_fn_dice, loss_fn_focal, scaler)
        val_loss = test_fn(test_loader, model, loss_fn_dice, loss_fn_focal)
        scheduler.step(val_loss)

        # Get dice and iou from updated utils.py
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
            print(f"New best Dice score: {best_dice_score:.4f}. Saving checkpoint.")
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=CHECKPOINT_PATH)

    total_time = time.time() - total_start_time
    print(f"\nTotal training time: {total_time/60:.2f} minutes")

    # Plot metrics
    plot_metrics(train_losses, val_losses, dice_scores, iou_scores, plot_folder="plots")

    # Optional: Save predictions
    # save_predictions_as_imgs(test_loader, model, folder="saved_images/", device=DEVICE)

if __name__ == "__main__":
    main()
