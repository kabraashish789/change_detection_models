import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

def save_checkpoint(state, filename="best_model.pth.tar"):
    """Save model checkpoint to file."""
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    """Load model checkpoint into given model."""
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def check_accuracy(loader, model, device, save_folder="debug_plots/"):
    """
    Evaluates the model on the given data loader and calculates:
      - Pixel accuracy
      - Dice score
      - IoU score
    Also saves debug plots for the first few samples.

    Returns:
        avg_dice_score (float)
        avg_iou_score (float)
    """
    model.eval()
    os.makedirs(save_folder, exist_ok=True)

    total_correct = 0
    total_pixels = 0
    total_dice = 0.0
    total_iou = 0.0
    samples_saved = 0

    with torch.no_grad():
        for img_before, img_after, y in loader:
            img_before = img_before.to(device)
            img_after = img_after.to(device)
            y = y.to(device).unsqueeze(1).float()  # (B, 1, H, W)

            preds = model(img_before, img_after)
            preds_prob = torch.sigmoid(preds)
            preds_bin = (preds_prob > 0.75).float()

            # Pixel-wise accuracy
            total_correct += (preds_bin == y).sum().item()
            total_pixels += y.numel()

            # Dice and IoU calculation
            intersection = (preds_bin * y).sum(dim=(1, 2, 3))
            union = preds_bin.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
            iou_denominator = preds_bin.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3)) - intersection + 1e-8

            dice_batch = (2. * intersection + 1e-8) / (union + 1e-8)
            iou_batch = (intersection + 1e-8) / iou_denominator

            total_dice += dice_batch.mean().item()
            total_iou += iou_batch.mean().item()

            # Save debug plots for the first few samples
            for i in range(min(img_before.size(0), 5 - samples_saved)):
                save_debug_plot(
                    img_before[i],
                    img_after[i],
                    y[i],
                    preds_prob[i],
                    preds_bin[i],
                    index=samples_saved,
                    folder=save_folder,
                    normalization_mean=[0.5, 0.5, 0.5],
                    normalization_std=[0.5, 0.5, 0.5]
                )
                samples_saved += 1

    avg_dice_score = total_dice / len(loader)
    avg_iou_score = total_iou / len(loader)
    accuracy = 100 * total_correct / total_pixels

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Dice Score: {avg_dice_score:.4f}")
    print(f"IoU Score: {avg_iou_score:.4f}")

    model.train()
    return avg_dice_score, avg_iou_score

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    """
    Save model predictions and ground truths as images.
    """
    model.eval()
    os.makedirs(folder, exist_ok=True)

    with torch.no_grad():
        for idx, (img_before, img_after, y) in enumerate(loader):
            img_before = img_before.to(device)
            img_after = img_after.to(device)
            preds = torch.sigmoid(model(img_before, img_after))
            preds_bin = (preds > 0.5).float()

            # Save predictions and ground truths
            torchvision.utils.save_image(preds_bin, f"{folder}/pred_{idx}.png")
            torchvision.utils.save_image(y.unsqueeze(1).float(), f"{folder}/gt_{idx}.png")

    model.train()

def save_debug_plot(img_before, img_after, ground_truth, prediction, pred_thresh, index=0, folder="debug_plots/", normalization_mean=None, normalization_std=None):
    """
    Save a side-by-side debug plot showing:
        - Before Image
        - After Image
        - Ground Truth Mask
        - Predicted Probability Mask
        - Thresholded Prediction
    """
    # Convert tensors to CPU NumPy arrays
    img_before = img_before.cpu().numpy()
    img_after = img_after.cpu().numpy()
    ground_truth = ground_truth.cpu().squeeze().numpy()
    prediction = prediction.cpu().squeeze().numpy()
    pred_thresh = pred_thresh.cpu().squeeze().numpy()

    # Convert from (C, H, W) to (H, W, C)
    img_before = np.transpose(img_before, (1, 2, 0))
    img_after = np.transpose(img_after, (1, 2, 0))

    # Denormalize images if needed
    if normalization_mean is not None and normalization_std is not None:
        normalization_mean = np.array(normalization_mean)
        normalization_std = np.array(normalization_std)
        img_before = img_before * normalization_std + normalization_mean
        img_after = img_after * normalization_std + normalization_mean

    img_before = np.clip(img_before, 0, 1)
    img_after = np.clip(img_after, 0, 1)

    # Plot
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    axes[0].imshow(img_before)
    axes[0].set_title("Before Image")
    axes[0].axis("off")

    axes[1].imshow(img_after)
    axes[1].set_title("After Image")
    axes[1].axis("off")

    axes[2].imshow(ground_truth, cmap="gray")
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")

    axes[3].imshow(prediction, cmap="gray")
    axes[3].set_title("Prediction (Probabilities)")
    axes[3].axis("off")

    axes[4].imshow(pred_thresh, cmap="gray")
    axes[4].set_title("Prediction (Thresholded)")
    axes[4].axis("off")

    plt.tight_layout()
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, f"debug_plot_{index}.png"))
    plt.close(fig)
