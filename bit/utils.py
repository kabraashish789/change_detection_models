import torch
import torchvision
import os
import matplotlib.pyplot as plt
import numpy as np

def save_checkpoint(state, filename="best_model.pth.tar"):
    """
    Save model and optimizer state.
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    """
    Load model weights from checkpoint.
    """
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def check_accuracy(loader, model, device, save_folder="debug_plots/"):
    """
    Evaluate the model on the dataset:
    - Computes average Dice score
    - Computes average IoU (Jaccard Index)
    - Optionally saves debug plots for visualization

    Args:
        loader: DataLoader for the dataset
        model: segmentation model
        device: device to run evaluation
        save_folder: where to save debug plots

    Returns:
        avg_dice_score: Average Dice coefficient
        avg_iou_score: Average Intersection over Union
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    iou_score = 0

    model.eval()

    os.makedirs(save_folder, exist_ok=True)
    samples_saved = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)  # add channel dim if missing

            preds = model(x)
            preds_sigmoid = torch.sigmoid(preds)
            preds_thresh = (preds_sigmoid > 0.75).float()

            # Pixel-wise accuracy
            num_correct += (preds_thresh == y).sum().item()
            num_pixels += torch.numel(preds_thresh)

            # Dice coefficient
            intersection = (preds_thresh * y).sum()
            union = preds_thresh.sum() + y.sum()
            dice = (2 * intersection) / (union + 1e-8) if union > 0 else torch.tensor(0.0, device=device)
            dice_score += dice.item()

            # IoU (Jaccard Index)
            intersection = (preds_thresh * y).sum()
            union = ((preds_thresh + y) > 0).float().sum()
            iou = intersection / (union + 1e-8) if union > 0 else torch.tensor(0.0, device=device)
            iou_score += iou.item()

            # Save debug plots for the first few samples
            for i in range(x.size(0)):
                save_debug_plot(
                    input_image=x[i],
                    ground_truth=y[i],
                    prediction=preds_sigmoid[i],
                    pred_thresh=preds_thresh[i],
                    index=samples_saved,
                    folder=save_folder,
                    normalization_mean=[0.5, 0.5, 0.5],
                    normalization_std=[0.5, 0.5, 0.5]
                )
                samples_saved += 1

    accuracy = (num_correct / num_pixels) * 100
    avg_dice_score = dice_score / len(loader)
    avg_iou_score = iou_score / len(loader)

    print(f"Pixel Accuracy: {accuracy:.2f}%")
    print(f"Average Dice score: {avg_dice_score:.4f}")
    print(f"Average IoU score: {avg_iou_score:.4f}")

    model.train()
    return avg_dice_score, avg_iou_score

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    os.makedirs(folder, exist_ok=True)

    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")

        # Normalize y to [0,1] for saving
        y = y.unsqueeze(1).float()
        torchvision.utils.save_image(y, f"{folder}/{idx}.png")

    model.train()

def save_debug_plot(input_image, ground_truth, prediction, pred_thresh, index=0, folder="debug_plots/", normalization_mean=None, normalization_std=None):
    """
    Saves a plot showing:
    - Before and After images (split from input)
    - Ground truth mask
    - Predicted mask probabilities
    - Thresholded binary mask

    Args:
        input_image: input tensor with stacked before and after images (shape [6,H,W])
        ground_truth: true mask
        prediction: predicted mask probabilities
        pred_thresh: binary thresholded prediction
        index: sample index for filename
        folder: directory to save plots
        normalization_mean: list of mean values for normalization
        normalization_std: list of std values for normalization
    """

    input_image = input_image.cpu().numpy()
    ground_truth = ground_truth.cpu().squeeze().numpy()
    prediction = prediction.cpu().squeeze().numpy()
    pred_thresh = pred_thresh.cpu().squeeze().numpy()

    # Separate before and after images
    num_channels = input_image.shape[0]
    channels_per_image = num_channels // 2

    before_image = input_image[:channels_per_image, :, :]
    after_image = input_image[channels_per_image:, :, :]

    # Convert to HWC format
    before_image = np.transpose(before_image, (1, 2, 0))
    after_image = np.transpose(after_image, (1, 2, 0))

    # Denormalize if needed
    if normalization_mean and normalization_std:
        normalization_mean = np.array(normalization_mean)
        normalization_std = np.array(normalization_std)
        if len(normalization_mean) == num_channels:
            mean_before = normalization_mean[:channels_per_image]
            std_before = normalization_std[:channels_per_image]
            mean_after = normalization_mean[channels_per_image:]
            std_after = normalization_std[channels_per_image:]
        else:
            mean_before = mean_after = normalization_mean
            std_before = std_after = normalization_std

        before_image = before_image * std_before + mean_before
        after_image = after_image * std_after + mean_after

    before_image = np.clip(before_image, 0, 1)
    after_image = np.clip(after_image, 0, 1)

    # Create the figure
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    axes[0].imshow(before_image)
    axes[0].set_title("Before Image")
    axes[0].axis("off")

    axes[1].imshow(after_image)
    axes[1].set_title("After Image")
    axes[1].axis("off")

    axes[2].imshow(ground_truth, cmap="gray")
    axes[2].set_title("Ground Truth Mask")
    axes[2].axis("off")

    axes[3].imshow(prediction, cmap="gray")
    axes[3].set_title("Predicted Mask (Probabilities)")
    axes[3].axis("off")

    axes[4].imshow(pred_thresh, cmap="gray")
    axes[4].set_title("Predicted Mask (Thresholded)")
    axes[4].axis("off")

    plt.tight_layout()
    save_path = os.path.join(folder, f"debug_plot_{index}.png")
    fig.savefig(save_path)
    plt.close(fig)
