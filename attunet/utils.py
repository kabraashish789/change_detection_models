import torch
import torchvision
import os
import matplotlib.pyplot as plt
import numpy as np

def save_checkpoint(state, filename="best_model.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def check_accuracy(loader, model, device, save_folder="debug_plots/"):
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
            y = y.to(device).unsqueeze(1)
            preds = model(x)
            preds_sigmoid = torch.sigmoid(preds)
            preds_thresh = (preds_sigmoid > 0.75).float()

            num_correct += int((preds_thresh == y).sum().item())
            num_pixels += torch.numel(preds_thresh)

            intersection = (preds_thresh * y).sum()
            union = preds_thresh.sum() + y.sum()
            dice = (2 * intersection) / (union + 1e-8) if union > 0 else 0
            dice_score += dice.item()

            # IoU calculation
            tp = (preds_thresh * y).sum().item()
            fp = ((preds_thresh == 1) & (y == 0)).sum().item()
            fn = ((preds_thresh == 0) & (y == 1)).sum().item()

            iou = tp / (tp + fp + fn + 1e-8) if (tp + fp + fn) > 0 else 0
            iou_score += iou

            # Save debug plots
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

    accuracy = num_correct / num_pixels * 100
    avg_dice_score = dice_score / len(loader)
    avg_iou = iou_score / len(loader)

    print(f"Got {num_correct}/{num_pixels} with acc {accuracy:.2f}%")
    print(f"Dice score: {avg_dice_score:.4f}")
    print(f"IoU score: {avg_iou:.4f}")

    model.train()
    return avg_dice_score, avg_iou

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    os.makedirs(folder, exist_ok=True)

    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")

        y = y.unsqueeze(1).float()
        torchvision.utils.save_image(y, f"{folder}/{idx}.png")

    model.train()

def save_debug_plot(input_image, ground_truth, prediction, pred_thresh, index=0, folder="debug_plots/", normalization_mean=None, normalization_std=None):
    """
    Saves a plot of the input images (before and after), ground truth mask, predicted mask, and thresholded prediction.
    """

    input_image = input_image.cpu().numpy()
    ground_truth = ground_truth.cpu().squeeze().numpy()
    prediction = prediction.cpu().squeeze().numpy()
    pred_thresh = pred_thresh.cpu().squeeze().numpy()

    # Split the concatenated input image into before and after images
    num_channels = input_image.shape[0]  # Should be 6 for two RGB images
    channels_per_image = num_channels // 2

    before_image = input_image[:channels_per_image, :, :]
    after_image = input_image[channels_per_image:, :, :]

    before_image = np.transpose(before_image, (1, 2, 0))
    after_image = np.transpose(after_image, (1, 2, 0))

    if normalization_mean is not None and normalization_std is not None:
        before_image = before_image * normalization_std + normalization_mean
        after_image = after_image * normalization_std + normalization_mean

    before_image = np.clip(before_image, 0, 1)
    after_image = np.clip(after_image, 0, 1)

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
