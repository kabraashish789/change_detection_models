import numpy as np
import re
from torch.utils.data import Dataset
from PIL import Image
import os
import cv2

class CDDataset(Dataset):
    """Change Detection Dataset"""

    def __init__(self, txt_file, transform=None):
        """
        Args:
            txt_file (string): Path to the file containing image paths.
            transform (callable, optional): Optional transform to be applied.
        """
        self.transform = transform
        base_dir = "/home/foj-ak/data/"

        # Read image paths from txt_file
        with open(txt_file, "r") as f:
            self.image_paths = [os.path.join(base_dir, line.strip()) for line in f]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Get current image path
        img_path_t2 = self.image_paths[idx]
        
        # Construct the previous image path
        img_path_t1 = self._get_previous_image(img_path_t2)
        
        # For debugging: print image paths
        #print(f"Current image path (t2): {img_path_t2}")
        #print(f"Previous image path (t1): {img_path_t1}")
        
        # Load images
        image_t2 = np.array(Image.open(img_path_t2).convert("RGB"))
        image_t1 = np.array(Image.open(img_path_t1).convert("RGB"))

        # Concatenate along channel dimension to create a 6-channel input
        image = np.concatenate([image_t1, image_t2], axis=-1)  # Shape: (H, W, 6)

        # Load mask
        mask_path = self._get_mask_path(img_path_t2)
        
        # For debugging: print mask path
        #print(f"Mask path: {mask_path}")
        
        mask_image = Image.open(mask_path)

        # Convert mask to binary
        #mask = np.array(mask_image.convert('L'), dtype=np.float32)

        # Process the mask based on dataset type.
        if "synth-iter" in img_path_t2:
            # For synth-iter, the mask is in color.
            # Use HSV conversion and thresholding to capture different shades of red.
            mask_image = mask_image.convert("RGB")  # Ensure 3 channels
            mask_array = np.array(mask_image)
            mask_hsv = cv2.cvtColor(mask_array, cv2.COLOR_RGB2HSV)

            # Define thresholds for red.
            # These values can be adjusted depending on the shade variations.
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])

            # Create masks for each range
            mask1 = cv2.inRange(mask_hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(mask_hsv, lower_red2, upper_red2)
            # Combine masks via bitwise OR
            mask_binary = cv2.bitwise_or(mask1, mask2)

            # Convert to binary float mask (with values of 0 or 1)
            mask = (mask_binary.astype(np.float32) / 255.0)

        elif "real" in img_path_t2:
            # For real data, the mask images are already binary.
            # Convert using grayscale conversion and a threshold.
            mask = np.array(mask_image.convert('L'), dtype=np.float32)
            mask = (mask > 0).astype(np.float32)
        else:
            # Default case if the dataset type does not match.
            mask = np.array(mask_image.convert('L'), dtype=np.float32)
            mask = (mask > 0).astype(np.float32)

        # Apply transformations, if any
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

    def _get_previous_image(self, img_path):
        """Constructs the previous image path according to dataset type."""
        if 'synth-iter' in img_path:
            # For 'synth-iter' data
            # Example path: /home/.../synth-iter/6/2/RGB/2_0.png
            path_parts = img_path.split(os.sep)
            try:
                # Identify indices
                synth_idx = path_parts.index('synth-iter')
                # Subfolder index is synth_idx + 2 (after 'synth-iter' and its immediate subfolder)
                subfolder_idx = synth_idx + 2
                current_subfolder = int(path_parts[subfolder_idx])
                prev_subfolder = current_subfolder - 1
                if prev_subfolder < 0:
                    print(f"No previous subfolder for {img_path}, using current image.")
                    return img_path
                # Update the subfolder in the path
                path_parts[subfolder_idx] = str(prev_subfolder)
                prev_img_path = os.sep.join(path_parts)
                if os.path.exists(prev_img_path):
                    return prev_img_path
                else:
                    print(f"Previous image {prev_img_path} not found for {img_path}, using current image.")
                    return img_path
            except ValueError as e:
                print(f"Error processing {img_path}: {e}. Using current image.")
                return img_path
        elif 'real' in img_path:
            # For 'real' data
            # Example path: /home/.../real/train/249/left_rgb/3_left.png
            match = re.search(r'(\d+)_left\.png$', img_path)
            if match:
                num = int(match.group(1))
                prev_num = num - 1
                if prev_num < 0:
                    print(f"No previous image for {img_path}, using current image.")
                    return img_path
                prev_img_name = f"{prev_num}_left.png"
                prev_img_path = re.sub(r'\d+_left\.png$', prev_img_name, img_path)
                if os.path.exists(prev_img_path):
                    return prev_img_path
                else:
                    print(f"Previous image {prev_img_path} not found for {img_path}, using current image.")
                    return img_path
            else:
                print(f"Filename format not matched for {img_path}, using current image.")
                return img_path
        else:
            # Default behavior
            print(f"Dataset type not recognized in path {img_path}, using current image.")
            return img_path

    def _get_mask_path(self, img_path):
        """Generates the corresponding mask path."""
        if 'synth-iter' in img_path:
            # For 'synth-iter' data
            # Replace 'RGB' with 'Instance' and modify filename
            path_parts = img_path.split(os.sep)
            path_parts[-2] = 'Instance'  # Replace directory 'RGB' with 'Instance'
            filename = path_parts[-1]
            # Filename format: '2_0.png' -> Mask filename: '2.png'
            match = re.match(r'(\d+)_\d+\.png$', filename)
            if match:
                mask_filename = f"{match.group(1)}.png"
                path_parts[-1] = mask_filename
            else:
                print(f"Filename format not matched for mask in {img_path}, using current image filename.")
                # Fallback: remove everything after first '_'
                mask_filename = filename.split('_')[0] + '.png'
                path_parts[-1] = mask_filename
            mask_path = os.sep.join(path_parts)
            return mask_path
        elif 'real' in img_path:
            # For 'real' data
            # Replace 'left_rgb' with 'matting'
            mask_path = img_path.replace('left_rgb', 'matting')
            return mask_path
        else:
            # Default behavior
            print(f"Dataset type not recognized in path {img_path}, cannot generate mask path.")
            return img_path