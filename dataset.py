import json, os
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
import cv2
import torch

# ----------------- convert LabelMe JSON to binary masks -----------------
def json_to_mask(json_path, out_shape):
    data = json.load(open(json_path))
    mask = Image.new("L", out_shape, 0)
    draw = ImageDraw.Draw(mask)
    for shape in data["shapes"]: # iterate over each annotation in json file
        if shape["label"] == "ground":
            pts = shape["points"] # polygon boundary: fill ground boundary with '1'
            draw.polygon([tuple(p) for p in pts], outline=1, fill=1)
    return np.array(mask, dtype=np.uint8) * 255  # 0: background or 255: ground


# ----------------- create PyTorch Dataset -----------------
class GroundSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, file_list, transforms=None):
        self.img_dir, self.mask_dir = img_dir, mask_dir
        self.ids = [line.strip() for line in open(file_list)]
        self.transforms = transforms

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]

        img_path = os.path.join(self.img_dir, f"{name}.png")
        mask_path = os.path.join(self.mask_dir, f"{name}.png")

        # Debug check
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"Mask not found:  {mask_path}")

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # returns H×W×3 (BGR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # returns H×W

        if self.transforms: # resizing to 320×320, flips, brightness changes, and normalization
            aug = self.transforms(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]

        # convert to tensor: ready for model
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask)[None].float()

        return img, mask
