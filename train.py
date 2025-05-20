import albumentations as A
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from dataset import GroundSegDataset
import os
import argparse

def train(args):
    IMG_DIR  = "/Users/virginiaceccatelli/Documents/CompVisionMorbius/hospital_environment"
    TRAIN_SPLIT = "/Users/virginiaceccatelli/Documents/CompVisionMorbius/splits/train.txt"
    VAL_SPLIT = "/Users/virginiaceccatelli/Documents/CompVisionMorbius/splits/val.txt"
    MASK_DIR = "/Users/virginiaceccatelli/Documents/CompVisionMorbius/train_masks"

    # ----------------- define transforms -----------------
    train_transform = A.Compose([
        A.Resize(320, 320), # Resize to a fixed 320×320
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(),  # to 0–1 and mean/std
    ])

    val_transform = A.Compose([
        A.Resize(320, 320),
        A.Normalize(),
    ])

    # ----------------- training script -----------------
    # 1) Datasets & loaders: need to run command line prompt before this
    train_ds = GroundSegDataset(IMG_DIR,MASK_DIR,TRAIN_SPLIT,train_transform)
    val_ds   = GroundSegDataset(IMG_DIR,MASK_DIR,VAL_SPLIT,val_transform)

    for split_name, ds in [("train", train_ds), ("val", val_ds)]:
        print(f"Checking first 5 files in {split_name} split:")
        for name in ds.ids[:5]:
            mask_path = os.path.join(MASK_DIR, name + ".png")
            print(" ", mask_path, "→", os.path.isfile(mask_path))
        print()

    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    # 2) Model: U-Net with a MobileNetV2 encoder pretrained on ImageNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet("mobilenet_v2", encoder_weights="imagenet", classes=1, activation=None)
    model = model.to(device)

    # 3) Loss & optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ----------------- Training Loop -----------------
    for epoch in range(1, args.epochs + 1):
        # -- train --
        model.train()
        train_loss = 0.0
        for imgs, masks in train_dl:
            imgs, masks = imgs.to(device), masks.to(device) / 255.0
            logits = model(imgs)
            loss = criterion(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        # ----------------- validate -----------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs, masks = imgs.to(device), masks.to(device) / 255.0
                logits = model(imgs)
                val_loss += criterion(logits, masks).item()
        val_loss /= len(val_dl)

        print(f"[Epoch {epoch}/{args.epochs}]  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # -- checkpoint --
        ckpt_path = os.path.join(args.output_dir, f"unet_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)

    print("Training complete. Models saved to", args.output_dir)

def main():
    p = argparse.ArgumentParser(description="Train U-Net for ground segmentation")
    p.add_argument("--img_dir",     type=str, default="/path/to/images",
                   help="Folder with input images (no extension)")
    p.add_argument("--mask_dir",    type=str, default="/path/to/masks",
                   help="Folder with binary mask PNGs")
    p.add_argument("--train_split", type=str, default="splits/train.txt",
                   help="TXT file listing train IDs")
    p.add_argument("--val_split",   type=str, default="splits/val.txt",
                   help="TXT file listing validation IDs")
    p.add_argument("--img_ext",     type=str, default="png",
                   help="Extension of input images (png or jpg)")
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--output_dir",  type=str, default="checkpoints",
                   help="Where to save model checkpoints")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train(args)

if __name__ == "__main__":
    main()
