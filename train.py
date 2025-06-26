import os
import argparse
import albumentations as A
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.metrics import IoU
import cv2
import matplotlib.pyplot as plt
from dataset import GroundSegDataset

IMG_DIR = "/Users/virginiaceccatelli/Documents/vision_control/CompVisionMorbius/hospital_environment"
TRAIN_SPLIT = "/Users/virginiaceccatelli/Documents/vision_control/CompVisionMorbius/splits/train.txt"
VAL_SPLIT = "/Users/virginiaceccatelli/Documents/vision_control/CompVisionMorbius/splits/val.txt"
MASK_DIR = "/Users/virginiaceccatelli/Documents/vision_control/CompVisionMorbius/train_masks"

def train(args):

    # ----------------- transforms config -----------------
    
    train_transform = A.Compose([ # image augmentation
        A.Resize(320, 320), # Resize to a fixed 320×320
        A.HorizontalFlip(p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.2, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # centered at 0
        A.OneOf([
            A.ToGray(p=1.0),
            A.ChannelDropout(p=1.0)
        ], p=0.1) # drop either one color channel or grayscale 10%
    ])
   

    val_transform = A.Compose([
        A.Resize(320, 320),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # ----------------- training script -----------------
    # Datasets & loaders: need to run command line prompt before this - pairs each ID with its image+mask
    train_ds = GroundSegDataset(IMG_DIR,MASK_DIR,TRAIN_SPLIT,train_transform)
    val_ds   = GroundSegDataset(IMG_DIR,MASK_DIR,VAL_SPLIT,val_transform)

    # PyTorch DataLoader
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model: U-Net with a MobileNetV2 encoder pretrained on ImageNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda: NVIDIA parallel processing GPU (not available to me but for future versions)
    model = smp.Unet("mobilenet_v2", encoder_weights="imagenet", classes=1, activation=None) # single channel: one score x pixel
    model = model.to(device)

    # Loss & optimizer
    criterion = nn.BCEWithLogitsLoss() # binary cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3) # Adam: stochastic gradient with adaptive learning rates

    # ----------------- Training Loop -----------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_dl:
            imgs, masks = imgs.to(device), masks.to(device) / 255.0
            logits = model(imgs)
            loss = criterion(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() # Update weights
            train_loss += loss.item()
        train_loss /= len(train_dl) # average train loss

        # ----------------- Validate Loop -----------------
        model.eval()
        val_loss = 0.0
        iou = IoU(threshold=0.5) # Area of Intersection / Area of Union:
        total_iou = 0.0
        num_batches = 0
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs, masks = imgs.to(device), masks.to(device) / 255.0
                logits = model(imgs)
                val_loss += criterion(logits, masks).item()
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                batch_iou = iou(preds, masks)
                total_iou += batch_iou.item()
                num_batches += 1
        val_loss /= len(val_dl) # average val loss
        mean_iou = total_iou / num_batches

        print(f"[Epoch {epoch}/{args.epochs}]  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_iou={mean_iou:.4f}")

        # ----------------- Checkpoints -----------------
        ckpt_path = os.path.join(args.output_dir, f"unet_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)

    print("Training complete. Models saved to", args.output_dir)

def visualize_predictions(args, model):
    model.eval() # evaluation mode
    device = next(model.parameters()).device

    best_ckpt = os.path.join(args.output_dir, f"unet_epoch{args.best_epoch}.pt")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))

    # Use the same val_transform
    transform = A.Compose([
        A.Resize(320, 320),
        A.Normalize(),
    ])

    val_ds = GroundSegDataset(args.img_dir, args.mask_dir, args.val_split, transform) # val dataset
    for name in val_ds.ids[:args.num_visualize]:
        # original image
        img_path = os.path.join(args.img_dir, f"{name}.{args.img_ext}")
        img = cv2.imread(img_path, cv2.IMREAD_COLOR) # read as BGR -> convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0, _ = img_rgb.shape

        aug = transform(image=img_rgb)
        inp = torch.from_numpy(aug["image"]).permute(2, 0, 1).unsqueeze(0).float().to(device)

        # Inference: training-oriented output -> 2D probability map -> numpy
        with torch.no_grad(): # no gradients needed for inference
            logits = model(inp)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()

        mask = cv2.resize(prob, (w0, h0)) # original image size
        mask_bin = (mask > 0.5).astype("uint8") # banker's rounding

        # Overlay
        plt.figure(figsize=(6,6))
        plt.imshow(img_rgb) # original RGB image
        plt.imshow(mask_bin, alpha=0.4, cmap="jet") # predicted mask overlay
        plt.title(f"Prediction for {name}")
        plt.axis("off")
        plt.show()

# ----------------- Argument Parser -----------------

def main():
    p = argparse.ArgumentParser(description="Train U-Net for ground segmentation")
    p.add_argument("-img_dir",     type=str, default="/Users/virginiaceccatelli/Documents/vision_control/CompVisionMorbius/hospital_environment")
    p.add_argument("-mask_dir",    type=str, default="/Users/virginiaceccatelli/Documents/vision_control/CompVisionMorbius/hospital_environment/hospital_json")
    p.add_argument("-train_split", type=str, default="/Users/virginiaceccatelli/Documents/vision_control/CompVisionMorbius/splits/train.txt")
    p.add_argument("-val_split",   type=str, default="/Users/virginiaceccatelli/Documents/vision_control/CompVisionMorbius/splits/val.txt")
    p.add_argument("-img_ext",     type=str, default="png")
    p.add_argument("-mode", type=str, default="train", choices=["train", "infer"])
    p.add_argument("-batch_size",  type=int, default=8)
    p.add_argument("-epochs",      type=int, default=40)
    p.add_argument("-lr",          type=float, default=1e-4) # learning rate
    p.add_argument("-output_dir",  type=str, default="checkpoints_new")
    p.add_argument("-best_epoch", type=int, default=47) # best epoch to visualize
    p.add_argument("-num_visualize", type=int, default=5) # num pictures to visualize
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "train":
        train(args)
    else:  # infer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = smp.Unet("mobilenet_v2",
                         encoder_weights="imagenet",
                         classes=1, activation=None).to(device)
        visualize_predictions(args, model)

if __name__ == "__main__":
    main()
