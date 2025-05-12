import albumentations as A
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from dataset import GroundSegDataset

# ----------------- define transforms -----------------
train_transform = A.Compose([
    A.Resize(320, 320),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(),  # to 0–1 and mean/std
])

val_transform = A.Compose([
    A.Resize(320, 320),
    A.Normalize(),
])

# ----------------- training script -----------------
# 1) Datasets & loaders
train_ds = GroundSegDataset("data/images","data/masks","data/splits/train.txt",train_transform)
val_ds   = GroundSegDataset("data/images","data/masks","data/splits/val.txt",val_transform)
train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
val_dl   = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)

# 2) Model
model = smp.Unet("mobilenet_v2", encoder_weights="imagenet", classes=1, activation=None)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# 3) Loss & optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4) Quick overfit test (1 batch)
imgs, masks = next(iter(train_dl))
logits = model(imgs.to(model.device))
loss = criterion(logits, masks.to(model.device)/255.0)
print("Initial loss:", loss.item()); exit()

# Once that runs without error, remove the exit() and implement the full epoch loop.
