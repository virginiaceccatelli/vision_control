import csv
import pandas as pd
import matplotlib.pyplot as plt

# Sample metric lists from training run
train_loss = [
    0.6122, 0.3730, 0.2967, 0.2529, 0.1992, 0.1721, 0.1290, 0.1502, 0.1104, 0.1028,
    0.0864, 0.0895, 0.0801, 0.0707, 0.0636, 0.0676, 0.0672, 0.0688, 0.0592, 0.0477,
    0.0484, 0.0511, 0.0550, 0.0518, 0.0510, 0.0476, 0.0426, 0.0381, 0.0419, 0.0645,
    0.0648, 0.0610, 0.0480, 0.0695, 0.0571
]
val_loss = [
    7.3327, 3.5433, 3.0144, 0.2999, 0.2296, 0.2165, 0.1756, 0.1712, 0.1507, 0.1109,
    0.1036, 0.0939, 0.0936, 0.0933, 0.0972, 0.0936, 0.0864, 0.0858, 0.0797, 0.0856,
    0.0799, 0.0766, 0.0791, 0.0745, 0.0775, 0.0780, 0.0773, 0.0748, 0.0839, 0.0855,
    0.0880, 0.1738, 0.0917, 0.0997, 0.0683
]
epochs = list(range(1, len(train_loss) + 1))

csv_path = 'metrics.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'val_loss'])
    for e, tr, va in zip(epochs, train_loss, val_loss):
        writer.writerow([e, tr, va])
print(f"Metrics saved to {csv_path}")

df = pd.read_csv(csv_path)

plt.figure()
plt.plot(df['epoch'], df['train_loss'])
plt.plot(df['epoch'], df['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs. Validation Loss Over Epochs')
plt.legend(['Train Loss', 'Val Loss'])
plt.show()
