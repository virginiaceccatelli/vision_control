import csv
import pandas as pd
import matplotlib.pyplot as plt

csv_path = 'metrics.csv'
df = pd.read_csv(csv_path)

# normal
plt.figure()
plt.plot(df['epoch'], df['train_loss'], color='blue')
plt.plot(df['epoch'], df['val_loss'], color='green')
plt.plot(df['epoch'], df['val_iou'], color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs. Validation Loss Over Epochs')
plt.legend(['Train Loss', 'Val Loss', 'Val IoU'])
plt.show()

# zoomed in
plt.figure()
plt.plot(df['epoch'], df['train_loss'], color='blue')
plt.plot(df['epoch'], df['val_loss'], color='green')
plt.plot(df['epoch'], df['val_iou'], color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs. Validation Loss Over Epochs')
plt.legend(['Train Loss', 'Val Loss', 'Val IoU'])
ax = plt.gca()
ax.set_ylim([0, 1])
plt.show()
