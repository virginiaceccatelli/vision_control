import torch
import cv2
import numpy as np

model = torch.jit.load("unet_ground_plane.pt")
model.eval()

cap = cv2.VideoCapture("example_video.mp4")
if not cap.isOpened():
    raise IOError("Cannot open video")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_masked_video.mp4", fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    original = frame.copy()

    # Resize and preprocess
    resized = cv2.resize(frame, (320, 320))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    with torch.no_grad():
        output = model(tensor)
    output = output.squeeze().cpu().numpy()

    # Convert to binary mask
    mask = (output > 0.5).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask, (width, height))

    color_mask = cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(original, 0.7, color_mask, 0.3, 0)

    out.write(blended)

    cv2.imshow("Segmentation Overlay", blended)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
